// Native C + CUDA reference for Formal's corrected order-4 2D
// advection-diffusion workspace.  The CPU path is the FP32 numerical oracle;
// see README.md for provenance and PERFORMANCE.md for the kernel variants.

#include <cuda_runtime.h>

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const float FA_PI = 3.14159265358979323846f;
static const float FA_DIFFUSION = 0.5f;
static const float FA_SIGMA = 3.14159265358979323846f / 8.0f;
static const float FA_X0 = -3.14159265358979323846f / 2.0f;
static const float FA_Y0 = -3.14159265358979323846f / 2.0f;
static const char *FA_FORMAL_51_COORDS[] = {
    "-3.141592741013", "-3.079992771149", "-2.956793069839", "-2.833593368530",
    "-2.710393667221", "-2.587193965912", "-2.463994264603", "-2.340794563293",
    "-2.217594623566", "-2.094395160675", "-1.971195340157", "-1.847995638847",
    "-1.724795937538", "-1.601596117020", "-1.478396415710", "-1.355196714401",
    "-1.231997013092", "-1.108797311783", "-0.9855976104736", "-0.8623979091644",
    "-0.7391982078552", "-0.6159985065460", "-0.4927988052368", "-0.3695991039276",
    "-0.2463994026184", "-0.1231994628906", "0.2384185791016E-06", "0.1231999397278",
    "0.2463996410370", "0.3695993423462", "0.4927990436554", "0.6159987449646",
    "0.7391984462738", "0.8623981475830", "0.9855980873108", "1.108797550201",
    "1.231997489929", "1.355196952820", "1.478396892548", "1.601596355438",
    "1.724796295166", "1.847995758057", "1.971195697784", "2.094395160675",
    "2.217595100403", "2.340794563293", "2.463994503021", "2.587193965912",
    "2.710393905640", "2.833593845367", "2.956793308258", "3.079993247986",
    "3.141592741013",
};

typedef struct {
    uint32_t nx;
    uint32_t ny;
    uint32_t steps;
    float dt;
} fa_config_t;

typedef struct {
    size_t scalar_len;
    size_t xface_len;
    size_t yface_len;
    float *base;
    float *stage[2];
    float *accum;
    float *vx;
    float *vy;
} fa_fields_t;

typedef enum {
    FA_KERNEL_SPLIT,
    FA_KERNEL_FUSED,
    FA_KERNEL_TILED,
    FA_KERNEL_PREFETCH,
    FA_KERNEL_REGY,
    FA_KERNEL_REGYF,
    FA_KERNEL_STREAM,
    FA_KERNEL_STEP,
    FA_KERNEL_PAIR,
    FA_KERNEL_COUNT_
} fa_kernel_t;

static const char *FA_KERNEL_NAMES[FA_KERNEL_COUNT_] = {
    "split",
    "fused",
    "tiled",
    "prefetch",
    "regy",
    "regyf",
    "stream",
    "step",
    "pair",
};

typedef struct {
    bool cuda;
    bool dump_csv;
    bool dump_bin;
    char *input_path;
    char *output_path;
    char *binary_output_path;
    fa_config_t cfg;
    bool dt_supplied;
    uint32_t block_x;
    uint32_t block_y;
    bool block_supplied;
    uint32_t tile_x;
    uint32_t tile_y;
    fa_kernel_t kernel;
    bool graph;
} fa_options_t;

typedef struct {
    double init_seconds;
    double solve_seconds;
    double gpu_busy;
    bool final_all_finite;
} fa_run_stats_t;

static double fa_now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void fa_die(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static void fa_dief(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static void fa_cuda_check(cudaError_t err, const char *what)
{
    if (err != cudaSuccess) {
        fa_dief("CUDA error in %s: %s", what, cudaGetErrorString(err));
    }
}

static size_t fa_checked_mul_size(size_t a, size_t b, const char *what)
{
    if (a != 0 && b > SIZE_MAX / a) {
        fa_dief("%s size overflow", what);
    }
    return a * b;
}

static size_t fa_checked_add_size(size_t a, size_t b, const char *what)
{
    if (b > SIZE_MAX - a) {
        fa_dief("%s size overflow", what);
    }
    return a + b;
}

static const char *fa_center_coord_text(size_t i, uint32_t n)
{
    if (n == 51 && i < sizeof(FA_FORMAL_51_COORDS) / sizeof(FA_FORMAL_51_COORDS[0])) {
        return FA_FORMAL_51_COORDS[i];
    }
    return NULL;
}

static float fa_center_coord(size_t i, uint32_t n, float min, float max)
{
    const char *text = fa_center_coord_text(i, n);
    if (text) {
        return strtof(text, NULL);
    }
    volatile float dx = (max - min) / (float)n;
    if (i == 0) {
        return min;
    }
    if (i == (size_t)n + 1) {
        return max;
    }
    volatile float base = min + 0.5f * dx;
    volatile float offset = ((float)i - 1.0f) * dx;
    return base + offset;
}

static float fa_face_coord(size_t i, uint32_t n, float min, float max)
{
    volatile float dx = (max - min) / (float)n;
    if (i == 0) {
        return min;
    }
    if (i == (size_t)n) {
        return max;
    }
    return min + (float)i * dx;
}

static size_t fa_scalar_index(size_t i, size_t j, uint32_t nx)
{
    return i + (size_t)(nx + 2) * j;
}

static size_t fa_xface_index(size_t i, size_t j, uint32_t nx)
{
    return i + (size_t)(nx + 1) * j;
}

static size_t fa_yface_index(size_t i, size_t j, uint32_t nx)
{
    return i + (size_t)(nx + 2) * j;
}

static float fa_scalar_init(float x, float y)
{
    const float dx = x - FA_X0;
    const float dy = y - FA_Y0;
    const float r = sqrtf(dx * dx + dy * dy);
    return expf(-((r * r) / (2.0f * FA_SIGMA * FA_SIGMA)));
}

static float fa_velocity_x(float x, float y)
{
    return 10.0f * sinf(x) * cosf(y);
}

static float fa_velocity_y(float x, float y)
{
    return -10.0f * cosf(x) * sinf(y);
}

static void fa_fields_free(fa_fields_t *fields)
{
    free(fields->base);
    free(fields->stage[0]);
    free(fields->stage[1]);
    free(fields->accum);
    free(fields->vx);
    free(fields->vy);
    memset(fields, 0, sizeof(*fields));
}

static void fa_fields_alloc(fa_fields_t *fields, const fa_config_t *cfg)
{
    memset(fields, 0, sizeof(*fields));
    fields->scalar_len = fa_checked_mul_size((size_t)cfg->nx + 2,
        (size_t)cfg->ny + 2, "scalar field");
    fields->xface_len = fa_checked_mul_size((size_t)cfg->nx + 1,
        (size_t)cfg->ny + 2, "x-face field");
    fields->yface_len = fa_checked_mul_size((size_t)cfg->nx + 2,
        (size_t)cfg->ny + 1, "y-face field");

    fields->base = (float *)calloc(fields->scalar_len, sizeof(float));
    fields->stage[0] = (float *)calloc(fields->scalar_len, sizeof(float));
    fields->stage[1] = (float *)calloc(fields->scalar_len, sizeof(float));
    fields->accum = (float *)calloc(fields->scalar_len, sizeof(float));
    fields->vx = (float *)calloc(fields->xface_len, sizeof(float));
    fields->vy = (float *)calloc(fields->yface_len, sizeof(float));
    if (!fields->base || !fields->stage[0] || !fields->stage[1] ||
        !fields->accum || !fields->vx || !fields->vy) {
        fa_die("cannot allocate fields");
    }
}

static void fa_generate_analytic(const fa_config_t *cfg, fa_fields_t *fields)
{
    for (size_t j = 0; j < (size_t)cfg->ny + 2; ++j) {
        const float y = fa_center_coord(j, cfg->ny, -FA_PI, FA_PI);
        for (size_t i = 0; i < (size_t)cfg->nx + 2; ++i) {
            const float x = fa_center_coord(i, cfg->nx, -FA_PI, FA_PI);
            const size_t idx = fa_scalar_index(i, j, cfg->nx);
            fields->base[idx] = fa_scalar_init(x, y);
            fields->stage[0][idx] = fields->base[idx];
            fields->stage[1][idx] = fields->base[idx];
        }
    }

    for (size_t j = 0; j < (size_t)cfg->ny + 2; ++j) {
        const float y = fa_center_coord(j, cfg->ny, -FA_PI, FA_PI);
        for (size_t i = 0; i < (size_t)cfg->nx + 1; ++i) {
            const float x = fa_face_coord(i, cfg->nx, -FA_PI, FA_PI);
            fields->vx[fa_xface_index(i, j, cfg->nx)] = fa_velocity_x(x, y);
        }
    }

    for (size_t j = 0; j < (size_t)cfg->ny + 1; ++j) {
        const float y = fa_face_coord(j, cfg->ny, -FA_PI, FA_PI);
        for (size_t i = 0; i < (size_t)cfg->nx + 2; ++i) {
            const float x = fa_center_coord(i, cfg->nx, -FA_PI, FA_PI);
            fields->vy[fa_yface_index(i, j, cfg->nx)] = fa_velocity_y(x, y);
        }
    }
}

static void fa_validate_cfg(fa_config_t *cfg, bool dt_supplied)
{
    if (cfg->nx < 9 || cfg->ny < 9) {
        fa_die("cells must be at least 9 in each dimension");
    }
    if (cfg->steps == 0) {
        fa_die("steps must be positive");
    }
    if (!(isfinite(cfg->dt) && cfg->dt > 0.0f)) {
        fa_die("dt must be finite and positive");
    }
    if (!dt_supplied) {
        const float sx = 51.0f / (float)cfg->nx;
        const float sy = 51.0f / (float)cfg->ny;
        const float scale = fminf(sx, sy);
        cfg->dt = 1.0e-4f * scale * scale;
    }
}

static uint32_t fa_parse_u32_strict(const char *raw, const char *flag_name)
{
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || value == 0 ||
        value > UINT32_MAX) {
        fa_dief("%s parse error: expected positive unsigned integer", flag_name);
    }
    return (uint32_t)value;
}

// Host fields, plus the CUDA device buffers (four scalar fields, both
// velocities and the four face scratch fields of the split variant).
static void fa_validate_working_set(const fa_config_t *cfg)
{
    const size_t scalar_len = fa_checked_mul_size((size_t)cfg->nx + 2,
        (size_t)cfg->ny + 2, "scalar field");
    const size_t xface_len = fa_checked_mul_size((size_t)cfg->nx + 1,
        (size_t)cfg->ny + 2, "x-face field");
    const size_t yface_len = fa_checked_mul_size((size_t)cfg->nx + 2,
        (size_t)cfg->ny + 1, "y-face field");

    const size_t scalar_bytes = fa_checked_mul_size(scalar_len, sizeof(float),
        "scalar field bytes");
    const size_t xface_bytes = fa_checked_mul_size(xface_len, sizeof(float),
        "x-face field bytes");
    const size_t yface_bytes = fa_checked_mul_size(yface_len, sizeof(float),
        "y-face field bytes");
    const size_t host_fields_bytes = fa_checked_add_size(
        fa_checked_add_size(
            fa_checked_mul_size(scalar_bytes, 4, "host scalar buffers"),
            xface_bytes, "host fields"),
        yface_bytes, "host fields");
    const size_t device_buffers_bytes = fa_checked_add_size(
        fa_checked_add_size(
            fa_checked_mul_size(scalar_bytes, 4, "device scalar buffers"),
            fa_checked_mul_size(xface_bytes, 3, "device x-face buffers"),
            "device buffers"),
        fa_checked_mul_size(yface_bytes, 3, "device y-face buffers"),
        "device buffers");
    const size_t total_bytes = fa_checked_add_size(host_fields_bytes,
        device_buffers_bytes, "working set");
    const size_t max_bytes = (size_t)4u * 1024u * 1024u * 1024u;
    if (total_bytes > max_bytes) {
        fa_die("estimated working set exceeds 4 GiB");
    }
}

static float fa_parse_float_strict(const char *raw, const char *flag_name)
{
    errno = 0;
    char *end = NULL;
    float value = strtof(raw, &end);
    if (errno != 0 || end == raw || *end != '\0' || !isfinite(value)) {
        fa_dief("%s parse error: expected finite real", flag_name);
    }
    return value;
}

static void fa_read_raw_input(const char *path, fa_config_t *cfg, fa_fields_t *fields)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fa_dief("cannot open input file: %s", path);
    }

    char line[256];
    if (!fgets(line, sizeof(line), f) || strncmp(line, "FADV2D1", 7) != 0) {
        fclose(f);
        fa_die("invalid input header");
    }
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        fa_die("missing input dimensions");
    }
    if (sscanf(line, "%u %u %u %f", &cfg->nx, &cfg->ny, &cfg->steps, &cfg->dt) != 4) {
        fclose(f);
        fa_die("cannot parse input dimensions");
    }
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        fa_die("missing input mode");
    }
    char mode[64] = {0};
    if (sscanf(line, "mode %63s", mode) != 1) {
        fclose(f);
        fa_die("cannot parse input mode");
    }
    fa_validate_cfg(cfg, true);
    fa_validate_working_set(cfg);
    fa_fields_alloc(fields, cfg);
    if (strcmp(mode, "raw") == 0) {
        if (fread(fields->base, sizeof(float), fields->scalar_len, f) !=
            fields->scalar_len ||
            fread(fields->vx, sizeof(float), fields->xface_len, f) !=
            fields->xface_len ||
            fread(fields->vy, sizeof(float), fields->yface_len, f) !=
            fields->yface_len) {
            fclose(f);
            fa_die("cannot read raw field payload");
        }
        memcpy(fields->stage[0], fields->base, fields->scalar_len * sizeof(float));
        memcpy(fields->stage[1], fields->base, fields->scalar_len * sizeof(float));
    } else if (strcmp(mode, "analytic") == 0) {
        fa_generate_analytic(cfg, fields);
    } else {
        fclose(f);
        fa_die("unknown input mode");
    }
    fclose(f);
}

static void fa_write_csv(const char *path, const fa_config_t *cfg,
    const fa_fields_t *fields)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fa_dief("cannot open output file: %s", path);
    }
    fprintf(f, "x,y,scalar\n");
    for (size_t j = 0; j < (size_t)cfg->ny + 2; ++j) {
        const char *y_text = fa_center_coord_text(j, cfg->ny);
        const float y = y_text ? strtof(y_text, NULL) :
            fa_center_coord(j, cfg->ny, -FA_PI, FA_PI);
        for (size_t i = 0; i < (size_t)cfg->nx + 2; ++i) {
            const char *x_text = fa_center_coord_text(i, cfg->nx);
            const float x = x_text ? strtof(x_text, NULL) :
                fa_center_coord(i, cfg->nx, -FA_PI, FA_PI);
            const size_t idx = fa_scalar_index(i, j, cfg->nx);
            if (x_text && y_text) {
                fprintf(f, "%s,%s,%.13g\n", x_text, y_text,
                    (double)fields->base[idx]);
            } else {
                fprintf(f, "%.12f,%.12f,%.13g\n", (double)x, (double)y,
                    (double)fields->base[idx]);
            }
        }
        if (j + 1 < (size_t)cfg->ny + 2) {
            fputc('\n', f);
        }
    }
    fclose(f);
}

static void fa_write_binary(const char *path, const fa_fields_t *fields)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        fa_dief("cannot open output file: %s", path);
    }
    if (fwrite(fields->base, sizeof(float), fields->scalar_len, f) !=
        fields->scalar_len) {
        fclose(f);
        fa_die("cannot write binary field payload");
    }
    fclose(f);
}

static void fa_check_finite(const float *data, size_t n, const char *name)
{
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(data[i])) {
            fa_dief("%s contains non-finite values", name);
        }
    }
}

// ---------------------------------------------------------------------------
// FP32 CPU oracle (unchanged method from the Metal reference).
// ---------------------------------------------------------------------------

static inline float fa_grad_face_line(const float *line, size_t stride, size_t n, size_t i,
    float inv_h)
{
    if (i == 0) {
        return (((-352.0f / 105.0f) * inv_h) * line[0 * stride] +
                ((35.0f / 8.0f) * inv_h) * line[1 * stride] +
                ((-35.0f / 24.0f) * inv_h) * line[2 * stride] +
                ((21.0f / 40.0f) * inv_h) * line[3 * stride] +
                ((-5.0f / 56.0f) * inv_h) * line[4 * stride]);
    }
    if (i == 1) {
        return (((16.0f / 105.0f) * inv_h) * line[0 * stride] +
                ((-31.0f / 24.0f) * inv_h) * line[1 * stride] +
                ((29.0f / 24.0f) * inv_h) * line[2 * stride] +
                ((-3.0f / 40.0f) * inv_h) * line[3 * stride] +
                ((1.0f / 168.0f) * inv_h) * line[4 * stride]);
    }
    if (i + 3 < n) {
        return (((1.0f / 24.0f) * inv_h) * line[(i - 1) * stride] +
                ((-9.0f / 8.0f) * inv_h) * line[i * stride] +
                ((9.0f / 8.0f) * inv_h) * line[(i + 1) * stride] +
                ((-1.0f / 24.0f) * inv_h) * line[(i + 2) * stride]);
    }
    if (i == n - 3) {
        return (((-1.0f / 168.0f) * inv_h) * line[(n - 5) * stride] +
                ((3.0f / 40.0f) * inv_h) * line[(n - 4) * stride] +
                ((-29.0f / 24.0f) * inv_h) * line[(n - 3) * stride] +
                ((31.0f / 24.0f) * inv_h) * line[(n - 2) * stride] +
                ((-16.0f / 105.0f) * inv_h) * line[(n - 1) * stride]);
    }
    return (((5.0f / 56.0f) * inv_h) * line[(n - 5) * stride] +
            ((-21.0f / 40.0f) * inv_h) * line[(n - 4) * stride] +
            ((35.0f / 24.0f) * inv_h) * line[(n - 3) * stride] +
            ((-35.0f / 8.0f) * inv_h) * line[(n - 2) * stride] +
            ((352.0f / 105.0f) * inv_h) * line[(n - 1) * stride]);
}

static inline float fa_interp_c2f_line(const float *line, size_t stride, size_t n, size_t i)
{
    if (i == 0) {
        return line[0 * stride];
    }
    if (i == 1) {
        return (-16.0f / 112.0f) * line[0 * stride] +
               (70.0f / 112.0f) * line[1 * stride] +
               (70.0f / 112.0f) * line[2 * stride] +
               (-14.0f / 112.0f) * line[3 * stride] +
               (2.0f / 112.0f) * line[4 * stride];
    }
    if (i + 3 < n) {
        return (-7.0f / 112.0f) * line[(i - 1) * stride] +
               (63.0f / 112.0f) * line[i * stride] +
               (63.0f / 112.0f) * line[(i + 1) * stride] +
               (-7.0f / 112.0f) * line[(i + 2) * stride];
    }
    if (i == n - 3) {
        return (2.0f / 112.0f) * line[(n - 5) * stride] +
               (-14.0f / 112.0f) * line[(n - 4) * stride] +
               (70.0f / 112.0f) * line[(n - 3) * stride] +
               (70.0f / 112.0f) * line[(n - 2) * stride] +
               (-16.0f / 112.0f) * line[(n - 1) * stride];
    }
    if (i == n - 2) {
        return line[(n - 1) * stride];
    }
    fa_die("invalid center-to-face stencil index");
    return 0.0f;
}

static inline float fa_div_face_line(const float *line, size_t stride, size_t n, size_t i,
    float inv_h)
{
    if (i == 0) {
        return (((-11.0f / 12.0f) * inv_h) * line[0 * stride] +
                ((17.0f / 24.0f) * inv_h) * line[1 * stride] +
                ((3.0f / 8.0f) * inv_h) * line[2 * stride] +
                ((-5.0f / 24.0f) * inv_h) * line[3 * stride] +
                ((1.0f / 24.0f) * inv_h) * line[4 * stride]);
    }
    if (i + 2 < n) {
        return (((1.0f / 24.0f) * inv_h) * line[(i - 1) * stride] +
                ((-9.0f / 8.0f) * inv_h) * line[i * stride] +
                ((9.0f / 8.0f) * inv_h) * line[(i + 1) * stride] +
                ((-1.0f / 24.0f) * inv_h) * line[(i + 2) * stride]);
    }
    return (((-1.0f / 24.0f) * inv_h) * line[(n - 5) * stride] +
            ((5.0f / 24.0f) * inv_h) * line[(n - 4) * stride] +
            ((-3.0f / 8.0f) * inv_h) * line[(n - 3) * stride] +
            ((-17.0f / 24.0f) * inv_h) * line[(n - 2) * stride] +
            ((11.0f / 12.0f) * inv_h) * line[(n - 1) * stride]);
}

static void fa_rhs_cpu(const fa_config_t *cfg, const fa_fields_t *fields,
    const float *stage_in, float *rhs_out)
{
    const float inv_dx = (float)cfg->nx / (2.0f * FA_PI);
    const float inv_dy = (float)cfg->ny / (2.0f * FA_PI);
    const size_t sx = (size_t)cfg->nx + 2;
    const size_t sy = (size_t)cfg->ny + 2;

    float *diff_x = (float *)calloc(fields->xface_len, sizeof(float));
    float *adv_x = (float *)calloc(fields->xface_len, sizeof(float));
    float *diff_y = (float *)calloc(fields->yface_len, sizeof(float));
    float *adv_y = (float *)calloc(fields->yface_len, sizeof(float));
    if (!diff_x || !adv_x || !diff_y || !adv_y) {
        fa_die("cannot allocate RHS scratch");
    }

    for (size_t j = 0; j < sy; ++j) {
        const float *line = stage_in + j * sx;
        for (size_t i = 0; i < sx - 1; ++i) {
            const size_t idx = fa_xface_index(i, j, cfg->nx);
            diff_x[idx] = FA_DIFFUSION * fa_grad_face_line(line, 1, sx, i, inv_dx);
            adv_x[idx] = fields->vx[idx] * fa_interp_c2f_line(line, 1, sx, i);
        }
    }

    for (size_t i = 0; i < sx; ++i) {
        const float *line = stage_in + i;
        for (size_t j = 0; j < sy - 1; ++j) {
            const size_t idx = fa_yface_index(i, j, cfg->nx);
            diff_y[idx] = FA_DIFFUSION * fa_grad_face_line(line, sx, sy, j, inv_dy);
            adv_y[idx] = fields->vy[idx] * fa_interp_c2f_line(line, sx, sy, j);
        }
    }

    memset(rhs_out, 0, fields->scalar_len * sizeof(float));
    for (size_t j = 0; j < (size_t)cfg->ny; ++j) {
        for (size_t i = 0; i < (size_t)cfg->nx; ++i) {
            const float diff = fa_div_face_line(diff_x + (j + 1) * ((size_t)cfg->nx + 1),
                1, (size_t)cfg->nx + 1, i, inv_dx) +
                fa_div_face_line(diff_y + (i + 1), sx, (size_t)cfg->ny + 1, j, inv_dy);
            const float adv = fa_div_face_line(adv_x + (j + 1) * ((size_t)cfg->nx + 1),
                1, (size_t)cfg->nx + 1, i, inv_dx) +
                fa_div_face_line(adv_y + (i + 1), sx, (size_t)cfg->ny + 1, j, inv_dy);
            rhs_out[fa_scalar_index(i + 1, j + 1, cfg->nx)] = diff - adv;
        }
    }

    free(diff_x);
    free(adv_x);
    free(diff_y);
    free(adv_y);
}

static void fa_rk4_step_cpu(const fa_config_t *cfg, fa_fields_t *fields)
{
    const float dt = cfg->dt;
    const float dt_half = 0.5f * dt;
    const float dt_over_6 = dt / 6.0f;
    float *rhs = (float *)calloc(fields->scalar_len, sizeof(float));
    if (!rhs) {
        fa_die("cannot allocate RHS buffer");
    }

    for (uint32_t stage = 1; stage <= 4; ++stage) {
        const float *stage_in = fields->stage[(stage - 1) & 1u];
        float *stage_out = fields->stage[stage & 1u];
        fa_rhs_cpu(cfg, fields, stage_in, rhs);

        for (size_t j = 0; j < (size_t)cfg->ny + 2; ++j) {
            for (size_t i = 0; i < (size_t)cfg->nx + 2; ++i) {
                const size_t idx = fa_scalar_index(i, j, cfg->nx);
                if (i == 0 || j == 0 || i == (size_t)cfg->nx + 1 ||
                    j == (size_t)cfg->ny + 1) {
                    stage_out[idx] = fields->base[idx];
                    continue;
                }
                if (stage == 1) {
                    fields->accum[idx] = rhs[idx];
                    stage_out[idx] = fields->base[idx] + dt_half * rhs[idx];
                } else if (stage == 2) {
                    fields->accum[idx] = fields->accum[idx] + 2.0f * rhs[idx];
                    stage_out[idx] = fields->base[idx] + dt_half * rhs[idx];
                } else if (stage == 3) {
                    fields->accum[idx] = fields->accum[idx] + 2.0f * rhs[idx];
                    stage_out[idx] = fields->base[idx] + dt * rhs[idx];
                } else {
                    fields->accum[idx] = fields->accum[idx] + rhs[idx];
                    fields->base[idx] = fields->base[idx] + dt_over_6 * fields->accum[idx];
                    stage_out[idx] = fields->base[idx];
                }
            }
        }
    }

    free(rhs);
}

static void fa_run_cpu(const fa_options_t *opt, fa_fields_t *fields,
    fa_run_stats_t *stats)
{
    stats->init_seconds = 0.0;
    stats->gpu_busy = 0.0;
    double t0 = fa_now_seconds();
    for (uint32_t step = 0; step < opt->cfg.steps; ++step) {
        fa_rk4_step_cpu(&opt->cfg, fields);
    }
    stats->solve_seconds = fa_now_seconds() - t0;
    fa_check_finite(fields->base, fields->scalar_len, "final scalar");
    stats->final_all_finite = true;
}

// ---------------------------------------------------------------------------
// CUDA kernels.
// ---------------------------------------------------------------------------

typedef struct {
    int nx;
    int ny;
    float inv_dx;
    float inv_dy;
    float alpha;
    float weight;
    float dt_over_6;
    int stage;
} fa_stage_params_t;

// Device copies of the operator rows.  The algebra is written exactly as in
// the CPU oracle: (coefficient * inv_h) * value, summed left to right.
__device__ __forceinline__ float fa_d_grad(const float *line, int stride, int n, int i,
    float inv_h)
{
    if (i == 0) {
        return (((-352.0f / 105.0f) * inv_h) * line[0 * stride] +
                ((35.0f / 8.0f) * inv_h) * line[1 * stride] +
                ((-35.0f / 24.0f) * inv_h) * line[2 * stride] +
                ((21.0f / 40.0f) * inv_h) * line[3 * stride] +
                ((-5.0f / 56.0f) * inv_h) * line[4 * stride]);
    }
    if (i == 1) {
        return (((16.0f / 105.0f) * inv_h) * line[0 * stride] +
                ((-31.0f / 24.0f) * inv_h) * line[1 * stride] +
                ((29.0f / 24.0f) * inv_h) * line[2 * stride] +
                ((-3.0f / 40.0f) * inv_h) * line[3 * stride] +
                ((1.0f / 168.0f) * inv_h) * line[4 * stride]);
    }
    if (i + 3 < n) {
        return (((1.0f / 24.0f) * inv_h) * line[(i - 1) * stride] +
                ((-9.0f / 8.0f) * inv_h) * line[i * stride] +
                ((9.0f / 8.0f) * inv_h) * line[(i + 1) * stride] +
                ((-1.0f / 24.0f) * inv_h) * line[(i + 2) * stride]);
    }
    if (i == n - 3) {
        return (((-1.0f / 168.0f) * inv_h) * line[(n - 5) * stride] +
                ((3.0f / 40.0f) * inv_h) * line[(n - 4) * stride] +
                ((-29.0f / 24.0f) * inv_h) * line[(n - 3) * stride] +
                ((31.0f / 24.0f) * inv_h) * line[(n - 2) * stride] +
                ((-16.0f / 105.0f) * inv_h) * line[(n - 1) * stride]);
    }
    return (((5.0f / 56.0f) * inv_h) * line[(n - 5) * stride] +
            ((-21.0f / 40.0f) * inv_h) * line[(n - 4) * stride] +
            ((35.0f / 24.0f) * inv_h) * line[(n - 3) * stride] +
            ((-35.0f / 8.0f) * inv_h) * line[(n - 2) * stride] +
            ((352.0f / 105.0f) * inv_h) * line[(n - 1) * stride]);
}

__device__ __forceinline__ float fa_d_interp(const float *line, int stride, int n, int i)
{
    if (i == 0) {
        return line[0 * stride];
    }
    if (i == 1) {
        return (-16.0f / 112.0f) * line[0 * stride] +
               (70.0f / 112.0f) * line[1 * stride] +
               (70.0f / 112.0f) * line[2 * stride] +
               (-14.0f / 112.0f) * line[3 * stride] +
               (2.0f / 112.0f) * line[4 * stride];
    }
    if (i + 3 < n) {
        return (-7.0f / 112.0f) * line[(i - 1) * stride] +
               (63.0f / 112.0f) * line[i * stride] +
               (63.0f / 112.0f) * line[(i + 1) * stride] +
               (-7.0f / 112.0f) * line[(i + 2) * stride];
    }
    if (i == n - 3) {
        return (2.0f / 112.0f) * line[(n - 5) * stride] +
               (-14.0f / 112.0f) * line[(n - 4) * stride] +
               (70.0f / 112.0f) * line[(n - 3) * stride] +
               (70.0f / 112.0f) * line[(n - 2) * stride] +
               (-16.0f / 112.0f) * line[(n - 1) * stride];
    }
    return line[(n - 1) * stride];
}

__device__ __forceinline__ float fa_d_div(const float *line, int stride, int n, int i,
    float inv_h)
{
    if (i == 0) {
        return (((-11.0f / 12.0f) * inv_h) * line[0 * stride] +
                ((17.0f / 24.0f) * inv_h) * line[1 * stride] +
                ((3.0f / 8.0f) * inv_h) * line[2 * stride] +
                ((-5.0f / 24.0f) * inv_h) * line[3 * stride] +
                ((1.0f / 24.0f) * inv_h) * line[4 * stride]);
    }
    if (i + 2 < n) {
        return (((1.0f / 24.0f) * inv_h) * line[(i - 1) * stride] +
                ((-9.0f / 8.0f) * inv_h) * line[i * stride] +
                ((9.0f / 8.0f) * inv_h) * line[(i + 1) * stride] +
                ((-1.0f / 24.0f) * inv_h) * line[(i + 2) * stride]);
    }
    return (((-1.0f / 24.0f) * inv_h) * line[(n - 5) * stride] +
            ((5.0f / 24.0f) * inv_h) * line[(n - 4) * stride] +
            ((-3.0f / 8.0f) * inv_h) * line[(n - 3) * stride] +
            ((-17.0f / 24.0f) * inv_h) * line[(n - 2) * stride] +
            ((11.0f / 12.0f) * inv_h) * line[(n - 1) * stride]);
}

// RK4 stage update for one interior cell, in the oracle's operation order.
__device__ __forceinline__ void fa_d_rk_update(float *base, float *stage_out,
    float *accum, size_t idx, float rhs, const fa_stage_params_t &p)
{
    if (p.stage == 1) {
        accum[idx] = rhs;
        stage_out[idx] = base[idx] + p.alpha * rhs;
    } else if (p.stage == 2 || p.stage == 3) {
        accum[idx] = accum[idx] + p.weight * rhs;
        stage_out[idx] = base[idx] + p.alpha * rhs;
    } else {
        const float a = accum[idx] + rhs;
        accum[idx] = a;
        const float b = base[idx] + p.dt_over_6 * a;
        base[idx] = b;
        if (stage_out) {
            stage_out[idx] = b;
        }
    }
}

// Variant "split": a direct port of the Metal kernels.  One thread per
// scalar point computes its x and y face fluxes into scratch arrays; a second
// kernel takes the divergence and performs the RK stage update.
__global__ void fa_split_fluxes(const float *__restrict__ stage,
    const float *__restrict__ vx, const float *__restrict__ vy,
    float *__restrict__ diff_x, float *__restrict__ adv_x,
    float *__restrict__ diff_y, float *__restrict__ adv_y, fa_stage_params_t p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int sx = p.nx + 2;
    const int sy = p.ny + 2;
    if (i <= p.nx && j <= p.ny + 1) {
        const float *line = stage + (size_t)j * sx;
        const size_t idx = (size_t)i + (size_t)(p.nx + 1) * j;
        diff_x[idx] = 0.5f * fa_d_grad(line, 1, sx, i, p.inv_dx);
        adv_x[idx] = vx[idx] * fa_d_interp(line, 1, sx, i);
    }
    if (i <= p.nx + 1 && j <= p.ny) {
        const float *line = stage + i;
        const size_t idx = (size_t)i + (size_t)sx * j;
        diff_y[idx] = 0.5f * fa_d_grad(line, sx, sy, j, p.inv_dy);
        adv_y[idx] = vy[idx] * fa_d_interp(line, sx, sy, j);
    }
}

__global__ void fa_split_update(float *base, float *stage_out, float *accum,
    const float *__restrict__ diff_x, const float *__restrict__ adv_x,
    const float *__restrict__ diff_y, const float *__restrict__ adv_y,
    fa_stage_params_t p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > p.nx + 1 || j > p.ny + 1) {
        return;
    }
    const size_t idx = (size_t)i + (size_t)(p.nx + 2) * j;
    if (i == 0 || j == 0 || i == p.nx + 1 || j == p.ny + 1) {
        stage_out[idx] = base[idx];
        return;
    }
    const int x_idx = i - 1;
    const int y_idx = j - 1;
    const float *x_line_diff = diff_x + (size_t)j * (p.nx + 1);
    const float *x_line_adv = adv_x + (size_t)j * (p.nx + 1);
    const float *y_line_diff = diff_y + i;
    const float *y_line_adv = adv_y + i;
    const float diffx = fa_d_div(x_line_diff, 1, p.nx + 1, x_idx, p.inv_dx);
    const float diffy = fa_d_div(y_line_diff, p.nx + 2, p.ny + 1, y_idx, p.inv_dy);
    const float advx = fa_d_div(x_line_adv, 1, p.nx + 1, x_idx, p.inv_dx);
    const float advy = fa_d_div(y_line_adv, p.nx + 2, p.ny + 1, y_idx, p.inv_dy);
    const float rhs = (diffx + diffy) - (advx + advy);
    fa_d_rk_update(base, stage_out, accum, idx, rhs, p);
}

// ---------------------------------------------------------------------------
// Host CUDA driver.
// ---------------------------------------------------------------------------

typedef struct {
    float *base;
    float *base_alt;
    float *stage[2];
    float *accum;
    float *vx;
    float *vy;
    float *diff_x;
    float *adv_x;
    float *diff_y;
    float *adv_y;
    cudaStream_t side;
    cudaEvent_t fork;
    cudaEvent_t join;
} fa_device_t;

static fa_stage_params_t fa_stage_params(const fa_config_t *cfg, int stage)
{
    const float dt_half = 0.5f * cfg->dt;
    fa_stage_params_t p;
    p.nx = (int)cfg->nx;
    p.ny = (int)cfg->ny;
    p.inv_dx = (float)cfg->nx / (2.0f * FA_PI);
    p.inv_dy = (float)cfg->ny / (2.0f * FA_PI);
    p.alpha = (stage == 1 || stage == 2) ? dt_half : (stage == 3 ? cfg->dt : 0.0f);
    p.weight = (stage == 1 || stage == 4) ? 1.0f : 2.0f;
    p.dt_over_6 = cfg->dt / 6.0f;
    p.stage = stage;
    return p;
}

// Operator rows on an arbitrary line accessor, so the same expressions can read
// global memory or a shared-memory tile.  `at(k)` returns line element k in
// global line coordinates.
template <class L>
__device__ __forceinline__ float fa_t_grad(const L &at, int n, int i, float inv_h)
{
    if (i == 0) {
        return (((-352.0f / 105.0f) * inv_h) * at(0) +
                ((35.0f / 8.0f) * inv_h) * at(1) +
                ((-35.0f / 24.0f) * inv_h) * at(2) +
                ((21.0f / 40.0f) * inv_h) * at(3) +
                ((-5.0f / 56.0f) * inv_h) * at(4));
    }
    if (i == 1) {
        return (((16.0f / 105.0f) * inv_h) * at(0) +
                ((-31.0f / 24.0f) * inv_h) * at(1) +
                ((29.0f / 24.0f) * inv_h) * at(2) +
                ((-3.0f / 40.0f) * inv_h) * at(3) +
                ((1.0f / 168.0f) * inv_h) * at(4));
    }
    if (i + 3 < n) {
        return (((1.0f / 24.0f) * inv_h) * at(i - 1) +
                ((-9.0f / 8.0f) * inv_h) * at(i) +
                ((9.0f / 8.0f) * inv_h) * at(i + 1) +
                ((-1.0f / 24.0f) * inv_h) * at(i + 2));
    }
    if (i == n - 3) {
        return (((-1.0f / 168.0f) * inv_h) * at(n - 5) +
                ((3.0f / 40.0f) * inv_h) * at(n - 4) +
                ((-29.0f / 24.0f) * inv_h) * at(n - 3) +
                ((31.0f / 24.0f) * inv_h) * at(n - 2) +
                ((-16.0f / 105.0f) * inv_h) * at(n - 1));
    }
    return (((5.0f / 56.0f) * inv_h) * at(n - 5) +
            ((-21.0f / 40.0f) * inv_h) * at(n - 4) +
            ((35.0f / 24.0f) * inv_h) * at(n - 3) +
            ((-35.0f / 8.0f) * inv_h) * at(n - 2) +
            ((352.0f / 105.0f) * inv_h) * at(n - 1));
}

template <class L>
__device__ __forceinline__ float fa_t_interp(const L &at, int n, int i)
{
    if (i == 0) {
        return at(0);
    }
    if (i == 1) {
        return (-16.0f / 112.0f) * at(0) +
               (70.0f / 112.0f) * at(1) +
               (70.0f / 112.0f) * at(2) +
               (-14.0f / 112.0f) * at(3) +
               (2.0f / 112.0f) * at(4);
    }
    if (i + 3 < n) {
        return (-7.0f / 112.0f) * at(i - 1) +
               (63.0f / 112.0f) * at(i) +
               (63.0f / 112.0f) * at(i + 1) +
               (-7.0f / 112.0f) * at(i + 2);
    }
    if (i == n - 3) {
        return (2.0f / 112.0f) * at(n - 5) +
               (-14.0f / 112.0f) * at(n - 4) +
               (70.0f / 112.0f) * at(n - 3) +
               (70.0f / 112.0f) * at(n - 2) +
               (-16.0f / 112.0f) * at(n - 1);
    }
    return at(n - 1);
}

template <class L>
__device__ __forceinline__ float fa_t_div(const L &at, int n, int i, float inv_h)
{
    if (i == 0) {
        return (((-11.0f / 12.0f) * inv_h) * at(0) +
                ((17.0f / 24.0f) * inv_h) * at(1) +
                ((3.0f / 8.0f) * inv_h) * at(2) +
                ((-5.0f / 24.0f) * inv_h) * at(3) +
                ((1.0f / 24.0f) * inv_h) * at(4));
    }
    if (i + 2 < n) {
        return (((1.0f / 24.0f) * inv_h) * at(i - 1) +
                ((-9.0f / 8.0f) * inv_h) * at(i) +
                ((9.0f / 8.0f) * inv_h) * at(i + 1) +
                ((-1.0f / 24.0f) * inv_h) * at(i + 2));
    }
    return (((-1.0f / 24.0f) * inv_h) * at(n - 5) +
            ((5.0f / 24.0f) * inv_h) * at(n - 4) +
            ((-3.0f / 8.0f) * inv_h) * at(n - 3) +
            ((-17.0f / 24.0f) * inv_h) * at(n - 2) +
            ((11.0f / 12.0f) * inv_h) * at(n - 1));
}

// Shared-memory line accessor: element k of a line that starts at global
// line index `first` with element stride `stride`.
struct fa_sline_t {
    const float *p;
    int first;
    int stride;
    __device__ __forceinline__ float operator()(int k) const
    {
        return p[(k - first) * stride];
    }
};

// RK4 stage update with the stage as a compile-time constant.  Stage 4 writes
// only `base`; the next timestep's stage 1 reads `base` as its input, which
// equals the stage-4 output bit for bit.  Scalar boundaries are never
// written: base and both stage buffers start with identical boundaries.
template <int STAGE>
__device__ __forceinline__ void fa_rk_stage(float *base, float *stage_out,
    float *accum, size_t idx, float rhs, const fa_stage_params_t &p)
{
    if (STAGE == 1) {
        accum[idx] = rhs;
        stage_out[idx] = base[idx] + p.alpha * rhs;
    } else if (STAGE == 2 || STAGE == 3) {
        accum[idx] = accum[idx] + p.weight * rhs;
        stage_out[idx] = base[idx] + p.alpha * rhs;
    } else {
        const float a = accum[idx] + rhs;
        base[idx] = base[idx] + p.dt_over_6 * a;
    }
}

// Variant "fused": one kernel per RK stage.  A block owns a TX x TY tile of
// interior cells.  It stages the scalar tile plus a 5-point halo (enough for
// the one-sided boundary rows) in shared memory, computes the diffusive and
// advective x/y face fluxes of the tile into shared memory, then each thread
// takes the divergence and applies the RK stage update for its cells.
template <int TX, int TY, int STAGE>
__global__ void __launch_bounds__(TX * 8)
fa_fused_stage(const float *__restrict__ stage_in, float *base, float *stage_out,
    float *accum, const float *__restrict__ vx, const float *__restrict__ vy,
    fa_stage_params_t p)
{
    constexpr int SW = TX + 8;          // scalar columns [i0-5, i0+TX+2]
    constexpr int SH = TY + 8;          // scalar rows    [j0-5, j0+TY+2]
    constexpr int FXW = TX + 5;         // x faces [i0-4, i0+TX]
    constexpr int FYH = TY + 5;         // y faces [j0-4, j0+TY]
    __shared__ float s[SH][SW];
    __shared__ float fxd[TY][FXW];
    __shared__ float fxa[TY][FXW];
    __shared__ float fyd[FYH][TX];
    __shared__ float fya[FYH][TX];

    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int sy = ny + 2;
    const int i0 = 1 + blockIdx.x * TX;
    const int j0 = 1 + blockIdx.y * TY;
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int nthreads = blockDim.x * blockDim.y;

    for (int k = tid; k < SW * SH; k += nthreads) {
        const int li = k % SW;
        const int lj = k / SW;
        const int gi = i0 - 5 + li;
        const int gj = j0 - 5 + lj;
        if (gi >= 0 && gi < sx && gj >= 0 && gj < sy) {
            s[lj][li] = stage_in[(size_t)gi + (size_t)sx * gj];
        }
    }
    __syncthreads();

    for (int k = tid; k < TY * FXW; k += nthreads) {
        const int lf = k % FXW;
        const int lj = k / FXW;
        const int f = i0 - 4 + lf;
        const int gj = j0 + lj;
        if (f >= 0 && f <= nx && gj <= ny) {
            const fa_sline_t line = {&s[lj + 5][0], i0 - 5, 1};
            fxd[lj][lf] = 0.5f * fa_t_grad(line, sx, f, p.inv_dx);
            fxa[lj][lf] = vx[(size_t)f + (size_t)(nx + 1) * gj] * fa_t_interp(line, sx, f);
        }
    }
    for (int k = tid; k < FYH * TX; k += nthreads) {
        const int li = k % TX;
        const int lg = k / TX;
        const int g = j0 - 4 + lg;
        const int gi = i0 + li;
        if (g >= 0 && g <= ny && gi <= nx) {
            const fa_sline_t line = {&s[0][li + 5], j0 - 5, SW};
            fyd[lg][li] = 0.5f * fa_t_grad(line, sy, g, p.inv_dy);
            fya[lg][li] = vy[(size_t)gi + (size_t)sx * g] * fa_t_interp(line, sy, g);
        }
    }
    __syncthreads();

    for (int k = tid; k < TX * TY; k += nthreads) {
        const int li = k % TX;
        const int lj = k / TX;
        const int gi = i0 + li;
        const int gj = j0 + lj;
        if (gi > nx || gj > ny) {
            continue;
        }
        const fa_sline_t xd = {&fxd[lj][0], i0 - 4, 1};
        const fa_sline_t xa = {&fxa[lj][0], i0 - 4, 1};
        const fa_sline_t yd = {&fyd[0][li], j0 - 4, TX};
        const fa_sline_t ya = {&fya[0][li], j0 - 4, TX};
        const float diffx = fa_t_div(xd, nx + 1, gi - 1, p.inv_dx);
        const float diffy = fa_t_div(yd, ny + 1, gj - 1, p.inv_dy);
        const float advx = fa_t_div(xa, nx + 1, gi - 1, p.inv_dx);
        const float advy = fa_t_div(ya, ny + 1, gj - 1, p.inv_dy);
        const float rhs = (diffx + diffy) - (advx + advy);
        fa_rk_stage<STAGE>(base, stage_out, accum, (size_t)gi + (size_t)sx * gj, rhs, p);
    }
}

template <int TX, int TY>
static void fa_launch_fused(const fa_config_t *cfg, fa_device_t *d, uint32_t threads_y,
    cudaStream_t s)
{
    const dim3 block(TX, threads_y, 1);
    const dim3 grid((cfg->nx + TX - 1) / TX, (cfg->ny + TY - 1) / TY, 1);
    const fa_stage_params_t p1 = fa_stage_params(cfg, 1);
    const fa_stage_params_t p2 = fa_stage_params(cfg, 2);
    const fa_stage_params_t p3 = fa_stage_params(cfg, 3);
    const fa_stage_params_t p4 = fa_stage_params(cfg, 4);
    fa_fused_stage<TX, TY, 1><<<grid, block, 0, s>>>(d->base, d->base, d->stage[0],
        d->accum, d->vx, d->vy, p1);
    fa_fused_stage<TX, TY, 2><<<grid, block, 0, s>>>(d->stage[0], d->base, d->stage[1],
        d->accum, d->vx, d->vy, p2);
    fa_fused_stage<TX, TY, 3><<<grid, block, 0, s>>>(d->stage[1], d->base, d->stage[0],
        d->accum, d->vx, d->vy, p3);
    fa_fused_stage<TX, TY, 4><<<grid, block, 0, s>>>(d->stage[0], d->base, NULL,
        d->accum, d->vx, d->vy, p4);
}

// Interior rows of the operators, for tiles that touch no boundary row.
__device__ __forceinline__ float fa_grad_interior(float a, float b, float c, float d,
    float inv_h)
{
    return (((1.0f / 24.0f) * inv_h) * a + ((-9.0f / 8.0f) * inv_h) * b +
            ((9.0f / 8.0f) * inv_h) * c + ((-1.0f / 24.0f) * inv_h) * d);
}

__device__ __forceinline__ float fa_interp_interior(float a, float b, float c, float d)
{
    return (-7.0f / 112.0f) * a + (63.0f / 112.0f) * b + (63.0f / 112.0f) * c +
           (-7.0f / 112.0f) * d;
}

// The interior divergence row has the same coefficients as the gradient row.
__device__ __forceinline__ float fa_div_interior(float a, float b, float c, float d,
    float inv_h)
{
    return fa_grad_interior(a, b, c, d, inv_h);
}

// Variant "tiled": the "fused" structure with warp-shaped indexing.  A block
// is 32 x BY threads (one warp per thread row) and owns a 32 x TY cell tile.
// Interior tiles use the branch-free interior operator rows with 32-bit
// indexing; tiles touching a boundary fall back to the general rows.
//
// Variant "prefetch" (PF = true) additionally issues every global load of an
// interior tile (velocities, base, accumulator) into registers before the
// first barrier, so their latency overlaps the scalar-tile load instead of
// being exposed once per phase.
template <int TY, int BY, int STAGE, bool PF>
__global__ void __launch_bounds__(32 * BY)
fa_tiled_stage(const float *__restrict__ stage_in, float *base, float *stage_out,
    float *accum, const float *__restrict__ vx, const float *__restrict__ vy,
    fa_stage_params_t p)
{
    constexpr int TX = 32;
    constexpr int SW = TX + 8;          // scalar columns [i0-5, i0+TX+2]
    constexpr int SH = TY + 8;          // scalar rows    [j0-5, j0+TY+2]
    constexpr int FXW = TX + 5;         // x faces [i0-4, i0+TX]
    constexpr int FYH = TY + 5;         // y faces [j0-4, j0+TY]
    constexpr int NR = (TY + BY - 1) / BY;      // cell rows per thread
    constexpr int NG = (FYH + BY - 1) / BY;     // y-face rows per thread
    __shared__ float s[SH][SW];
    __shared__ float fxd[TY][FXW];
    __shared__ float fxa[TY][FXW];
    __shared__ float fyd[FYH][TX];
    __shared__ float fya[FYH][TX];

    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int sy = ny + 2;
    const int i0 = 1 + blockIdx.x * TX;
    const int j0 = 1 + blockIdx.y * TY;
    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    // Interior tile: every scalar of the halo exists, every face uses the
    // interior gradient/interpolation row and every cell the interior
    // divergence row.
    const bool interior = i0 >= 6 && i0 + TX + 2 <= nx &&
                          j0 >= 6 && j0 + TY + 2 <= ny;
    const int gi = i0 + lane;

    float pvx[NR][2];
    float pvy[NG];
    float pbase[NR];
    float pacc[NR];
    if (PF && interior) {
        const float *vxr = vx + (i0 - 4) + (nx + 1) * j0;
        const float *vyr = vy + i0 + sx * (j0 - 4);
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int lj = ty + BY * r;
            if (lj < TY) {
                pvx[r][0] = vxr[lane + (nx + 1) * lj];
                pvx[r][1] = lane < FXW - 32 ? vxr[lane + 32 + (nx + 1) * lj] : 0.0f;
                const int idx = gi + sx * (j0 + lj);
                pbase[r] = base[idx];
                pacc[r] = STAGE == 1 ? 0.0f : accum[idx];
            }
        }
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int lg = ty + BY * g;
            pvy[g] = lg < FYH ? vyr[lane + sx * lg] : 0.0f;
        }
    }

    if (interior) {
        const float *src = stage_in + (i0 - 5) + sx * (j0 - 5);
        for (int lj = ty; lj < SH; lj += BY) {
            s[lj][lane] = src[lane + sx * lj];
            if (lane < SW - 32) {
                s[lj][lane + 32] = src[lane + 32 + sx * lj];
            }
        }
    } else {
        for (int lj = ty; lj < SH; lj += BY) {
            const int gj = j0 - 5 + lj;
            for (int li = lane; li < SW; li += 32) {
                const int gi = i0 - 5 + li;
                if (gi >= 0 && gi < sx && gj >= 0 && gj < sy) {
                    s[lj][li] = stage_in[gi + sx * gj];
                }
            }
        }
    }
    __syncthreads();

    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;
    if (PF && interior) {
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int lj = ty + BY * r;
            if (lj < TY) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int lf = lane + 32 * h;
                    if (lf < FXW) {
                        const float a = s[lj + 5][lf], b = s[lj + 5][lf + 1];
                        const float c = s[lj + 5][lf + 2], d = s[lj + 5][lf + 3];
                        fxd[lj][lf] = 0.5f * fa_grad_interior(a, b, c, d, inv_dx);
                        fxa[lj][lf] = pvx[r][h] * fa_interp_interior(a, b, c, d);
                    }
                }
            }
        }
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int lg = ty + BY * g;
            if (lg < FYH) {
                const float a = s[lg][lane + 5], b = s[lg + 1][lane + 5];
                const float c = s[lg + 2][lane + 5], d = s[lg + 3][lane + 5];
                fyd[lg][lane] = 0.5f * fa_grad_interior(a, b, c, d, inv_dy);
                fya[lg][lane] = pvy[g] * fa_interp_interior(a, b, c, d);
            }
        }
    } else if (interior) {
        const float *vxr = vx + (i0 - 4) + (nx + 1) * j0;
        for (int lj = ty; lj < TY; lj += BY) {
            for (int lf = lane; lf < FXW; lf += 32) {
                const float a = s[lj + 5][lf], b = s[lj + 5][lf + 1];
                const float c = s[lj + 5][lf + 2], d = s[lj + 5][lf + 3];
                fxd[lj][lf] = 0.5f * fa_grad_interior(a, b, c, d, inv_dx);
                fxa[lj][lf] = vxr[lf + (nx + 1) * lj] * fa_interp_interior(a, b, c, d);
            }
        }
        const float *vyr = vy + i0 + sx * (j0 - 4);
        for (int lg = ty; lg < FYH; lg += BY) {
            const float a = s[lg][lane + 5], b = s[lg + 1][lane + 5];
            const float c = s[lg + 2][lane + 5], d = s[lg + 3][lane + 5];
            fyd[lg][lane] = 0.5f * fa_grad_interior(a, b, c, d, inv_dy);
            fya[lg][lane] = vyr[lane + sx * lg] * fa_interp_interior(a, b, c, d);
        }
    } else {
        for (int lj = ty; lj < TY; lj += BY) {
            const int gj = j0 + lj;
            for (int lf = lane; lf < FXW; lf += 32) {
                const int f = i0 - 4 + lf;
                if (f >= 0 && f <= nx && gj <= ny) {
                    const fa_sline_t line = {&s[lj + 5][0], i0 - 5, 1};
                    fxd[lj][lf] = 0.5f * fa_t_grad(line, sx, f, inv_dx);
                    fxa[lj][lf] = vx[f + (nx + 1) * gj] * fa_t_interp(line, sx, f);
                }
            }
        }
        for (int lg = ty; lg < FYH; lg += BY) {
            const int g = j0 - 4 + lg;
            if (g >= 0 && g <= ny && gi <= nx) {
                const fa_sline_t line = {&s[0][lane + 5], j0 - 5, SW};
                fyd[lg][lane] = 0.5f * fa_t_grad(line, sy, g, inv_dy);
                fya[lg][lane] = vy[gi + sx * g] * fa_t_interp(line, sy, g);
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int lj = ty + BY * r;
        if (lj >= TY) {
            continue;
        }
        const int gj = j0 + lj;
        float rhs;
        if (interior) {
            const float diffx = fa_div_interior(fxd[lj][lane + 2], fxd[lj][lane + 3],
                fxd[lj][lane + 4], fxd[lj][lane + 5], inv_dx);
            const float diffy = fa_div_interior(fyd[lj + 2][lane], fyd[lj + 3][lane],
                fyd[lj + 4][lane], fyd[lj + 5][lane], inv_dy);
            const float advx = fa_div_interior(fxa[lj][lane + 2], fxa[lj][lane + 3],
                fxa[lj][lane + 4], fxa[lj][lane + 5], inv_dx);
            const float advy = fa_div_interior(fya[lj + 2][lane], fya[lj + 3][lane],
                fya[lj + 4][lane], fya[lj + 5][lane], inv_dy);
            rhs = (diffx + diffy) - (advx + advy);
        } else {
            if (gi > nx || gj > ny) {
                continue;
            }
            const fa_sline_t xd = {&fxd[lj][0], i0 - 4, 1};
            const fa_sline_t xa = {&fxa[lj][0], i0 - 4, 1};
            const fa_sline_t yd = {&fyd[0][lane], j0 - 4, TX};
            const fa_sline_t ya = {&fya[0][lane], j0 - 4, TX};
            const float diffx = fa_t_div(xd, nx + 1, gi - 1, inv_dx);
            const float diffy = fa_t_div(yd, ny + 1, gj - 1, inv_dy);
            const float advx = fa_t_div(xa, nx + 1, gi - 1, inv_dx);
            const float advy = fa_t_div(ya, ny + 1, gj - 1, inv_dy);
            rhs = (diffx + diffy) - (advx + advy);
        }
        const int idx = gi + sx * gj;
        if (PF && interior) {
            if (STAGE == 1) {
                accum[idx] = rhs;
                stage_out[idx] = pbase[r] + p.alpha * rhs;
            } else if (STAGE == 2 || STAGE == 3) {
                accum[idx] = pacc[r] + p.weight * rhs;
                stage_out[idx] = pbase[r] + p.alpha * rhs;
            } else {
                const float a = pacc[r] + rhs;
                base[idx] = pbase[r] + p.dt_over_6 * a;
            }
        } else {
            fa_rk_stage<STAGE>(base, stage_out, accum, (size_t)idx, rhs, p);
        }
    }
}

template <int TY, int BY, bool PF>
static void fa_launch_tiled(const fa_config_t *cfg, fa_device_t *d, cudaStream_t s)
{
    const dim3 block(32, BY, 1);
    const dim3 grid((cfg->nx + 31) / 32, (cfg->ny + TY - 1) / TY, 1);
    const fa_stage_params_t p1 = fa_stage_params(cfg, 1);
    const fa_stage_params_t p2 = fa_stage_params(cfg, 2);
    const fa_stage_params_t p3 = fa_stage_params(cfg, 3);
    const fa_stage_params_t p4 = fa_stage_params(cfg, 4);
    fa_tiled_stage<TY, BY, 1, PF><<<grid, block, 0, s>>>(d->base, d->base, d->stage[0],
        d->accum, d->vx, d->vy, p1);
    fa_tiled_stage<TY, BY, 2, PF><<<grid, block, 0, s>>>(d->stage[0], d->base, d->stage[1],
        d->accum, d->vx, d->vy, p2);
    fa_tiled_stage<TY, BY, 3, PF><<<grid, block, 0, s>>>(d->stage[1], d->base, d->stage[0],
        d->accum, d->vx, d->vy, p3);
    fa_tiled_stage<TY, BY, 4, PF><<<grid, block, 0, s>>>(d->stage[0], d->base, NULL,
        d->accum, d->vx, d->vy, p4);
}

// Interior/frame split ("regyf", "pair"): tiles are split into an interior
// rectangle [bx_lo, bx_hi] x [by_lo, by_hi] (no boundary handling compiled
// in, BND = false) and the surrounding frame (BND = true), launched as two
// concurrent kernels so the rarely needed general code does not raise the
// register count, and lower the occupancy, of the main kernel.
struct fa_tiles_t {
    int nbx;
    int nby;
    int bx_lo;
    int bx_hi;
    int by_lo;
    int by_hi;
};

__device__ __forceinline__ void fa_frame_tile(const fa_tiles_t &t, int k, int &bx, int &by)
{
    const int top = t.by_lo * t.nbx;
    if (k < top) {
        bx = k % t.nbx;
        by = k / t.nbx;
        return;
    }
    k -= top;
    const int mid_rows = t.by_hi - t.by_lo + 1;
    const int side = t.bx_lo + (t.nbx - 1 - t.bx_hi);
    if (mid_rows > 0 && k < mid_rows * side) {
        by = t.by_lo + k / side;
        const int c = k % side;
        bx = c < t.bx_lo ? c : t.bx_hi + 1 + (c - t.bx_lo);
        return;
    }
    k -= mid_rows > 0 ? mid_rows * side : 0;
    const int first_bottom = mid_rows > 0 ? t.by_hi + 1 : t.by_lo;
    bx = k % t.nbx;
    by = first_bottom + k / t.nbx;
}

// Clamp an interior tile range to [1, n - 2] (the last tile may be shifted
// and the first touches the boundary) and count interior and frame tiles.
static fa_tiles_t fa_make_tiles(int nbx, int nby, int bx_hi, int by_hi, int *interior,
    int *frame)
{
    fa_tiles_t t;
    t.nbx = nbx;
    t.nby = nby;
    t.bx_lo = 1;
    t.by_lo = 1;
    t.bx_hi = bx_hi < nbx - 2 ? bx_hi : nbx - 2;
    t.by_hi = by_hi < nby - 2 ? by_hi : nby - 2;
    if (t.bx_hi < t.bx_lo || t.by_hi < t.by_lo) {
        t.bx_lo = nbx;
        t.bx_hi = nbx - 1;
        t.by_lo = nby;
        t.by_hi = nby - 1;
        *interior = 0;
    } else {
        *interior = (t.bx_hi - t.bx_lo + 1) * (t.by_hi - t.by_lo + 1);
    }
    *frame = nbx * nby - *interior;
    return t;
}

// General path of a 32 x TY tile (any tile, including those touching a
// boundary): shared scalar tile with a 5-point halo, shared face fluxes from
// the general operator rows, then divergence and RK update.  Shared arrays
// are passed in so that kernels can reuse their own allocations.
template <int TY, int BY, int STAGE>
__device__ __forceinline__ void fa_tile_general(const float *__restrict__ stage_in,
    float *base, float *stage_out, float *accum, const float *__restrict__ vx,
    const float *__restrict__ vy, const fa_stage_params_t &p, int i0, int j0,
    float (*s)[40], float (*fxd)[37], float (*fxa)[37], float (*fyd)[32],
    float (*fya)[32])
{
    constexpr int TX = 32;
    constexpr int SW = TX + 8;
    constexpr int SH = TY + 8;
    constexpr int FXW = TX + 5;
    constexpr int FYH = TY + 5;
    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int sy = ny + 2;
    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;
    const int gi = i0 + lane;

    for (int lj = ty; lj < SH; lj += BY) {
        const int gj = j0 - 5 + lj;
        for (int li = lane; li < SW; li += 32) {
            const int gs = i0 - 5 + li;
            if (gs >= 0 && gs < sx && gj >= 0 && gj < sy) {
                s[lj][li] = stage_in[gs + sx * gj];
            }
        }
    }
    __syncthreads();
    for (int lj = ty; lj < TY; lj += BY) {
        const int gj = j0 + lj;
        for (int lf = lane; lf < FXW; lf += 32) {
            const int f = i0 - 4 + lf;
            if (f >= 0 && f <= nx && gj <= ny) {
                const fa_sline_t line = {&s[lj + 5][0], i0 - 5, 1};
                fxd[lj][lf] = 0.5f * fa_t_grad(line, sx, f, inv_dx);
                fxa[lj][lf] = vx[f + (nx + 1) * gj] * fa_t_interp(line, sx, f);
            }
        }
    }
    for (int lg = ty; lg < FYH; lg += BY) {
        const int g = j0 - 4 + lg;
        if (g >= 0 && g <= ny && gi <= nx) {
            const fa_sline_t line = {&s[0][lane + 5], j0 - 5, SW};
            fyd[lg][lane] = 0.5f * fa_t_grad(line, sy, g, inv_dy);
            fya[lg][lane] = vy[gi + sx * g] * fa_t_interp(line, sy, g);
        }
    }
    __syncthreads();
    for (int lj = ty; lj < TY; lj += BY) {
        const int gj = j0 + lj;
        if (gi > nx || gj > ny) {
            continue;
        }
        const fa_sline_t xd = {&fxd[lj][0], i0 - 4, 1};
        const fa_sline_t xa = {&fxa[lj][0], i0 - 4, 1};
        const fa_sline_t yd = {&fyd[0][lane], j0 - 4, TX};
        const fa_sline_t ya = {&fya[0][lane], j0 - 4, TX};
        const float diffx = fa_t_div(xd, nx + 1, gi - 1, inv_dx);
        const float diffy = fa_t_div(yd, ny + 1, gj - 1, inv_dy);
        const float advx = fa_t_div(xa, nx + 1, gi - 1, inv_dx);
        const float advy = fa_t_div(ya, ny + 1, gj - 1, inv_dy);
        const float rhs = (diffx + diffy) - (advx + advy);
        fa_rk_stage<STAGE>(base, stage_out, accum, (size_t)(gi + sx * gj), rhs, p);
    }
}

// Variant "regy": "prefetch" with the y direction register-blocked.  Each
// thread owns RY = TY / BY consecutive cells of one column; it evaluates the
// RY + 3 y faces those cells need directly from the shared scalar column into
// registers (no shared y-face arrays, no y-face loads in the divergence).  The
// x faces of the tile are distributed over all threads as a flat list and go
// through shared memory as before.  All global loads of an interior tile are
// issued before the first barrier.
//
// Variant "regyf": the same kernel launched as an interior kernel (BND =
// false: no general code) plus a concurrent frame kernel (BND = true).
template <int TY, int BY, int STAGE, int SPLIT>
__global__ void __launch_bounds__(32 * BY)
fa_regy_stage(const float *__restrict__ stage_in, float *base, float *stage_out,
    float *accum, const float *__restrict__ vx, const float *__restrict__ vy,
    fa_stage_params_t p, fa_tiles_t tiles)
{
    constexpr int TX = 32;
    constexpr int SW = TX + 8;          // scalar columns [i0-5, i0+TX+2]
    constexpr int SH = TY + 8;          // scalar rows    [j0-5, j0+TY+2]
    constexpr int FXW = TX + 5;         // x faces [i0-4, i0+TX]
    constexpr int FYH = TY + 5;
    constexpr int RY = TY / BY;
    constexpr int NT = 32 * BY;
    constexpr int NXF = (TY * FXW + NT - 1) / NT;   // x-face items per thread
    static_assert(TY % BY == 0, "TY must be a multiple of BY");
    __shared__ float s[SH][SW];
    __shared__ float fxd[TY][FXW];
    __shared__ float fxa[TY][FXW];
    __shared__ float fyd[FYH][TX];
    __shared__ float fya[FYH][TX];

    // SPLIT: 0 = single kernel over all tiles, 1 = interior tiles only,
    // 2 = frame tiles only.
    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    if (SPLIT == 1) {
        bx += tiles.bx_lo;
        by += tiles.by_lo;
    } else if (SPLIT == 2) {
        fa_frame_tile(tiles, blockIdx.x, bx, by);
    }
    const int i0 = 1 + bx * TX;
    const int j0 = 1 + by * TY;
    if (SPLIT == 2 || (SPLIT == 0 && !(i0 >= 6 && i0 + TX + 2 <= nx &&
                                       j0 >= 6 && j0 + TY + 2 <= ny))) {
        fa_tile_general<TY, BY, STAGE>(stage_in, base, stage_out, accum, vx, vy, p,
            i0, j0, s, fxd, fxa, fyd, fya);
        return;
    }

    const int lane = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = lane + 32 * ty;
    const int lj0 = ty * RY;            // first owned row (tile-local)
    const int gi = i0 + lane;
    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;

    // Global loads first: x-face velocities, y-face velocities, base, accum.
    float pvx[NXF];
    const float *vxr = vx + (i0 - 4) + (nx + 1) * j0;
#pragma unroll
    for (int m = 0; m < NXF; ++m) {
        const int k = tid + NT * m;
        pvx[m] = k < TY * FXW ? vxr[(k % FXW) + (nx + 1) * (k / FXW)] : 0.0f;
    }
    float pvy[RY + 3];
    const float *vyc = vy + gi + sx * (j0 + lj0 - 2);
#pragma unroll
    for (int r = 0; r < RY + 3; ++r) {
        pvy[r] = vyc[sx * r];
    }
    float pbase[RY];
    float pacc[RY];
#pragma unroll
    for (int r = 0; r < RY; ++r) {
        const int idx = gi + sx * (j0 + lj0 + r);
        pbase[r] = base[idx];
        pacc[r] = STAGE == 1 ? 0.0f : accum[idx];
    }

    const float *src = stage_in + (i0 - 5) + sx * (j0 - 5);
    for (int lj = ty; lj < SH; lj += BY) {
        s[lj][lane] = src[lane + sx * lj];
        if (lane < SW - 32) {
            s[lj][lane + 32] = src[lane + 32 + sx * lj];
        }
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < NXF; ++m) {
        const int k = tid + NT * m;
        if (k < TY * FXW) {
            const int lf = k % FXW;
            const int lj = k / FXW;
            const float a = s[lj + 5][lf], b = s[lj + 5][lf + 1];
            const float c = s[lj + 5][lf + 2], d = s[lj + 5][lf + 3];
            fxd[lj][lf] = 0.5f * fa_grad_interior(a, b, c, d, inv_dx);
            fxa[lj][lf] = pvx[m] * fa_interp_interior(a, b, c, d);
        }
    }

    // y faces j0+lj0-2 .. j0+lj0+RY: scalars rows (face-1)..(face+2), i.e.
    // shared rows lj0+2 .. lj0+RY+7.
    float col[RY + 6];
#pragma unroll
    for (int r = 0; r < RY + 6; ++r) {
        col[r] = s[lj0 + 2 + r][lane + 5];
    }
    float ydf[RY + 3];
    float yaf[RY + 3];
#pragma unroll
    for (int r = 0; r < RY + 3; ++r) {
        ydf[r] = 0.5f * fa_grad_interior(col[r], col[r + 1], col[r + 2], col[r + 3], inv_dy);
        yaf[r] = pvy[r] * fa_interp_interior(col[r], col[r + 1], col[r + 2], col[r + 3]);
    }
    __syncthreads();

#pragma unroll
    for (int r = 0; r < RY; ++r) {
        const int lj = lj0 + r;
        const float diffx = fa_div_interior(fxd[lj][lane + 2], fxd[lj][lane + 3],
            fxd[lj][lane + 4], fxd[lj][lane + 5], inv_dx);
        const float diffy = fa_div_interior(ydf[r], ydf[r + 1], ydf[r + 2], ydf[r + 3], inv_dy);
        const float advx = fa_div_interior(fxa[lj][lane + 2], fxa[lj][lane + 3],
            fxa[lj][lane + 4], fxa[lj][lane + 5], inv_dx);
        const float advy = fa_div_interior(yaf[r], yaf[r + 1], yaf[r + 2], yaf[r + 3], inv_dy);
        const float rhs = (diffx + diffy) - (advx + advy);
        const int idx = gi + sx * (j0 + lj);
        if (STAGE == 1) {
            accum[idx] = rhs;
            stage_out[idx] = pbase[r] + p.alpha * rhs;
        } else if (STAGE == 2 || STAGE == 3) {
            accum[idx] = pacc[r] + p.weight * rhs;
            stage_out[idx] = pbase[r] + p.alpha * rhs;
        } else {
            base[idx] = pbase[r] + p.dt_over_6 * (pacc[r] + rhs);
        }
    }
}

template <int TY, int BY, int STAGE>
static void fa_launch_regy_stage(const fa_config_t *cfg, fa_device_t *d, bool split,
    const float *in, float *out, cudaStream_t s)
{
    const dim3 block(32, BY, 1);
    const int nx = (int)cfg->nx;
    const int ny = (int)cfg->ny;
    const fa_stage_params_t p = fa_stage_params(cfg, STAGE);
    // Interior tile b: 1 + 32b >= 6 and 1 + 32b + 34 <= nx (same for y).
    int interior, frame;
    const fa_tiles_t t = fa_make_tiles((nx + 31) / 32, (ny + TY - 1) / TY,
        (nx - 35) / 32, (ny - 3 - TY) / TY, &interior, &frame);
    if (!split) {
        const dim3 grid(t.nbx, t.nby, 1);
        fa_regy_stage<TY, BY, STAGE, 0><<<grid, block, 0, s>>>(in, d->base, out,
            d->accum, d->vx, d->vy, p, t);
        return;
    }
    fa_cuda_check(cudaEventRecord(d->fork, s), "fork record");
    fa_cuda_check(cudaStreamWaitEvent(d->side, d->fork, 0), "fork wait");
    if (frame > 0) {
        fa_regy_stage<TY, BY, STAGE, 2><<<frame, block, 0, d->side>>>(in, d->base, out,
            d->accum, d->vx, d->vy, p, t);
    }
    if (interior > 0) {
        const dim3 grid(t.bx_hi - t.bx_lo + 1, t.by_hi - t.by_lo + 1, 1);
        fa_regy_stage<TY, BY, STAGE, 1><<<grid, block, 0, s>>>(in, d->base, out,
            d->accum, d->vx, d->vy, p, t);
    }
    fa_cuda_check(cudaEventRecord(d->join, d->side), "join record");
    fa_cuda_check(cudaStreamWaitEvent(s, d->join, 0), "join wait");
}

template <int TY, int BY>
static void fa_launch_regy(const fa_config_t *cfg, fa_device_t *d, bool split,
    cudaStream_t s)
{
    fa_launch_regy_stage<TY, BY, 1>(cfg, d, split, d->base, d->stage[0], s);
    fa_launch_regy_stage<TY, BY, 2>(cfg, d, split, d->stage[0], d->stage[1], s);
    fa_launch_regy_stage<TY, BY, 3>(cfg, d, split, d->stage[1], d->stage[0], s);
    fa_launch_regy_stage<TY, BY, 4>(cfg, d, split, d->stage[0], NULL, s);
}

// Variant "stream": no shared memory and no barriers.  Each warp owns 26
// consecutive cell columns and walks a strip of TYS rows.  Lane L holds
// scalar column g = i0 - 3 + L, so all x-direction neighbours of the 26 cell
// lanes (L = 3..28) are reachable with warp shuffles:
//   x face g    = f(s[g-1], s[g], s[g+1], s[g+2])   (shuffles of the row)
//   x div at g  = f(face[g-2], face[g-1], face[g], face[g+1])
// The y direction is a register pipeline along the strip: four scalar rows
// and four y faces (diffusive and advective) are kept per lane, so each new
// row costs one scalar load, one y-face evaluation and one divergence.
// Cells whose x (or y) stencil uses a one-sided boundary row take that part
// of the RHS from the general operator rows, read directly from global
// memory; the x and y parts are then combined in the oracle's order.
struct fa_gline_t {
    const float *p;
    int stride;
    __device__ __forceinline__ float operator()(int k) const
    {
        return __ldg(p + k * stride);
    }
};

__device__ __forceinline__ void fa_general_xpart(const float *__restrict__ stage_in,
    const float *__restrict__ vx, int nx, int g, int j, float inv_dx, float &diffx,
    float &advx)
{
    const int sx = nx + 2;
    const fa_gline_t row = {stage_in + sx * j, 1};
    const float *vxr = vx + (nx + 1) * j;
    const auto dface = [&](int f) { return 0.5f * fa_t_grad(row, sx, f, inv_dx); };
    const auto aface = [&](int f) { return __ldg(vxr + f) * fa_t_interp(row, sx, f); };
    diffx = fa_t_div(dface, nx + 1, g - 1, inv_dx);
    advx = fa_t_div(aface, nx + 1, g - 1, inv_dx);
}

__device__ __forceinline__ void fa_general_ypart(const float *__restrict__ stage_in,
    const float *__restrict__ vy, int nx, int ny, int g, int j, float inv_dy,
    float &diffy, float &advy)
{
    const int sx = nx + 2;
    const int sy = ny + 2;
    const fa_gline_t col = {stage_in + g, sx};
    const float *vyc = vy + g;
    const auto dface = [&](int f) { return 0.5f * fa_t_grad(col, sy, f, inv_dy); };
    const auto aface = [&](int f) { return __ldg(vyc + sx * f) * fa_t_interp(col, sy, f); };
    diffy = fa_t_div(dface, ny + 1, j - 1, inv_dy);
    advy = fa_t_div(aface, ny + 1, j - 1, inv_dy);
}

template <int TYS, int STAGE>
__global__ void __launch_bounds__(128)
fa_stream_stage(const float *__restrict__ stage_in, float *base,
    float *__restrict__ stage_out, float *__restrict__ accum,
    const float *__restrict__ vx, const float *__restrict__ vy, fa_stage_params_t p,
    int warps_x)
{
    constexpr int CW = 26;               // cell columns per warp
    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int lane = threadIdx.x & 31;
    const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int wx = gw % warps_x;
    const int strip = gw / warps_x;
    const int j0 = 1 + strip * TYS;
    if (j0 > ny) {
        return;
    }
    const int j1 = min(j0 + TYS, ny + 1);
    const int i0 = 1 + wx * CW;
    const int g = i0 - 3 + lane;                     // this lane's column
    const int gl = min(max(g, 0), nx + 1);           // clamped for loads
    const int gf = min(max(g, 0), nx);               // clamped x-face index
    const bool cell = lane >= 3 && lane < 3 + CW && g <= nx;
    const bool x_interior = g >= 4 && g <= nx - 3;
    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;
    const float *col = stage_in + gl;
    const float *vyc = vy + gl;
    const float *vxc = vx + gf;

    // Scalar rows r0..r3 = rows j..j+3; y faces q0..q3 = faces j-2..j+1.
    const auto row_ld = [&](int j) { return __ldg(col + sx * min(max(j, 0), ny + 1)); };
    const auto yface = [&](int f, float a, float b, float c, float d, float &dif,
                           float &adv) {
        dif = 0.5f * fa_grad_interior(a, b, c, d, inv_dy);
        adv = __ldg(vyc + sx * min(max(f, 0), ny)) * fa_interp_interior(a, b, c, d);
    };
    float sm3 = row_ld(j0 - 3), sm2 = row_ld(j0 - 2), sm1 = row_ld(j0 - 1);
    float r0 = row_ld(j0), r1 = row_ld(j0 + 1), r2 = row_ld(j0 + 2);
    float dq0, aq0, dq1, aq1, dq2, aq2, dq3, aq3;
    yface(j0 - 2, sm3, sm2, sm1, r0, dq0, aq0);
    yface(j0 - 1, sm2, sm1, r0, r1, dq1, aq1);
    yface(j0, sm1, r0, r1, r2, dq2, aq2);

#pragma unroll 4
    for (int j = j0; j < j1; ++j) {
        const float r3 = row_ld(j + 3);
        const int idx = gl + sx * j;
        float b = 0.0f, a = 0.0f;
        if (cell) {
            b = base[idx];
            if (STAGE != 1) {
                a = accum[idx];
            }
        }
        const float vxj = __ldg(vxc + (nx + 1) * j);
        yface(j + 1, r0, r1, r2, r3, dq3, aq3);

        // x direction on row j: face g from scalars g-1..g+2.
        const float sl1 = __shfl_up_sync(0xffffffffu, r0, 1);
        const float sr1 = __shfl_down_sync(0xffffffffu, r0, 1);
        const float sr2 = __shfl_down_sync(0xffffffffu, r0, 2);
        const float dxf = 0.5f * fa_grad_interior(sl1, r0, sr1, sr2, inv_dx);
        const float axf = vxj * fa_interp_interior(sl1, r0, sr1, sr2);
        const float dxm2 = __shfl_up_sync(0xffffffffu, dxf, 2);
        const float dxm1 = __shfl_up_sync(0xffffffffu, dxf, 1);
        const float dxp1 = __shfl_down_sync(0xffffffffu, dxf, 1);
        const float axm2 = __shfl_up_sync(0xffffffffu, axf, 2);
        const float axm1 = __shfl_up_sync(0xffffffffu, axf, 1);
        const float axp1 = __shfl_down_sync(0xffffffffu, axf, 1);

        if (cell) {
            float diffx, advx, diffy, advy;
            if (x_interior) {
                diffx = fa_div_interior(dxm2, dxm1, dxf, dxp1, inv_dx);
                advx = fa_div_interior(axm2, axm1, axf, axp1, inv_dx);
            } else {
                fa_general_xpart(stage_in, vx, nx, g, j, inv_dx, diffx, advx);
            }
            if (j >= 4 && j <= ny - 3) {
                diffy = fa_div_interior(dq0, dq1, dq2, dq3, inv_dy);
                advy = fa_div_interior(aq0, aq1, aq2, aq3, inv_dy);
            } else {
                fa_general_ypart(stage_in, vy, nx, ny, g, j, inv_dy, diffy, advy);
            }
            const float rhs = (diffx + diffy) - (advx + advy);
            if (STAGE == 1) {
                accum[idx] = rhs;
                stage_out[idx] = b + p.alpha * rhs;
            } else if (STAGE == 2 || STAGE == 3) {
                accum[idx] = a + p.weight * rhs;
                stage_out[idx] = b + p.alpha * rhs;
            } else {
                base[idx] = b + p.dt_over_6 * (a + rhs);
            }
        }
        r0 = r1;
        r1 = r2;
        r2 = r3;
        dq0 = dq1;
        dq1 = dq2;
        dq2 = dq3;
        aq0 = aq1;
        aq1 = aq2;
        aq2 = aq3;
    }
}

template <int TYS>
static void fa_launch_stream(const fa_config_t *cfg, fa_device_t *d, cudaStream_t s)
{
    const int warps_x = (int)((cfg->nx + 25) / 26);
    const int strips = (int)((cfg->ny + TYS - 1) / TYS);
    const int warps = warps_x * strips;
    const dim3 block(128, 1, 1);
    const dim3 grid((warps + 3) / 4, 1, 1);
    const fa_stage_params_t p1 = fa_stage_params(cfg, 1);
    const fa_stage_params_t p2 = fa_stage_params(cfg, 2);
    const fa_stage_params_t p3 = fa_stage_params(cfg, 3);
    const fa_stage_params_t p4 = fa_stage_params(cfg, 4);
    fa_stream_stage<TYS, 1><<<grid, block, 0, s>>>(d->base, d->base, d->stage[0],
        d->accum, d->vx, d->vy, p1, warps_x);
    fa_stream_stage<TYS, 2><<<grid, block, 0, s>>>(d->stage[0], d->base, d->stage[1],
        d->accum, d->vx, d->vy, p2, warps_x);
    fa_stream_stage<TYS, 3><<<grid, block, 0, s>>>(d->stage[1], d->base, d->stage[0],
        d->accum, d->vx, d->vy, p3, warps_x);
    fa_stream_stage<TYS, 4><<<grid, block, 0, s>>>(d->stage[0], d->base, NULL,
        d->accum, d->vx, d->vy, p4, warps_x);
}

// Variant "step": one kernel per RK4 timestep (temporal blocking).  A block
// owns a TX x TY tile of cells and keeps a region extended by H = 12 points
// (3 per stage, the interior stencil radius) in shared memory.  Each stage is
// evaluated on a region that shrinks by 3 points per stage, so the tile's
// stage-4 result needs only the step's input `base_in`; all stage values and
// the RK accumulator stay on chip.  The last tile in each direction is
// shifted inwards to end at the last cell (tiles overlap and write identical
// values), so a tile always contains at least five cells from any interior
// edge: the wider one-sided boundary rows then never read outside the valid
// region.  `base_in` and `base_out` alternate between timesteps because
// neighbouring blocks read the halo of `base_in`.
struct fa_step_params_t {
    int nx;
    int ny;
    float inv_dx;
    float inv_dy;
    float dt;
    float dt_half;
    float dt_over_6;
};

template <bool INTERIOR, class X>
__device__ __forceinline__ float fa_cell_rhs(const X &row, const X &col,
    const float *__restrict__ vx, const float *__restrict__ vy, int gi, int gj,
    const fa_step_params_t &p)
{
    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int sy = ny + 2;
    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;
    const float *vxr = vx + (nx + 1) * gj;
    const float *vyc = vy + gi;
    if (INTERIOR) {
        float s[7], t[7];
#pragma unroll
        for (int k = 0; k < 7; ++k) {
            s[k] = row(gi - 3 + k);
            t[k] = col(gj - 3 + k);
        }
        float dxf[4], axf[4], dyf[4], ayf[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            // Face f = gi - 2 + k uses scalars f-1..f+2 = s[k..k+3].
            dxf[k] = 0.5f * fa_grad_interior(s[k], s[k + 1], s[k + 2], s[k + 3], inv_dx);
            axf[k] = __ldg(vxr + gi - 2 + k) *
                fa_interp_interior(s[k], s[k + 1], s[k + 2], s[k + 3]);
            dyf[k] = 0.5f * fa_grad_interior(t[k], t[k + 1], t[k + 2], t[k + 3], inv_dy);
            ayf[k] = __ldg(vyc + sx * (gj - 2 + k)) *
                fa_interp_interior(t[k], t[k + 1], t[k + 2], t[k + 3]);
        }
        const float diffx = fa_div_interior(dxf[0], dxf[1], dxf[2], dxf[3], inv_dx);
        const float diffy = fa_div_interior(dyf[0], dyf[1], dyf[2], dyf[3], inv_dy);
        const float advx = fa_div_interior(axf[0], axf[1], axf[2], axf[3], inv_dx);
        const float advy = fa_div_interior(ayf[0], ayf[1], ayf[2], ayf[3], inv_dy);
        return (diffx + diffy) - (advx + advy);
    }
    const auto dx_face = [&](int f) { return 0.5f * fa_t_grad(row, sx, f, inv_dx); };
    const auto ax_face = [&](int f) { return __ldg(vxr + f) * fa_t_interp(row, sx, f); };
    const auto dy_face = [&](int g) { return 0.5f * fa_t_grad(col, sy, g, inv_dy); };
    const auto ay_face = [&](int g) {
        return __ldg(vyc + sx * g) * fa_t_interp(col, sy, g);
    };
    const float diffx = fa_t_div(dx_face, nx + 1, gi - 1, inv_dx);
    const float diffy = fa_t_div(dy_face, ny + 1, gj - 1, inv_dy);
    const float advx = fa_t_div(ax_face, nx + 1, gi - 1, inv_dx);
    const float advy = fa_t_div(ay_face, ny + 1, gj - 1, inv_dy);
    return (diffx + diffy) - (advx + advy);
}

template <int TX, int TY, int BY>
struct fa_step_shape_t {
    static constexpr int H = 12;
    static constexpr int EW = TX + 2 * H;
    static constexpr int EH = TY + 2 * H;
    static constexpr int SA = (EW + 31) / 32;   // slots per thread in x
    static constexpr int SB = (EH + BY - 1) / BY;
};

template <int TX, int TY, int BY, bool INTERIOR>
__device__ __forceinline__ void fa_step_body(const float *__restrict__ base_in,
    float *__restrict__ base_out, const float *__restrict__ vx,
    const float *__restrict__ vy, float (*sa)[fa_step_shape_t<TX, TY, BY>::EW],
    float (*sb)[fa_step_shape_t<TX, TY, BY>::EW], int i0, int j0,
    const fa_step_params_t &p)
{
    using S = fa_step_shape_t<TX, TY, BY>;
    constexpr int H = S::H;
    constexpr int EW = S::EW;
    constexpr int EH = S::EH;
    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int ex0 = i0 - H;
    const int ey0 = j0 - H;
    const int lane = threadIdx.x;
    const int ty = threadIdx.y;

    float b[S::SB][S::SA];
    float acc[S::SB][S::SA];
#pragma unroll
    for (int q = 0; q < S::SB; ++q) {
#pragma unroll
        for (int r = 0; r < S::SA; ++r) {
            const int lx = lane + 32 * r;
            const int ly = ty + BY * q;
            const int gi = ex0 + lx;
            const int gj = ey0 + ly;
            float v = 0.0f;
            if (lx < EW && ly < EH && gi >= 0 && gi <= nx + 1 && gj >= 0 && gj <= ny + 1) {
                v = base_in[gi + sx * gj];
                sa[ly][lx] = v;
                sb[ly][lx] = v;
            }
            b[q][r] = v;
            acc[q][r] = 0.0f;
        }
    }
    __syncthreads();

#pragma unroll
    for (int stage = 1; stage <= 4; ++stage) {
        float (*in)[EW] = (stage & 1) ? sa : sb;
        float (*out)[EW] = (stage & 1) ? sb : sa;
        const int lo = 3 * stage;            // valid local range [lo, E - lo)
#pragma unroll
        for (int q = 0; q < S::SB; ++q) {
#pragma unroll
            for (int r = 0; r < S::SA; ++r) {
                const int lx = lane + 32 * r;
                const int ly = ty + BY * q;
                const int gi = ex0 + lx;
                const int gj = ey0 + ly;
                if (lx < lo || lx >= EW - lo || ly < lo || ly >= EH - lo ||
                    gi < 1 || gi > nx || gj < 1 || gj > ny) {
                    continue;
                }
                const fa_sline_t row = {&in[ly][0], ex0, 1};
                const fa_sline_t col = {&in[0][lx], ey0, EW};
                const float rhs = fa_cell_rhs<INTERIOR>(row, col, vx, vy, gi, gj, p);
                if (stage == 1) {
                    acc[q][r] = rhs;
                    out[ly][lx] = b[q][r] + p.dt_half * rhs;
                } else if (stage == 2) {
                    acc[q][r] = acc[q][r] + 2.0f * rhs;
                    out[ly][lx] = b[q][r] + p.dt_half * rhs;
                } else if (stage == 3) {
                    acc[q][r] = acc[q][r] + 2.0f * rhs;
                    out[ly][lx] = b[q][r] + p.dt * rhs;
                } else {
                    const float a = acc[q][r] + rhs;
                    base_out[gi + sx * gj] = b[q][r] + p.dt_over_6 * a;
                }
            }
        }
        if (stage < 4) {
            __syncthreads();
        }
    }
}

template <int TX, int TY, int BY>
__global__ void __launch_bounds__(32 * BY)
fa_step_kernel(const float *__restrict__ base_in, float *__restrict__ base_out,
    const float *__restrict__ vx, const float *__restrict__ vy, fa_step_params_t p)
{
    using S = fa_step_shape_t<TX, TY, BY>;
    __shared__ float sa[S::EH][S::EW];
    __shared__ float sb[S::EH][S::EW];
    const int i0 = min(1 + (int)blockIdx.x * TX, max(1, p.nx - TX + 1));
    const int j0 = min(1 + (int)blockIdx.y * TY, max(1, p.ny - TY + 1));
    // Interior block: every cell of the stage-1 region uses the interior
    // divergence row and every face it touches the interior gradient and
    // interpolation rows.
    const bool interior = i0 - S::H + 3 >= 4 && i0 - S::H + S::EW - 4 <= p.nx - 3 &&
                          j0 - S::H + 3 >= 4 && j0 - S::H + S::EH - 4 <= p.ny - 3;
    if (interior) {
        fa_step_body<TX, TY, BY, true>(base_in, base_out, vx, vy, sa, sb, i0, j0, p);
    } else {
        fa_step_body<TX, TY, BY, false>(base_in, base_out, vx, vy, sa, sb, i0, j0, p);
    }
}

static fa_step_params_t fa_step_params(const fa_config_t *cfg)
{
    fa_step_params_t p;
    p.nx = (int)cfg->nx;
    p.ny = (int)cfg->ny;
    p.inv_dx = (float)cfg->nx / (2.0f * FA_PI);
    p.inv_dy = (float)cfg->ny / (2.0f * FA_PI);
    p.dt = cfg->dt;
    p.dt_half = 0.5f * cfg->dt;
    p.dt_over_6 = cfg->dt / 6.0f;
    return p;
}

template <int TX, int TY, int BY>
static void fa_launch_step_kernel(const fa_config_t *cfg, fa_device_t *d, int parity,
    cudaStream_t s)
{
    const dim3 block(32, BY, 1);
    const dim3 grid((cfg->nx + TX - 1) / TX, (cfg->ny + TY - 1) / TY, 1);
    const fa_step_params_t p = fa_step_params(cfg);
    const float *in = parity ? d->base_alt : d->base;
    float *out = parity ? d->base : d->base_alt;
    fa_step_kernel<TX, TY, BY><<<grid, block, 0, s>>>(in, out, d->vx, d->vy, p);
}

// x and y parts of the divergence at one cell from the general operator
// rows, on any line accessor (used for cells next to a boundary).
template <class L>
__device__ __forceinline__ void fa_xpart(const L &row, const float *__restrict__ vxr,
    int nx, int c, float inv_dx, float &diffx, float &advx)
{
    const int sx = nx + 2;
    const auto dface = [&](int f) { return 0.5f * fa_t_grad(row, sx, f, inv_dx); };
    const auto aface = [&](int f) { return __ldg(vxr + f) * fa_t_interp(row, sx, f); };
    diffx = fa_t_div(dface, nx + 1, c - 1, inv_dx);
    advx = fa_t_div(aface, nx + 1, c - 1, inv_dx);
}

template <class L>
__device__ __forceinline__ void fa_ypart(const L &col, const float *__restrict__ vyc,
    int sx, int ny, int j, float inv_dy, float &diffy, float &advy)
{
    const int sy = ny + 2;
    const auto dface = [&](int g) { return 0.5f * fa_t_grad(col, sy, g, inv_dy); };
    const auto aface = [&](int g) { return __ldg(vyc + sx * g) * fa_t_interp(col, sy, g); };
    diffy = fa_t_div(dface, ny + 1, j - 1, inv_dy);
    advy = fa_t_div(aface, ny + 1, j - 1, inv_dy);
}

// Variant "pair": two RK stages per kernel (stages 1+2, then 3+4), so the
// stage-1 and stage-3 fields never leave the chip.  A block owns a tile of
// TX = 32*WX - 6 by TY cells.  The first stage of the pair is evaluated on
// the tile extended by the interior stencil radius 3 (AW = 32*WX columns,
// one per thread column, by AH = TY + 6 rows) into shared memory; the second
// stage reads it and updates the tile.  Both stages use the "regy" scheme:
// x faces of each row go through shared memory (flat over the block), y
// faces are register-blocked along RY consecutive rows of one column.
//   pair 0: in = base           -> s1 = base + dt/2*k2, accum = k1 + 2*k2
//   pair 1: in = s1, accum      -> base_out = base + dt/6*((accum + 2*k3) + k4)
// base_in/base_out alternate between timesteps because pair 1 reads the base
// halo of neighbouring tiles.
//
// Boundaries: faces are always formed with the interior rows (values at
// non-interior faces are simply never used).  A cell within three cells of
// a boundary takes the x and/or y part of its divergence from the general
// operator rows, evaluated directly on the shared field; the one-sided rows
// reach five cells inwards, which stays inside the shared regions as long as
// a tile is at least three cells wide.  Hence the last tile in each direction
// is shifted to end at the last cell; overlapping tiles compute identical
// values with identical code.
template <int WX, int TY, int BY>
struct fa_pair_shape_t {
    static constexpr int AW = 32 * WX;          // stage-A columns [i0-3, i0+TX+2]
    static constexpr int TX = AW - 6;
    static constexpr int AH = TY + 6;           // stage-A rows    [j0-3, j0+TY+2]
    static constexpr int SWI = AW + 6;          // input columns   [i0-6, i0+TX+5]
    static constexpr int SHI = AH + 6;          // input rows      [j0-6, j0+TY+5]
    static constexpr int FXA = AW + 3;          // stage-A x faces [i0-5, i0+TX+3]
    static constexpr int RYA = AH / BY;
    static constexpr int RYB = TY / BY;
    // Interior tiles keep the stage-A rates in the (then dead) input tile;
    // frame tiles still read the input tile there and need their own array.
    static constexpr int FLOATS_FAST = SHI * SWI + AH * AW + 2 * AH * FXA;
    static constexpr int FLOATS_BND = FLOATS_FAST + AH * AW;
};

template <int WX, int TY, int BY, int PAIR, bool BND>
__global__ void __launch_bounds__(32 * WX * BY, BND ? 1 : (1536 / (32 * WX * BY) < 6 ? 1536 / (32 * WX * BY) : 6))
fa_pair_kernel(const float *__restrict__ in, const float *__restrict__ base_in,
    float *__restrict__ base_out, float *__restrict__ s1, float *__restrict__ accum,
    const float *__restrict__ vx, const float *__restrict__ vy, fa_step_params_t p,
    fa_tiles_t tiles)
{
    using S = fa_pair_shape_t<WX, TY, BY>;
    constexpr int AW = S::AW;
    constexpr int AH = S::AH;
    constexpr int TX = S::TX;
    constexpr int SWI = S::SWI;
    constexpr int SHI = S::SHI;
    constexpr int FXA = S::FXA;
    constexpr int RYA = S::RYA;
    constexpr int RYB = S::RYB;
    constexpr int NT = AW * BY;
    static_assert(AH % BY == 0 && TY % BY == 0, "BY must divide TY and TY + 6");
    __shared__ float smem[BND ? S::FLOATS_BND : S::FLOATS_FAST];
    float (*sin_)[SWI] = reinterpret_cast<float (*)[SWI]>(smem);
    float (*u)[AW] = reinterpret_cast<float (*)[AW]>(smem + SHI * SWI);
    float (*fxd)[FXA] = reinterpret_cast<float (*)[FXA]>(smem + SHI * SWI + AH * AW);
    float (*fxa)[FXA] = reinterpret_cast<float (*)[FXA]>(smem + SHI * SWI + AH * AW +
        AH * FXA);
    float (*ka)[AW] = reinterpret_cast<float (*)[AW]>(BND ? smem + S::FLOATS_FAST : smem);

    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    int bx, by;
    if (BND) {
        fa_frame_tile(tiles, blockIdx.x, bx, by);
    } else {
        bx = tiles.bx_lo + blockIdx.x;
        by = tiles.by_lo + blockIdx.y;
    }
    const int i0 = min(1 + bx * TX, max(1, nx - TX + 1));
    const int j0 = min(1 + by * TY, max(1, ny - TY + 1));
    // Interior tiles: every point of the input region exists (no load
    // guards) and every stage-A cell uses the interior rows.
    constexpr bool inside = !BND;
    const auto exists = [&](int gi, int gj) {
        return gi >= 0 && gi <= nx + 1 && gj >= 0 && gj <= ny + 1;
    };

    const int lc = threadIdx.x;         // stage-A column, 0..AW-1
    const int ty = threadIdx.y;
    const int tid = lc + AW * ty;
    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;
    const float alpha = PAIR == 0 ? p.dt_half : p.dt;

    // ---- stage A (1 or 3) on the extended region ----
    const int ra0 = ty * RYA;
    const int ca = i0 - 3 + lc;         // global column of this thread
    float pvy[RYA + 3];
    float pbase[RYA];
    {
        const int cc = min(max(ca, 0), nx + 1);
#pragma unroll
        for (int q = 0; q < RYA + 3; ++q) {
            const int g = min(max(j0 - 5 + ra0 + q, 0), ny);
            pvy[q] = __ldg(vy + cc + sx * g);
        }
#pragma unroll
        for (int r = 0; r < RYA; ++r) {
            const int gj = min(max(j0 - 3 + ra0 + r, 0), ny + 1);
            pbase[r] = __ldg(base_in + cc + sx * gj);
        }
    }
    if (inside) {
        const float *src = in + (i0 - 6) + sx * (j0 - 6);
        for (int q = tid; q < SHI * SWI; q += NT) {
            const int li = q % SWI;
            const int lj = q / SWI;
            sin_[lj][li] = __ldg(src + li + sx * lj);
        }
    } else {
        for (int q = tid; q < SHI * SWI; q += NT) {
            const int li = q % SWI;
            const int lj = q / SWI;
            const int gi = i0 - 6 + li;
            const int gj = j0 - 6 + lj;
            sin_[lj][li] = exists(gi, gj) ? __ldg(in + gi + sx * gj) : 0.0f;
        }
    }
    __syncthreads();
    for (int q = tid; q < AH * FXA; q += NT) {
        const int lf = q % FXA;
        const int r = q / FXA;
        const int f = min(max(i0 - 5 + lf, 0), nx);
        const int gj = min(max(j0 - 3 + r, 0), ny + 1);
        const float a = sin_[r + 3][lf], b = sin_[r + 3][lf + 1];
        const float c = sin_[r + 3][lf + 2], d = sin_[r + 3][lf + 3];
        fxd[r][lf] = 0.5f * fa_grad_interior(a, b, c, d, inv_dx);
        fxa[r][lf] = __ldg(vx + f + (nx + 1) * gj) * fa_interp_interior(a, b, c, d);
    }
    float ydf[RYA + 3];
    float yaf[RYA + 3];
    {
        float colv[RYA + 6];
#pragma unroll
        for (int q = 0; q < RYA + 6; ++q) {
            colv[q] = sin_[ra0 + q][lc + 3];
        }
#pragma unroll
        for (int q = 0; q < RYA + 3; ++q) {
            ydf[q] = 0.5f * fa_grad_interior(colv[q], colv[q + 1], colv[q + 2], colv[q + 3],
                inv_dy);
            yaf[q] = pvy[q] * fa_interp_interior(colv[q], colv[q + 1], colv[q + 2], colv[q + 3]);
        }
    }
    __syncthreads();
    const bool x_int_a = !BND || (ca >= 4 && ca <= nx - 3);
    const bool col_ok_a = !BND || (ca >= 1 && ca <= nx);
#pragma unroll
    for (int r = 0; r < RYA; ++r) {
        const int rr = ra0 + r;
        const int gj = j0 - 3 + rr;
        if (BND && !(col_ok_a && gj >= 1 && gj <= ny)) {
            // Fixed boundary (or nonexistent) point: the stage value is the
            // input value, which equals base there.
            u[rr][lc] = sin_[rr + 3][lc + 3];
            continue;
        }
        float diffx, advx, diffy, advy;
        if (!BND || x_int_a) {
            diffx = fa_div_interior(fxd[rr][lc], fxd[rr][lc + 1], fxd[rr][lc + 2],
                fxd[rr][lc + 3], inv_dx);
            advx = fa_div_interior(fxa[rr][lc], fxa[rr][lc + 1], fxa[rr][lc + 2],
                fxa[rr][lc + 3], inv_dx);
        } else {
            const fa_sline_t row = {&sin_[rr + 3][0], i0 - 6, 1};
            fa_xpart(row, vx + (nx + 1) * gj, nx, ca, inv_dx, diffx, advx);
        }
        if (!BND || (gj >= 4 && gj <= ny - 3)) {
            diffy = fa_div_interior(ydf[r], ydf[r + 1], ydf[r + 2], ydf[r + 3], inv_dy);
            advy = fa_div_interior(yaf[r], yaf[r + 1], yaf[r + 2], yaf[r + 3], inv_dy);
        } else {
            const fa_sline_t col = {&sin_[0][lc + 3], j0 - 6, SWI};
            fa_ypart(col, vy + ca, sx, ny, gj, inv_dy, diffy, advy);
        }
        const float rhs = (diffx + diffy) - (advx + advy);
        ka[rr][lc] = rhs;
        u[rr][lc] = pbase[r] + alpha * rhs;
    }

    // ---- stage B (2 or 4) on the tile ----
    const int t0 = ty * RYB;
    const int gi = ca;                  // cell column (cells: lc in [3, AW-3))
    const bool cell_col = lc >= 3 && lc < AW - 3 && (!BND || gi <= nx);
    float qvy[RYB + 3];
    float qbase[RYB];
    float qacc[RYB];
    {
        const int cc = min(max(gi, 0), nx + 1);
#pragma unroll
        for (int q = 0; q < RYB + 3; ++q) {
            const int g = min(max(j0 - 2 + t0 + q, 0), ny);
            qvy[q] = __ldg(vy + cc + sx * g);
        }
#pragma unroll
        for (int r = 0; r < RYB; ++r) {
            const int idx = cc + sx * min(j0 + t0 + r, ny + 1);
            qbase[r] = __ldg(base_in + idx);
            qacc[r] = PAIR == 1 ? accum[idx] : 0.0f;
        }
    }
    __syncthreads();
    {
        constexpr int FXB = AW - 3;     // stage-B x faces [i0-2, i0+TX]
        for (int q = tid; q < TY * FXB; q += NT) {
            const int lf = q % FXB;
            const int t = q / FXB;
            const int f = min(max(i0 - 2 + lf, 0), nx);
            const int gj = min(j0 + t, ny + 1);
            const float a = u[t + 3][lf], b = u[t + 3][lf + 1];
            const float c = u[t + 3][lf + 2], d = u[t + 3][lf + 3];
            fxd[t][lf] = 0.5f * fa_grad_interior(a, b, c, d, inv_dx);
            fxa[t][lf] = __ldg(vx + f + (nx + 1) * gj) * fa_interp_interior(a, b, c, d);
        }
    }
    float zdf[RYB + 3];
    float zaf[RYB + 3];
    {
        float colv[RYB + 6];
#pragma unroll
        for (int q = 0; q < RYB + 6; ++q) {
            colv[q] = u[t0 + q][lc];
        }
#pragma unroll
        for (int q = 0; q < RYB + 3; ++q) {
            zdf[q] = 0.5f * fa_grad_interior(colv[q], colv[q + 1], colv[q + 2], colv[q + 3],
                inv_dy);
            zaf[q] = qvy[q] * fa_interp_interior(colv[q], colv[q + 1], colv[q + 2], colv[q + 3]);
        }
    }
    float kb[RYB];
#pragma unroll
    for (int r = 0; r < RYB; ++r) {
        kb[r] = ka[t0 + r + 3][lc];
    }
    __syncthreads();
    if (!cell_col) {
        return;
    }
    const bool x_int_b = !BND || (gi >= 4 && gi <= nx - 3);
#pragma unroll
    for (int r = 0; r < RYB; ++r) {
        const int t = t0 + r;
        const int gj = j0 + t;
        if (BND && gj > ny) {
            continue;
        }
        float diffx, advx, diffy, advy;
        if (!BND || x_int_b) {
            const int lf = lc - 3;
            diffx = fa_div_interior(fxd[t][lf], fxd[t][lf + 1], fxd[t][lf + 2],
                fxd[t][lf + 3], inv_dx);
            advx = fa_div_interior(fxa[t][lf], fxa[t][lf + 1], fxa[t][lf + 2],
                fxa[t][lf + 3], inv_dx);
        } else {
            const fa_sline_t row = {&u[t + 3][0], i0 - 3, 1};
            fa_xpart(row, vx + (nx + 1) * gj, nx, gi, inv_dx, diffx, advx);
        }
        if (!BND || (gj >= 4 && gj <= ny - 3)) {
            diffy = fa_div_interior(zdf[r], zdf[r + 1], zdf[r + 2], zdf[r + 3], inv_dy);
            advy = fa_div_interior(zaf[r], zaf[r + 1], zaf[r + 2], zaf[r + 3], inv_dy);
        } else {
            const fa_sline_t col = {&u[0][lc], j0 - 3, AW};
            fa_ypart(col, vy + gi, sx, ny, gj, inv_dy, diffy, advy);
        }
        const float rhs = (diffx + diffy) - (advx + advy);
        const int idx = gi + sx * gj;
        if (PAIR == 0) {
            accum[idx] = kb[r] + 2.0f * rhs;
            s1[idx] = qbase[r] + p.dt_half * rhs;
        } else {
            const float acc = qacc[r] + 2.0f * kb[r];
            base_out[idx] = qbase[r] + p.dt_over_6 * (acc + rhs);
        }
    }
}

template <int WX, int TY, int BY>
static void fa_launch_pair(const fa_config_t *cfg, fa_device_t *d, int parity,
    cudaStream_t s)
{
    using S = fa_pair_shape_t<WX, TY, BY>;
    const int nx = (int)cfg->nx;
    const int ny = (int)cfg->ny;
    // Interior tile b: 1 + b*T >= 7 and 1 + b*T + T + 2 <= n - 3.
    int interior, frame;
    const fa_tiles_t t = fa_make_tiles((nx + S::TX - 1) / S::TX, (ny + TY - 1) / TY,
        (nx - 6 - S::TX) / S::TX, (ny - 6 - TY) / TY, &interior, &frame);
    const dim3 block(32 * WX, BY, 1);
    const fa_step_params_t p = fa_step_params(cfg);
    const float *bin = parity ? d->base_alt : d->base;
    float *bout = parity ? d->base : d->base_alt;
    for (int pair = 0; pair < 2; ++pair) {
        const float *in = pair == 0 ? bin : d->stage[1];
        fa_cuda_check(cudaEventRecord(d->fork, s), "fork record");
        fa_cuda_check(cudaStreamWaitEvent(d->side, d->fork, 0), "fork wait");
        if (frame > 0) {
            if (pair == 0) {
                fa_pair_kernel<WX, TY, BY, 0, true><<<frame, block, 0, d->side>>>(in, bin,
                    bout, d->stage[1], d->accum, d->vx, d->vy, p, t);
            } else {
                fa_pair_kernel<WX, TY, BY, 1, true><<<frame, block, 0, d->side>>>(in, bin,
                    bout, d->stage[1], d->accum, d->vx, d->vy, p, t);
            }
        }
        if (interior > 0) {
            const dim3 grid(t.bx_hi - t.bx_lo + 1, t.by_hi - t.by_lo + 1, 1);
            if (pair == 0) {
                fa_pair_kernel<WX, TY, BY, 0, false><<<grid, block, 0, s>>>(in, bin, bout,
                    d->stage[1], d->accum, d->vx, d->vy, p, t);
            } else {
                fa_pair_kernel<WX, TY, BY, 1, false><<<grid, block, 0, s>>>(in, bin, bout,
                    d->stage[1], d->accum, d->vx, d->vy, p, t);
            }
        }
        fa_cuda_check(cudaEventRecord(d->join, d->side), "join record");
        fa_cuda_check(cudaStreamWaitEvent(s, d->join, 0), "join wait");
    }
}

// Launch one timestep.  Only the "step" variant alternates base buffers by
// parity; the others always update d->base in place.
static void fa_launch_step(const fa_options_t *opt, fa_device_t *d, int parity,
    cudaStream_t s)
{
    const fa_config_t *cfg = &opt->cfg;
    const dim3 block(opt->block_x, opt->block_y, 1);
    switch (opt->kernel) {
    case FA_KERNEL_SPLIT: {
        const dim3 grid((cfg->nx + 2 + block.x - 1) / block.x,
            (cfg->ny + 2 + block.y - 1) / block.y, 1);
        int in = 0;
        for (int stage = 1; stage <= 4; ++stage) {
            const fa_stage_params_t p = fa_stage_params(cfg, stage);
            fa_split_fluxes<<<grid, block, 0, s>>>(d->stage[in], d->vx, d->vy,
                d->diff_x, d->adv_x, d->diff_y, d->adv_y, p);
            fa_split_update<<<grid, block, 0, s>>>(d->base, d->stage[in ^ 1],
                d->accum, d->diff_x, d->adv_x, d->diff_y, d->adv_y, p);
            in ^= 1;
        }
        break;
    }
    case FA_KERNEL_FUSED:
        if (opt->block_x != 32 || (opt->block_y != 4 && opt->block_y != 8)) {
            fa_die("fused kernel supports --block 32,4 or 32,8");
        }
        if (opt->block_y == 4) {
            fa_launch_fused<32, 16>(cfg, d, 4, s);
        } else {
            fa_launch_fused<32, 16>(cfg, d, 8, s);
        }
        break;
    case FA_KERNEL_TILED:
    case FA_KERNEL_PREFETCH: {
        if (opt->block_x != 32 || opt->tile_x != 32) {
            fa_die("tiled kernels require --block 32,<BY> and --tile-x 32");
        }
        const bool pf = opt->kernel == FA_KERNEL_PREFETCH;
        switch (opt->block_y * 1000 + opt->tile_y) {
        case 8016: pf ? fa_launch_tiled<16, 8, true>(cfg, d, s) : fa_launch_tiled<16, 8, false>(cfg, d, s); break;
        case 8032: pf ? fa_launch_tiled<32, 8, true>(cfg, d, s) : fa_launch_tiled<32, 8, false>(cfg, d, s); break;
        case 4016: pf ? fa_launch_tiled<16, 4, true>(cfg, d, s) : fa_launch_tiled<16, 4, false>(cfg, d, s); break;
        case 4008: pf ? fa_launch_tiled<8, 4, true>(cfg, d, s) : fa_launch_tiled<8, 4, false>(cfg, d, s); break;
        case 8008: pf ? fa_launch_tiled<8, 8, true>(cfg, d, s) : fa_launch_tiled<8, 8, false>(cfg, d, s); break;
        case 16032: pf ? fa_launch_tiled<32, 16, true>(cfg, d, s) : fa_launch_tiled<32, 16, false>(cfg, d, s); break;
        case 16016: pf ? fa_launch_tiled<16, 16, true>(cfg, d, s) : fa_launch_tiled<16, 16, false>(cfg, d, s); break;
        case 16064: pf ? fa_launch_tiled<64, 16, true>(cfg, d, s) : fa_launch_tiled<64, 16, false>(cfg, d, s); break;
        default: fa_die("unsupported tiled --block/--tile-y combination");
        }
        break;
    }
    case FA_KERNEL_REGY:
    case FA_KERNEL_REGYF: {
        if (opt->block_x != 32 || opt->tile_x != 32) {
            fa_die("regy kernels require --block 32,<BY> and --tile-x 32");
        }
        const bool sp = opt->kernel == FA_KERNEL_REGYF;
        switch (opt->block_y * 1000 + opt->tile_y) {
        case 8016: fa_launch_regy<16, 8>(cfg, d, sp, s); break;
        case 4016: fa_launch_regy<16, 4>(cfg, d, sp, s); break;
        case 8032: fa_launch_regy<32, 8>(cfg, d, sp, s); break;
        case 16032: fa_launch_regy<32, 16>(cfg, d, sp, s); break;
        case 4008: fa_launch_regy<8, 4>(cfg, d, sp, s); break;
        case 8008: fa_launch_regy<8, 8>(cfg, d, sp, s); break;
        case 16016: fa_launch_regy<16, 16>(cfg, d, sp, s); break;
        case 12024: fa_launch_regy<24, 12>(cfg, d, sp, s); break;
        case 6012: fa_launch_regy<12, 6>(cfg, d, sp, s); break;
        case 8024: fa_launch_regy<24, 8>(cfg, d, sp, s); break;
        case 6024: fa_launch_regy<24, 6>(cfg, d, sp, s); break;
        case 12012: fa_launch_regy<12, 12>(cfg, d, sp, s); break;
        default: fa_die("unsupported regy --block/--tile-y combination");
        }
        break;
    }
    case FA_KERNEL_STREAM:
        switch (opt->tile_y) {
        case 8: fa_launch_stream<8>(cfg, d, s); break;
        case 16: fa_launch_stream<16>(cfg, d, s); break;
        case 32: fa_launch_stream<32>(cfg, d, s); break;
        case 64: fa_launch_stream<64>(cfg, d, s); break;
        case 128: fa_launch_stream<128>(cfg, d, s); break;
        default: fa_die("unsupported stream --tile-y");
        }
        break;
    case FA_KERNEL_PAIR:
        switch (opt->tile_x * 1000000u + opt->tile_y * 1000u + opt->block_y) {
        case 26024006u: fa_launch_pair<1, 24, 6>(cfg, d, parity, s); break;
        case 26024003u: fa_launch_pair<1, 24, 3>(cfg, d, parity, s); break;
        case 26018006u: fa_launch_pair<1, 18, 6>(cfg, d, parity, s); break;
        case 26018003u: fa_launch_pair<1, 18, 3>(cfg, d, parity, s); break;
        case 26042006u: fa_launch_pair<1, 42, 6>(cfg, d, parity, s); break;
        case 26012006u: fa_launch_pair<1, 12, 6>(cfg, d, parity, s); break;
        case 58018006u: fa_launch_pair<2, 18, 6>(cfg, d, parity, s); break;
        case 58018003u: fa_launch_pair<2, 18, 3>(cfg, d, parity, s); break;
        case 58012006u: fa_launch_pair<2, 12, 6>(cfg, d, parity, s); break;
        case 58012003u: fa_launch_pair<2, 12, 3>(cfg, d, parity, s); break;
        default: fa_die("unsupported pair --tile-x/--tile-y/--block combination");
        }
        break;
    case FA_KERNEL_STEP:
        if (opt->block_x != 32) {
            fa_die("step kernel requires --block 32,<BY>");
        }
        switch (opt->tile_x * 1000000u + opt->tile_y * 1000u + opt->block_y) {
        case 40040016u: fa_launch_step_kernel<40, 40, 16>(cfg, d, parity, s); break;
        case 40040008u: fa_launch_step_kernel<40, 40, 8>(cfg, d, parity, s); break;
        case 40024008u: fa_launch_step_kernel<40, 24, 8>(cfg, d, parity, s); break;
        case 40056008u: fa_launch_step_kernel<40, 56, 8>(cfg, d, parity, s); break;
        case 40056016u: fa_launch_step_kernel<40, 56, 16>(cfg, d, parity, s); break;
        case 72024008u: fa_launch_step_kernel<72, 24, 8>(cfg, d, parity, s); break;
        case 72024016u: fa_launch_step_kernel<72, 24, 16>(cfg, d, parity, s); break;
        default: fa_die("unsupported step --tile-x/--tile-y/--block combination");
        }
        break;
    default:
        fa_die("unsupported kernel variant");
    }
}

static void fa_upload_state(const fa_fields_t *fields, fa_device_t *d)
{
    const size_t sb = fields->scalar_len * sizeof(float);
    fa_cuda_check(cudaMemcpy(d->base, fields->base, sb, cudaMemcpyHostToDevice), "upload base");
    if (d->base_alt) {
        fa_cuda_check(cudaMemcpy(d->base_alt, fields->base, sb, cudaMemcpyHostToDevice),
            "upload base_alt");
    }
    fa_cuda_check(cudaMemcpy(d->stage[0], fields->stage[0], sb, cudaMemcpyHostToDevice), "upload stage0");
    fa_cuda_check(cudaMemcpy(d->stage[1], fields->stage[1], sb, cudaMemcpyHostToDevice), "upload stage1");
    fa_cuda_check(cudaMemcpy(d->accum, fields->accum, sb, cudaMemcpyHostToDevice), "upload accum");
}

static void fa_run_cuda(const fa_options_t *opt, fa_fields_t *fields,
    fa_run_stats_t *stats)
{
    stats->init_seconds = 0.0;
    stats->solve_seconds = 0.0;
    stats->gpu_busy = 0.0;
    stats->final_all_finite = false;

    const double init_start = fa_now_seconds();
    fa_cuda_check(cudaSetDevice(0), "cudaSetDevice");
    fa_cuda_check(cudaFree(0), "context creation");
    const size_t sb = fields->scalar_len * sizeof(float);
    const size_t xb = fields->xface_len * sizeof(float);
    const size_t yb = fields->yface_len * sizeof(float);
    fa_device_t d;
    memset(&d, 0, sizeof(d));
    fa_cuda_check(cudaMalloc((void **)&d.base, sb), "malloc base");
    if (opt->kernel == FA_KERNEL_STEP || opt->kernel == FA_KERNEL_PAIR) {
        fa_cuda_check(cudaMalloc((void **)&d.base_alt, sb), "malloc base_alt");
    }
    fa_cuda_check(cudaMalloc((void **)&d.stage[0], sb), "malloc stage0");
    fa_cuda_check(cudaMalloc((void **)&d.stage[1], sb), "malloc stage1");
    fa_cuda_check(cudaMalloc((void **)&d.accum, sb), "malloc accum");
    fa_cuda_check(cudaMalloc((void **)&d.vx, xb), "malloc vx");
    fa_cuda_check(cudaMalloc((void **)&d.vy, yb), "malloc vy");
    if (opt->kernel == FA_KERNEL_SPLIT) {
        fa_cuda_check(cudaMalloc((void **)&d.diff_x, xb), "malloc diff_x");
        fa_cuda_check(cudaMalloc((void **)&d.adv_x, xb), "malloc adv_x");
        fa_cuda_check(cudaMalloc((void **)&d.diff_y, yb), "malloc diff_y");
        fa_cuda_check(cudaMalloc((void **)&d.adv_y, yb), "malloc adv_y");
    }
    fa_cuda_check(cudaMemcpy(d.vx, fields->vx, xb, cudaMemcpyHostToDevice), "upload vx");
    fa_cuda_check(cudaMemcpy(d.vy, fields->vy, yb, cudaMemcpyHostToDevice), "upload vy");
    fa_upload_state(fields, &d);

    cudaStream_t stream;
    fa_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    fa_cuda_check(cudaStreamCreateWithFlags(&d.side, cudaStreamNonBlocking), "side stream");
    fa_cuda_check(cudaEventCreateWithFlags(&d.fork, cudaEventDisableTiming), "fork event");
    fa_cuda_check(cudaEventCreateWithFlags(&d.join, cudaEventDisableTiming), "join event");
    cudaEvent_t ev_start, ev_end;
    fa_cuda_check(cudaEventCreate(&ev_start), "event");
    fa_cuda_check(cudaEventCreate(&ev_end), "event");

    // Optional CUDA Graphs holding one whole timestep, one per base-buffer
    // parity (identical for the in-place variants).
    cudaGraphExec_t step_graph[2] = {NULL, NULL};
    if (opt->graph) {
        for (int parity = 0; parity < 2; ++parity) {
            cudaGraph_t graph;
            fa_cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                "begin capture");
            fa_launch_step(opt, &d, parity, stream);
            fa_cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
            fa_cuda_check(cudaGraphInstantiate(&step_graph[parity], graph, 0),
                "graph instantiate");
            fa_cuda_check(cudaGraphDestroy(graph), "graph destroy");
        }
    }

    // Untimed warm-up timestep (module load, caches, clocks), then restore
    // the initial state; the host copies were never modified.
    if (opt->graph) {
        fa_cuda_check(cudaGraphLaunch(step_graph[0], stream), "warmup graph");
    } else {
        fa_launch_step(opt, &d, 0, stream);
    }
    fa_cuda_check(cudaGetLastError(), "warmup launch");
    fa_cuda_check(cudaStreamSynchronize(stream), "warmup");
    fa_upload_state(fields, &d);
    fa_cuda_check(cudaDeviceSynchronize(), "restore");
    stats->init_seconds = fa_now_seconds() - init_start;

    const double solve_start = fa_now_seconds();
    fa_cuda_check(cudaEventRecord(ev_start, stream), "record start");
    for (uint32_t step = 0; step < opt->cfg.steps; ++step) {
        const int parity = (int)(step & 1u);
        if (opt->graph) {
            fa_cuda_check(cudaGraphLaunch(step_graph[parity], stream), "graph launch");
        } else {
            fa_launch_step(opt, &d, parity, stream);
        }
    }
    fa_cuda_check(cudaGetLastError(), "solve launch");
    fa_cuda_check(cudaEventRecord(ev_end, stream), "record end");
    const float *final_base = (d.base_alt && (opt->cfg.steps & 1u)) ? d.base_alt : d.base;
    fa_cuda_check(cudaMemcpyAsync(fields->base, final_base, sb, cudaMemcpyDeviceToHost,
        stream), "download base");
    fa_cuda_check(cudaStreamSynchronize(stream), "solve");
    stats->solve_seconds = fa_now_seconds() - solve_start;
    float busy_ms = 0.0f;
    fa_cuda_check(cudaEventElapsedTime(&busy_ms, ev_start, ev_end), "event elapsed");
    stats->gpu_busy = 1e-3 * (double)busy_ms;
    fa_check_finite(fields->base, fields->scalar_len, "final scalar");
    stats->final_all_finite = true;

    for (int parity = 0; parity < 2; ++parity) {
        if (step_graph[parity]) {
            cudaGraphExecDestroy(step_graph[parity]);
        }
    }
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(d.side);
    cudaEventDestroy(d.fork);
    cudaEventDestroy(d.join);
    cudaFree(d.base);
    cudaFree(d.base_alt);
    cudaFree(d.stage[0]);
    cudaFree(d.stage[1]);
    cudaFree(d.accum);
    cudaFree(d.vx);
    cudaFree(d.vy);
    cudaFree(d.diff_x);
    cudaFree(d.adv_x);
    cudaFree(d.diff_y);
    cudaFree(d.adv_y);
}

// ---------------------------------------------------------------------------
// Command line.
// ---------------------------------------------------------------------------

static void fa_usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s [--mode cpu|cuda] [--cells nx,ny] [--steps n] [--dt dt]\n"
        "          [--input path] [--output path] [--dump [path]] [--dump-bin [path]]\n"
        "          [--kernel split|fused|tiled|prefetch|regy|regyf|stream|step|pair] [--block x[,y]]\n"
        "          [--tile-x n] [--tile-y n] [--graph on|off]\n",
        argv0);
}

static bool fa_parse_pair(const char *raw, uint32_t *a, uint32_t *b, bool allow_single,
    const char *flag)
{
    const char *comma = strchr(raw, ',');
    if (!comma) {
        if (!allow_single) {
            return false;
        }
        *a = fa_parse_u32_strict(raw, flag);
        *b = 1;
        return true;
    }
    size_t left_len = (size_t)(comma - raw);
    size_t right_len = strlen(comma + 1);
    if (left_len == 0 || right_len == 0 || left_len >= 32 || right_len >= 32) {
        return false;
    }
    char left[32];
    char right[32];
    memcpy(left, raw, left_len);
    left[left_len] = '\0';
    memcpy(right, comma + 1, right_len + 1);
    *a = fa_parse_u32_strict(left, flag);
    *b = fa_parse_u32_strict(right, flag);
    return true;
}

int main(int argc, char **argv)
{
    fa_options_t opt;
    memset(&opt, 0, sizeof(opt));
    opt.cfg.nx = 51;
    opt.cfg.ny = 51;
    opt.cfg.steps = 500;
    opt.cfg.dt = 1.0e-4f;
    opt.kernel = FA_KERNEL_FUSED;
    opt.graph = true;
    opt.tile_x = 0;
    opt.tile_y = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fa_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "cpu") == 0) {
                opt.cuda = false;
            } else if (strcmp(argv[i], "cuda") == 0) {
                opt.cuda = true;
            } else {
                fa_die("mode must be cpu or cuda");
            }
        } else if (strcmp(argv[i], "--cells") == 0 && i + 1 < argc) {
            if (!fa_parse_pair(argv[++i], &opt.cfg.nx, &opt.cfg.ny, false, "--cells")) {
                fa_die("cannot parse --cells nx,ny");
            }
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            opt.cfg.steps = fa_parse_u32_strict(argv[++i], "--steps");
        } else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc) {
            opt.cfg.dt = fa_parse_float_strict(argv[++i], "--dt");
            opt.dt_supplied = true;
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            opt.input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            opt.output_path = argv[++i];
            opt.dump_csv = true;
        } else if (strcmp(argv[i], "--dump-bin") == 0) {
            opt.dump_bin = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opt.binary_output_path = argv[++i];
            }
        } else if (strcmp(argv[i], "--dump") == 0) {
            opt.dump_csv = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                opt.output_path = argv[++i];
            }
        } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
            ++i;
            bool found = false;
            for (int k = 0; k < FA_KERNEL_COUNT_; ++k) {
                if (strcmp(argv[i], FA_KERNEL_NAMES[k]) == 0) {
                    opt.kernel = (fa_kernel_t)k;
                    found = true;
                }
            }
            if (!found) {
                fa_die("unknown --kernel variant");
            }
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            if (!fa_parse_pair(argv[++i], &opt.block_x, &opt.block_y, true, "--block")) {
                fa_die("cannot parse --block x[,y]");
            }
            opt.block_supplied = true;
        } else if (strcmp(argv[i], "--tile-x") == 0 && i + 1 < argc) {
            opt.tile_x = fa_parse_u32_strict(argv[++i], "--tile-x");
        } else if (strcmp(argv[i], "--tile-y") == 0 && i + 1 < argc) {
            opt.tile_y = fa_parse_u32_strict(argv[++i], "--tile-y");
        } else if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "on") == 0) {
                opt.graph = true;
            } else if (strcmp(argv[i], "off") == 0) {
                opt.graph = false;
            } else {
                fa_die("--graph must be on or off");
            }
        } else {
            fa_dief("unknown argument: %s", argv[i]);
        }
    }

    if (!opt.block_supplied) {
        opt.block_x = 32;
        opt.block_y = opt.kernel == FA_KERNEL_SPLIT ? 4 : 8;
    }
    if (opt.tile_x == 0) {
        opt.tile_x = opt.kernel == FA_KERNEL_STEP ? 40 : 32;
    }
    if (opt.tile_y == 0) {
        opt.tile_y = opt.kernel == FA_KERNEL_STEP ? 40 : 16;
    }
    if ((uint64_t)opt.block_x * opt.block_y > 1024) {
        fa_die("block exceeds 1024 threads");
    }
    fa_validate_cfg(&opt.cfg, opt.dt_supplied);
    fa_validate_working_set(&opt.cfg);

    const double t_process_start = fa_now_seconds();
    fa_fields_t fields;
    if (opt.input_path) {
        fa_read_raw_input(opt.input_path, &opt.cfg, &fields);
    } else {
        fa_fields_alloc(&fields, &opt.cfg);
        fa_generate_analytic(&opt.cfg, &fields);
    }
    fa_check_finite(fields.base, fields.scalar_len, "initial scalar");
    fa_check_finite(fields.vx, fields.xface_len, "initial vx");
    fa_check_finite(fields.vy, fields.yface_len, "initial vy");
    const double t_setup_end = fa_now_seconds();
    fa_run_stats_t stats;
    if (opt.cuda) {
        fa_run_cuda(&opt, &fields, &stats);
    } else {
        fa_run_cpu(&opt, &fields, &stats);
    }

    if (opt.dump_csv) {
        if (!opt.output_path) {
            opt.output_path = (char *)"final.csv";
        }
        fa_write_csv(opt.output_path, &opt.cfg, &fields);
    }
    if (opt.dump_bin) {
        if (!opt.binary_output_path) {
            opt.binary_output_path = (char *)"final.bin";
        }
        fa_write_binary(opt.binary_output_path, &fields);
    }
    if (!stats.final_all_finite) {
        fa_die("final field contains non-finite values");
    }
    const double t_process_end = fa_now_seconds();
    fprintf(stderr,
        "[startup] setup=%.6f init=%.6f solve=%.6f total_elapsed=%.6f\n",
        t_setup_end - t_process_start, stats.init_seconds, stats.solve_seconds,
        t_process_end - t_process_start);
    fprintf(stdout,
        "RESULT mode=%s solve_wall=%.6f gpu_busy=%.6f cells=%u,%u steps=%u dt=%.9g "
        "kernel=%s block=%u,%u tile=%u,%u graph=%s\n",
        opt.cuda ? "cuda" : "cpu", stats.solve_seconds, stats.gpu_busy,
        opt.cfg.nx, opt.cfg.ny, opt.cfg.steps, opt.cfg.dt,
        opt.cuda ? FA_KERNEL_NAMES[opt.kernel] : "-",
        opt.block_x, opt.block_y, opt.tile_x, opt.tile_y, opt.graph ? "on" : "off");
    fa_fields_free(&fields);
    return 0;
}
