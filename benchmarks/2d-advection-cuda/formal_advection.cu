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
    FA_KERNEL_PAIR,
    FA_KERNEL_MARCH,
    FA_KERNEL_LEAN,
    FA_KERNEL_AUTO,
    FA_KERNEL_COUNT_
} fa_kernel_t;

static const char *FA_KERNEL_NAMES[FA_KERNEL_COUNT_] = {
    "split",
    "fused",
    "tiled",
    "prefetch",
    "regy",
    "regyf",
    "pair",
    "march",
    "lean",
    "auto",
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
    uint32_t frame_parts;
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

// Padded device copies of base (double-buffered) and the velocities used by
// "march" and "lean" (plus the stage buffers and accumulator of "lean"):
// FA_M_PAD ghost points (zeros) on every side, pitches rounded up to 32
// floats; the pointers address element (0, 0).
constexpr int FA_M_PAD = 32;
typedef struct {
    float *alloc[7];
    float *base[2];
    float *vx;
    float *vy;
    float *stage[2];    // "lean" only
    float *acc;         // "lean" only
    int ps;
    int pf;
} fa_mfields_t;

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
    cudaStream_t mside[3];
    cudaEvent_t mjoin[3];
    fa_mfields_t m;
    int march_parts;
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

// Frame tiles of "regyf" split into TY / 8 parts of 32 x 8 cells, one
// block of 32 x 8 threads each: the general path has about twice the
// latency of the interior one, and shorter blocks keep it off the critical
// path when the interior kernel is short (small grids).
template <int TY, int STAGE>
__global__ void __launch_bounds__(256)
fa_regy_frame(const float *__restrict__ stage_in, float *base, float *stage_out,
    float *accum, const float *__restrict__ vx, const float *__restrict__ vy,
    fa_stage_params_t p, fa_tiles_t tiles)
{
    constexpr int TYF = 8;
    constexpr int PARTS = TY / TYF;
    static_assert(TY % TYF == 0, "frame parts need TY = 8k");
    __shared__ float s[TYF + 8][40];
    __shared__ float fxd[TYF][37];
    __shared__ float fxa[TYF][37];
    __shared__ float fyd[TYF + 5][32];
    __shared__ float fya[TYF + 5][32];
    int bx, by;
    fa_frame_tile(tiles, (int)blockIdx.x / PARTS, bx, by);
    const int part = (int)blockIdx.x % PARTS;
    fa_tile_general<TYF, 8, STAGE>(stage_in, base, stage_out, accum, vx, vy, p,
        1 + bx * 32, 1 + by * TY + part * TYF, s, fxd, fxa, fyd, fya);
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
        if (TY % 8 == 0) {
            fa_regy_frame<TY % 8 == 0 ? TY : 8, STAGE><<<frame * (TY / 8), dim3(32, 8, 1), 0,
                d->side>>>(in, d->base, out, d->accum, d->vx, d->vy, p, t);
        } else {
            fa_regy_stage<TY, BY, STAGE, 2><<<frame, block, 0, d->side>>>(in, d->base, out,
                d->accum, d->vx, d->vy, p, t);
        }
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

// Parameters of the kernels that advance whole RK4 stage pairs.
struct fa_step_params_t {
    int nx;
    int ny;
    float inv_dx;
    float inv_dy;
    float dt;
    float dt_half;
    float dt_over_6;
};

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

// One-sided first and last rows of the divergence (the i == 0 and last
// branches of fa_t_div), on face values held in registers.
__device__ __forceinline__ float fa_div_first(float f0, float f1, float f2, float f3,
    float f4, float inv_h)
{
    return (((-11.0f / 12.0f) * inv_h) * f0 +
            ((17.0f / 24.0f) * inv_h) * f1 +
            ((3.0f / 8.0f) * inv_h) * f2 +
            ((-5.0f / 24.0f) * inv_h) * f3 +
            ((1.0f / 24.0f) * inv_h) * f4);
}

__device__ __forceinline__ float fa_div_last(float f0, float f1, float f2, float f3,
    float f4, float inv_h)
{
    return (((-1.0f / 24.0f) * inv_h) * f0 +
            ((5.0f / 24.0f) * inv_h) * f1 +
            ((-3.0f / 8.0f) * inv_h) * f2 +
            ((-17.0f / 24.0f) * inv_h) * f3 +
            ((11.0f / 12.0f) * inv_h) * f4);
}

// 0.5f * fa_grad_interior(...) with the exact factor 0.5 folded into the
// coefficients: the same value bit for bit, one multiplication less.
__device__ __forceinline__ float fa_hgrad_interior(float a, float b, float c, float d,
    float inv_h)
{
    return (((0.5f / 24.0f) * inv_h) * a + ((-0.5f * 9.0f / 8.0f) * inv_h) * b +
            ((0.5f * 9.0f / 8.0f) * inv_h) * c + ((-0.5f / 24.0f) * inv_h) * d);
}

// Diffusive and advective face values of the five boundary-side faces from
// seven consecutive rows (or columns) `v`, with the general operator rows.
// TOP: v = points 0..6, faces 0..4.  Otherwise (bottom/right): v = points
// n-6..n (n = last point), faces n-5..n-1.
template <bool TOP>
__device__ __forceinline__ void fa_edge_faces(const float (&v)[7], const float (&vel)[5],
    float inv_h, float (&fd)[5], float (&fa)[5])
{
    const auto at = [&](int k) { return v[TOP ? k : k + 2]; };
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        // Face index within a virtual line of 5 + 2 points: the one-sided
        // faces are the first two (TOP) or the last two (bottom).
        if (TOP && q < 2) {
            fd[q] = 0.5f * fa_t_grad(at, 5, q, inv_h);
            fa[q] = vel[q] * fa_t_interp(at, 5, q);
        } else if (!TOP && q >= 3) {
            fd[q] = 0.5f * fa_t_grad(at, 5, q - 1, inv_h);
            fa[q] = vel[q] * fa_t_interp(at, 5, q - 1);
        } else {
            const int o = TOP ? q - 1 : q;      // first point of the interior row
            fd[q] = 0.5f * fa_grad_interior(v[o], v[o + 1], v[o + 2], v[o + 3], inv_h);
            fa[q] = vel[q] * fa_interp_interior(v[o], v[o + 1], v[o + 2], v[o + 3]);
        }
    }
}

// Variant "march": one kernel per timestep with all four RK stages fused
// (temporal blocking).  A block of NT = 32*NW threads owns a strip of
// W = 2*NT - 26 columns and a segment of H rows.  Thread t holds the two
// adjacent columns c0 - 13 + 2t and c0 - 12 + 2t, i.e. the strip plus the
// 4 x 3 column halo the four stages need.  The block marches down its
// segment in chunks of R = 8 rows, starting 12 rows above it: per chunk it
// loads R new rows of base and pushes them through the four stages, stage s
// producing rows 3s behind the newest input rows.  Stage values, the RK
// accumulator and the stage rates never leave the chip: global traffic per
// timestep is one read of base and of the velocities per stage (mostly L1
// hits) and one write of the new base, which is double-buffered because
// neighbouring blocks read its halo.  Values outside the shrinking valid
// region are garbage that is never stored.
//
// y direction in registers: every thread carries, per stage and column, the
// last three stage values and y faces from chunk to chunk and evaluates R new
// y faces per chunk.  x direction through shared memory: rows and faces move
// as float2, each thread keeps its own two values and faces in registers.
//
// The fields use a padded device layout (FA_M_PAD ghost points on every
// side, even pitches): loads never need bounds checks, and the column pair
// of a thread is 8-byte aligned, so it moves with one 64-bit load.
//
// Boundaries (frame blocks only, FRAME = true): the boundary-side x faces
// and the x parts of cells 1 and nx are recomputed from the general rows by
// the block in parallel after the interior-row pass.  In y the one-sided
// rows reach five cells; a block containing the top (YM = 1) starts its march
// at row -14 and a block whose dependency cone reaches the bottom (YM = 2) at
// a row r0 = ny - 1 (mod 8), which puts cells 1..3 (ny-2..ny) of every stage
// into one chunk whose window holds the seven rows their general rows need
// (at the bottom up to three rows are captured from the previous chunk).
// These cells take their y part from the general rows.
constexpr int FA_M_R = 8;

struct fa_mptr_t {
    const float *bin;   // element (0, 0) of the padded fields
    float *bout;
    const float *vx;
    const float *vy;
    int ps;             // pitch of the scalars and vy
    int pf;             // pitch of vx
};

struct fa_march_geo_t {
    int c;              // global column of the thread's first column (even)
    int cl;             // global column of local column 0
    int r0;             // first input row of the march
    int y_lo;           // owned rows [y_lo, y_hi), columns [c_lo, c_hi)
    int y_hi;
    int c_lo;
    int c_hi;
    bool xgen;          // the strip contains boundary-side columns
};

// Top/bottom special chunks: stage S handles the cells next to the boundary
// in the chunk with b = fa_top_b(S) (top, r0 = -14) or b = ny - fa_bot_d(S)
// (bottom, ny - r0 = 1 (mod 8)).
__device__ constexpr int fa_top_b(int s) { return s == 1 ? -1 : s == 2 ? -4 : s == 3 ? 1 : -2; }
__device__ constexpr int fa_bot_d(int s) { return s == 1 ? 4 : s == 2 ? 7 : s == 3 ? 2 : 5; }

__device__ __forceinline__ float2 fa_ld2(const float *p)
{
    return __ldg(reinterpret_cast<const float2 *>(p));
}

template <int NW, int S, bool FRAME, int YM>
__device__ __forceinline__ void fa_m2_stage(int a, int t, const fa_march_geo_t &g,
    const fa_step_params_t &p, const fa_mptr_t &m, float (*xb)[64 * NW + 8],
    float (*fd)[64 * NW + 8], float (*fa)[64 * NW + 8], float (*xs)[2][FA_M_R],
    float (&cur)[2][FA_M_R], float (&accv)[2][FA_M_R], float (&rc)[2][3],
    float (&ydc)[2][3], float (&yac)[2][3], float (&ac)[2][3], float (&cap)[2][3],
    bool prefetch)
{
    constexpr int R = FA_M_R;
    constexpr int NT = 32 * NW;
    const int nx = p.nx;
    const int ny = p.ny;
    const int ps = m.ps;
    const int c = g.c;
    const int b = a - 3 * S;            // output rows [b, b + R)

    // All global loads of the stage first (their latency overlaps the
    // first barrier): y-face and x-face velocities and base.
    float2 vyl[R];
    float2 vxl[R];
    float2 bl[R];
#pragma unroll
    for (int q = 0; q < R; ++q) {
        vyl[q] = fa_ld2(m.vy + c + ps * (b + 1 + q));
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        vxl[q] = fa_ld2(m.vx + c + m.pf * (b + q));
    }
#pragma unroll
    for (int q = 0; q < R; ++q) {
        bl[q] = fa_ld2(m.bin + c + ps * (b + q));
    }

    // Stage-input rows [b, b + R + 3) (three carried) and y faces
    // [b - 2, b + R + 1) (three carried, R new; face f uses rows f-1 .. f+2).
    float w[2][R + 3];
    float yd[2][R + 3];
    float ya[2][R + 3];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            w[e][q] = rc[e][q];
            yd[e][q] = ydc[e][q];
            ya[e][q] = yac[e][q];
        }
#pragma unroll
        for (int q = 0; q < R; ++q) {
            w[e][q + 3] = cur[e][q];
        }
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            rc[e][q] = w[e][R + q];
        }
    }
    if (S == 4 && prefetch) {
        // The next chunk's input rows (cur is dead once w is built).
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float2 v = fa_ld2(m.bin + c + ps * (a + R + r));
            cur[0][r] = v.x;
            cur[1][r] = v.y;
        }
    }

    // y parts of the cells next to a y boundary (general rows).
    float sdy[2][3];
    float say[2][3];
    constexpr int BT = fa_top_b(S);
    constexpr int DB = fa_bot_d(S);
    if (YM == 2 && DB < 5 && b == ny - DB - R) {
        // Rows ny-5 .. ny-DB-1 leave the window before the special chunk.
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int i = 0; i < 5 - DB; ++i) {
                cap[e][i] = w[e][DB + R - 5 + i];
            }
        }
    }
    const bool top_chunk = YM == 1 && b == BT;
    const bool bot_chunk = YM == 2 && b == ny - DB;
    if (top_chunk || bot_chunk) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int ce = c + e;
            float v[7];
            float vel[5];
            float ffd[5];
            float ffa[5];
            if (YM == 1) {
                // Rows 0..6: row 0 is the fixed boundary (= base).
                v[0] = __ldg(m.bin + ce);
#pragma unroll
                for (int i = 1; i < 7; ++i) {
                    v[i] = w[e][i - BT];
                }
#pragma unroll
                for (int q = 0; q < 5; ++q) {
                    vel[q] = __ldg(m.vy + ce + ps * q);
                }
                fa_edge_faces<true>(v, vel, p.inv_dy, ffd, ffa);
                // cells 1, 2, 3
                sdy[e][0] = fa_div_first(ffd[0], ffd[1], ffd[2], ffd[3], ffd[4], p.inv_dy);
                say[e][0] = fa_div_first(ffa[0], ffa[1], ffa[2], ffa[3], ffa[4], p.inv_dy);
                sdy[e][1] = fa_div_interior(ffd[0], ffd[1], ffd[2], ffd[3], p.inv_dy);
                say[e][1] = fa_div_interior(ffa[0], ffa[1], ffa[2], ffa[3], p.inv_dy);
                sdy[e][2] = fa_div_interior(ffd[1], ffd[2], ffd[3], ffd[4], p.inv_dy);
                say[e][2] = fa_div_interior(ffa[1], ffa[2], ffa[3], ffa[4], p.inv_dy);
            } else {
                // Rows ny-5 .. ny+1: row ny+1 is the fixed boundary.
#pragma unroll
                for (int i = 0; i < 6; ++i) {
                    // row ny - 5 + i is at window index DB - 5 + i
                    v[i] = (DB - 5 + i >= 0) ? w[e][max(DB - 5 + i, 0)] :
                        cap[e][min(i, 2)];
                }
                v[6] = __ldg(m.bin + ce + ps * (ny + 1));
#pragma unroll
                for (int q = 0; q < 5; ++q) {
                    vel[q] = __ldg(m.vy + ce + ps * (ny - 4 + q));
                }
                fa_edge_faces<false>(v, vel, p.inv_dy, ffd, ffa);
                // cells ny-2, ny-1, ny (faces ny-4 .. ny = ffd[0..4])
                sdy[e][0] = fa_div_interior(ffd[0], ffd[1], ffd[2], ffd[3], p.inv_dy);
                say[e][0] = fa_div_interior(ffa[0], ffa[1], ffa[2], ffa[3], p.inv_dy);
                sdy[e][1] = fa_div_interior(ffd[1], ffd[2], ffd[3], ffd[4], p.inv_dy);
                say[e][1] = fa_div_interior(ffa[1], ffa[2], ffa[3], ffa[4], p.inv_dy);
                sdy[e][2] = fa_div_last(ffd[0], ffd[1], ffd[2], ffd[3], ffd[4], p.inv_dy);
                say[e][2] = fa_div_last(ffa[0], ffa[1], ffa[2], ffa[3], ffa[4], p.inv_dy);
            }
        }
    }

    // x faces of rows [b, b + R): faces c and c + 1 of this thread.
#pragma unroll
    for (int r = 0; r < R; ++r) {
        *reinterpret_cast<float2 *>(&xb[r][2 * t + 4]) = make_float2(w[0][r], w[1][r]);
    }
    __syncthreads();
    // y faces (after the barrier, so that the velocity loads have landed).
#pragma unroll
    for (int q = 0; q < R; ++q) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            yd[e][q + 3] = fa_hgrad_interior(w[e][q], w[e][q + 1], w[e][q + 2], w[e][q + 3],
                p.inv_dy);
            ya[e][q + 3] = (e ? vyl[q].y : vyl[q].x) * fa_interp_interior(w[e][q],
                w[e][q + 1], w[e][q + 2], w[e][q + 3]);
        }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            ydc[e][q] = yd[e][R + q];
            yac[e][q] = ya[e][R + q];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float sm = xb[r][2 * t + 3];
        const float2 sp = *reinterpret_cast<const float2 *>(&xb[r][2 * t + 6]);
        const float2 v = vxl[r];
        const float d0 = fa_hgrad_interior(sm, w[0][r], w[1][r], sp.x, p.inv_dx);
        const float d1 = fa_hgrad_interior(w[0][r], w[1][r], sp.x, sp.y, p.inv_dx);
        const float a0 = v.x * fa_interp_interior(sm, w[0][r], w[1][r], sp.x);
        const float a1 = v.y * fa_interp_interior(w[0][r], w[1][r], sp.x, sp.y);
        *reinterpret_cast<float2 *>(&fd[r][2 * t + 4]) = make_float2(d0, d1);
        *reinterpret_cast<float2 *>(&fa[r][2 * t + 4]) = make_float2(a0, a1);
    }
    if (FRAME && g.xgen) {
        // Boundary-side x faces (0, 1, nx-1, nx) from the general rows, one
        // (face, row) item per thread, then the x parts of cells 1 and nx.
        static_assert(NT >= 4 * R, "fix-up needs 4R threads");
        __syncthreads();
        if (t < 4 * R) {
            const int k = t / R;
            const int r = t % R;
            const int f = k == 0 ? 0 : k == 1 ? 1 : k == 2 ? nx - 1 : nx;
            const int l = f - g.cl;
            if (l >= 0 && l < 2 * NT) {
                const fa_sline_t row = {&xb[r][0], g.cl - 4, 1};
                const float v = __ldg(m.vx + f + m.pf * (b + r));
                fd[r][l + 4] = 0.5f * fa_t_grad(row, nx + 2, f, p.inv_dx);
                fa[r][l + 4] = v * fa_t_interp(row, nx + 2, f);
            }
        }
        __syncthreads();
        if (t < 4 * R) {
            const int which = t / (2 * R);
            const int kind = (t / R) % 2;
            const int r = t % R;
            const int cell = which ? nx : 1;
            const int l = cell - g.cl;
            if (l >= 0 && l < 2 * NT) {
                const fa_sline_t row = {kind ? &fa[r][0] : &fd[r][0], g.cl - 4, 1};
                xs[which][kind][r] = fa_t_div(row, nx + 1, cell - 1, p.inv_dx);
            }
        }
    }
    __syncthreads();

    // Accumulator rows [b, b + R): three carried, R - 3 from the previous
    // stage's current outputs.
    float aw[2][R];
    if (S > 1) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                aw[e][q] = ac[e][q];
            }
#pragma unroll
            for (int q = 3; q < R; ++q) {
                aw[e][q] = accv[e][q - 3];
            }
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                ac[e][q] = accv[e][R - 3 + q];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int j = b + r;
        const float2 dm = *reinterpret_cast<const float2 *>(&fd[r][2 * t + 2]);
        const float dp = fd[r][2 * t + 6];
        const float2 am = *reinterpret_cast<const float2 *>(&fa[r][2 * t + 2]);
        const float ap = fa[r][2 * t + 6];
        const float2 d0 = *reinterpret_cast<const float2 *>(&fd[r][2 * t + 4]);
        const float2 a0 = *reinterpret_cast<const float2 *>(&fa[r][2 * t + 4]);
        float diffx[2], advx[2];
        diffx[0] = fa_div_interior(dm.x, dm.y, d0.x, d0.y, p.inv_dx);
        diffx[1] = fa_div_interior(dm.y, d0.x, d0.y, dp, p.inv_dx);
        advx[0] = fa_div_interior(am.x, am.y, a0.x, a0.y, p.inv_dx);
        advx[1] = fa_div_interior(am.y, a0.x, a0.y, ap, p.inv_dx);
        const float2 bj2 = bl[r];
        float out[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int ce = c + e;
            if (FRAME && (ce == 1 || ce == nx)) {
                diffx[e] = xs[ce == 1 ? 0 : 1][0][r];
                advx[e] = xs[ce == 1 ? 0 : 1][1][r];
            }
            float diffy = fa_div_interior(yd[e][r], yd[e][r + 1], yd[e][r + 2],
                yd[e][r + 3], p.inv_dy);
            float advy = fa_div_interior(ya[e][r], ya[e][r + 1], ya[e][r + 2],
                ya[e][r + 3], p.inv_dy);
            if (YM == 1 && r >= 1 - BT && r <= 3 - BT && top_chunk) {
                diffy = sdy[e][r - (1 - BT)];
                advy = say[e][r - (1 - BT)];
            }
            if (YM == 2 && r >= DB - 2 && r <= DB && bot_chunk) {
                diffy = sdy[e][r - (DB - 2)];
                advy = say[e][r - (DB - 2)];
            }
            const float rhs = (diffx[e] + diffy) - (advx[e] + advy);
            const float bj = e ? bj2.y : bj2.x;
            // Boundary and ghost points keep base (frame blocks only).
            const bool cell = !FRAME || (ce >= 1 && ce <= nx && (YM == 0 || (j >= 1 && j <= ny)));
            if (S == 1) {
                accv[e][r] = rhs;
                cur[e][r] = cell ? bj + p.dt_half * rhs : bj;
            } else if (S == 2) {
                accv[e][r] = aw[e][r] + 2.0f * rhs;
                cur[e][r] = cell ? bj + p.dt_half * rhs : bj;
            } else if (S == 3) {
                accv[e][r] = aw[e][r] + 2.0f * rhs;
                cur[e][r] = cell ? bj + p.dt * rhs : bj;
            } else {
                out[e] = bj + p.dt_over_6 * (aw[e][r] + rhs);
            }
        }
        if (S == 4 && j >= g.y_lo && j < g.y_hi) {
            const bool o0 = c >= g.c_lo && c < g.c_hi;
            const bool o1 = c + 1 >= g.c_lo && c + 1 < g.c_hi;
            if (o0 && o1) {
                *reinterpret_cast<float2 *>(m.bout + c + ps * j) = make_float2(out[0], out[1]);
            } else if (o0) {
                m.bout[c + ps * j] = out[0];
            } else if (o1) {
                m.bout[c + 1 + ps * j] = out[1];
            }
        }
    }
}

// The march of one block: chunks k = 0 .. K-1; stage s is skipped while its
// current and next chunks are below its valid rows ((k + 2) R <= 6 s, plus
// the extra warm-up rows of r0).
template <int NW, bool FRAME, int YM>
__device__ __forceinline__ void fa_m2_run(const fa_march_geo_t &g, int K,
    const fa_step_params_t &p, const fa_mptr_t &m, float (*xb)[64 * NW + 8],
    float (*fd)[64 * NW + 8], float (*fa)[64 * NW + 8])
{
    constexpr int R = FA_M_R;
    __shared__ float xs[2][2][R];       // x parts of cells 1 and nx (frame)
    const int t = threadIdx.x;
    float rc[4][2][3], ydc[4][2][3], yac[4][2][3], ac[4][2][3], cap[4][2][3];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                rc[s][e][q] = 0.0f;
                ydc[s][e][q] = 0.0f;
                yac[s][e][q] = 0.0f;
                ac[s][e][q] = 0.0f;
                cap[s][e][q] = 0.0f;
            }
        }
    }
    const int skip = g.y_lo - 12 - g.r0;
    float cur[2][R];
    float accv[2][R];
    bool have = false;                  // cur already holds the chunk input
    for (int k = 0; k < K; ++k) {
        const int a = g.r0 + k * R;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (!have) {
                const float2 v = fa_ld2(m.bin + g.c + m.ps * (a + r));
                cur[0][r] = v.x;
                cur[1][r] = v.y;
            }
            accv[0][r] = 0.0f;
            accv[1][r] = 0.0f;
        }
        const int kr = (k + 2) * R - skip;
        const bool pf = k + 1 < K;
        fa_m2_stage<NW, 1, FRAME, YM>(a, t, g, p, m, xb, fd, fa, xs, cur, accv, rc[0],
            ydc[0], yac[0], ac[0], cap[0], false);
        if (kr > 12) {
            fa_m2_stage<NW, 2, FRAME, YM>(a, t, g, p, m, xb, fd, fa, xs, cur, accv, rc[1],
                ydc[1], yac[1], ac[1], cap[1], false);
        }
        if (kr > 18) {
            fa_m2_stage<NW, 3, FRAME, YM>(a, t, g, p, m, xb, fd, fa, xs, cur, accv, rc[2],
                ydc[2], yac[2], ac[2], cap[2], false);
        }
        if (kr > 24) {
            fa_m2_stage<NW, 4, FRAME, YM>(a, t, g, p, m, xb, fd, fa, xs, cur, accv, rc[3],
                ydc[3], yac[3], ac[3], cap[3], pf);
        }
        have = kr > 24 && pf;
    }
}

template <int NW>
__device__ __forceinline__ void fa_m2_smem_init(float (*xb)[64 * NW + 8],
    float (*fd)[64 * NW + 8], float (*fa)[64 * NW + 8])
{
    constexpr int XS = 64 * NW + 8;
    for (int q = threadIdx.x; q < FA_M_R * XS; q += 32 * NW) {
        (&xb[0][0])[q] = 0.0f;
        (&fd[0][0])[q] = 0.0f;
        (&fa[0][0])[q] = 0.0f;
    }
    __syncthreads();
}

// Interior blocks: every point of the dependency cone uses the interior rows.
template <int NW>
__global__ void __launch_bounds__(32 * NW)
fa_march_fast(fa_mptr_t m, fa_step_params_t p, fa_tiles_t tiles, int H)
{
    constexpr int W = 64 * NW - 26;
    __shared__ __align__(16) float xb[FA_M_R][64 * NW + 8];
    __shared__ __align__(16) float fd[FA_M_R][64 * NW + 8];
    __shared__ __align__(16) float fa[FA_M_R][64 * NW + 8];
    fa_m2_smem_init<NW>(xb, fd, fa);
    const int c0 = 1 + (tiles.bx_lo + (int)blockIdx.x) * W;
    const int y0 = 1 + (tiles.by_lo + (int)blockIdx.y) * H;
    fa_march_geo_t g;
    g.cl = c0 - 13;
    g.c = g.cl + 2 * (int)threadIdx.x;
    g.r0 = y0 - 12;
    g.y_lo = y0;
    g.y_hi = y0 + H;
    g.c_lo = c0;
    g.c_hi = c0 + W;
    g.xgen = false;
    fa_m2_run<NW, false, 0>(g, (H + 24) / FA_M_R, p, m, xb, fd, fa);
}

// Frame blocks: strips and segments whose cone reaches a boundary, in three
// classes launched as separate kernels: the top segment row (YM = 1), the
// segment rows whose cone reaches the bottom (YM = 2, rows >= by_b0) and the
// remaining non-interior blocks (YM = 0).  The last strip is shifted to end
// at the last column (rounded to keep its columns pairs aligned) and writes
// only the columns it owns; segments are not shifted (the bottom march simply
// runs to row ny).
struct fa_march_map_t {
    fa_tiles_t t;       // interior rectangle (empty: bx_lo > bx_hi)
    int by_b0;          // first segment row of the bottom class
    int parts;          // frame tiles are split into this many row parts
};

__device__ __forceinline__ void fa_march_frame_tile(const fa_march_map_t &mm, int ym,
    int k, int &bx, int &by)
{
    const int nbx = mm.t.nbx;
    if (ym == 1) {
        bx = k;
        by = 0;
    } else if (ym == 2) {
        bx = k % nbx;
        by = mm.by_b0 + k / nbx;
    } else {
        // Rows [1, by_b0): the blocks beside the interior rectangle, then
        // the full rows below it.
        const int nint = mm.t.bx_hi - mm.t.bx_lo + 1;
        const int rows = mm.t.by_hi - mm.t.by_lo + 1;
        const bool has = nint > 0 && rows > 0;
        const int side = nbx - nint;
        if (has && k < rows * side) {
            by = mm.t.by_lo + k / side;
            const int c = k % side;
            bx = c < mm.t.bx_lo ? c : mm.t.bx_hi + 1 + (c - mm.t.bx_lo);
        } else {
            if (has) {
                k -= rows * side;
            }
            bx = k % nbx;
            by = (has ? mm.t.by_hi + 1 : 1) + k / nbx;
        }
    }
}

template <int NW, int YM>
__global__ void __launch_bounds__(32 * NW)
fa_march_frame(fa_mptr_t m, fa_step_params_t p, fa_march_map_t mm, int H)
{
    constexpr int R = FA_M_R;
    constexpr int W = 64 * NW - 26;
    __shared__ __align__(16) float xb[R][64 * NW + 8];
    __shared__ __align__(16) float fd[R][64 * NW + 8];
    __shared__ __align__(16) float fa[R][64 * NW + 8];
    fa_m2_smem_init<NW>(xb, fd, fa);
    const int nx = p.nx;
    const int ny = p.ny;
    int bx, by;
    fa_march_frame_tile(mm, YM, (int)blockIdx.x / mm.parts, bx, by);
    const int part = (int)blockIdx.x % mm.parts;
    const int hp = (H + mm.parts - 1) / mm.parts;
    fa_march_geo_t g;
    g.c_lo = 1 + bx * W;
    g.c_hi = min(g.c_lo + W, nx + 1);
    int c0 = min(g.c_lo, max(1, nx - W + 1));
    c0 += 1 - (c0 & 1);                 // odd: even column pairs
    g.cl = c0 - 13;
    g.c = g.cl + 2 * (int)threadIdx.x;
    g.xgen = g.cl <= 1 || g.cl + 64 * NW - 1 >= nx - 1;
    const int tile_hi = min(1 + (by + 1) * H, ny + 1);
    g.y_lo = 1 + by * H + part * hp;
    g.y_hi = min(g.y_lo + hp, tile_hi);
    if (g.y_lo >= g.y_hi) {
        return;
    }
    // r0: 12 warm-up rows, aligned to -14 (mod 8) at the top and to ny - 1
    // (mod 8) at the bottom so that the special chunks line up; the march
    // ends once stage 4 has passed y_hi (and, if the cone reaches the
    // bottom, the stage-4 special chunk a = ny + 7).
    const int lo = g.y_lo - 12;
    const int ref = YM == 1 ? -14 : YM == 2 ? ny - 1 : lo;
    g.r0 = YM == 1 && g.y_lo == 1 ? -14 : lo - (((lo - ref) % R) + R) % R;
    int a_end = g.y_hi + 4;
    if (YM == 2 && g.y_hi >= ny - 10) {
        a_end = max(a_end, ny + 7);
    }
    const int K = (a_end - g.r0 + R - 1) / R + 1;
    fa_m2_run<NW, true, YM>(g, K, p, m, xb, fd, fa);
}

// Tiling of "march": strips of W = 64*NW - 26 columns, segments of H rows.
// Interior block (bx, by): c0 = 1 + bx*W >= 13, c0 + W <= nx - 11 and
// y0 = 1 + by*H >= 25 (keeps the warm-up rows inside the domain),
// y0 + H <= ny - 11.  Returns false if the grid is too small (a segment
// would contain both y boundaries' cones).
template <int NW>
static bool fa_march_tiles(int nx, int ny, int H, fa_march_map_t *mm, int *interior,
    int *frame)
{
    constexpr int W = 64 * NW - 26;
    if (ny < H + 12 || H % FA_M_R != 0 || H < 24) {
        return false;
    }
    fa_tiles_t *t = &mm->t;
    *t = fa_make_tiles((nx + W - 1) / W, (ny + H - 1) / H, (nx - 12 - W) / W,
        (ny - 12 - H) / H, interior, frame);
    const int lo = (24 + H - 1) / H;
    if (*interior > 0 && lo > t->by_lo) {
        if (lo > t->by_hi) {
            *t = fa_make_tiles(t->nbx, t->nby, 0, 0, interior, frame);
        } else {
            t->by_lo = lo;
            *interior = (t->bx_hi - t->bx_lo + 1) * (t->by_hi - t->by_lo + 1);
            *frame = t->nbx * t->nby - *interior;
        }
    }
    // Bottom class: segment rows with y_hi >= ny - 10.
    mm->by_b0 = t->nby - 1;
    while (mm->by_b0 > 1 && 1 + mm->by_b0 * H >= ny - 10) {
        --mm->by_b0;
    }
    return true;
}

template <int NW>
static void fa_launch_march(const fa_config_t *cfg, fa_device_t *d, int parity, int H,
    cudaStream_t s)
{
    const int nx = (int)cfg->nx;
    const int ny = (int)cfg->ny;
    int interior, frame;
    fa_march_map_t mm;
    if (!fa_march_tiles<NW>(nx, ny, H, &mm, &interior, &frame)) {
        fa_die("march: grid too small for --tile-y");
    }
    mm.parts = d->march_parts;
    const int ntop = mm.t.nbx * mm.parts;
    const int nbot = (mm.t.nby - mm.by_b0) * mm.t.nbx * mm.parts;
    const int nmid = (frame - mm.t.nbx - (mm.t.nby - mm.by_b0) * mm.t.nbx) * mm.parts;
    const fa_step_params_t p = fa_step_params(cfg);
    fa_mptr_t m;
    m.bin = d->m.base[parity];
    m.bout = d->m.base[parity ^ 1];
    m.vx = d->m.vx;
    m.vy = d->m.vy;
    m.ps = d->m.ps;
    m.pf = d->m.pf;
    const dim3 block(32 * NW, 1, 1);
    fa_cuda_check(cudaEventRecord(d->fork, s), "fork record");
    for (int q = 0; q < 3; ++q) {
        fa_cuda_check(cudaStreamWaitEvent(d->mside[q], d->fork, 0), "fork wait");
    }
    fa_march_frame<NW, 1><<<ntop, block, 0, d->mside[0]>>>(m, p, mm, H);
    fa_march_frame<NW, 2><<<nbot, block, 0, d->mside[1]>>>(m, p, mm, H);
    if (nmid > 0) {
        fa_march_frame<NW, 0><<<nmid, block, 0, d->mside[2]>>>(m, p, mm, H);
    }
    if (interior > 0) {
        const dim3 grid(mm.t.bx_hi - mm.t.bx_lo + 1, mm.t.by_hi - mm.t.by_lo + 1, 1);
        fa_march_fast<NW><<<grid, block, 0, s>>>(m, p, mm.t, H);
    }
    for (int q = 0; q < 3; ++q) {
        fa_cuda_check(cudaEventRecord(d->mjoin[q], d->mside[q]), "join record");
        fa_cuda_check(cudaStreamWaitEvent(s, d->mjoin[q], 0), "join wait");
    }
}

// Variant "lean": one kernel per RK stage like "regyf", with the "march"
// thread layout.  The interior rectangle of the domain is cut into strips of
// W = 2*NT - 6 columns (NT threads, each owning two adjacent columns, plus a
// 3-column halo on each side) and segments of H rows.  A block marches down
// its segment in chunks of 8 rows: the y direction stays in registers (three
// stage values and three y faces carried from chunk to chunk, one new face
// per cell), the x direction goes through shared memory as float2.  The
// stage input is read with a 3-row halo above and below the segment only,
// every other field exactly once, all with aligned 64-bit loads from the
// padded layout, so the L2 traffic per stage is close to the 7 field
// accesses the method needs (read in, base, accumulator and both
// velocities, write the stage value and the accumulator).  The boundary
// bands (32 columns at each side, 8 rows at top and bottom) run concurrently
// as 32 x 8 tiles with the general operator rows.  base is double-buffered
// (stage 4 writes the other buffer) because neighbouring blocks read its
// halo.
struct fa_lptr_t {
    const float *in;
    const float *base;
    float *out;         // stage value, or the new base for stage 4
    float *acc;
    const float *vx;
    const float *vy;
    int ps;             // pitch of the scalars and vy
    int pf;             // pitch of vx
};

// Interior rectangle of "lean": columns [5, nx - 2), rows [4, ny - 2) (all
// cells whose operator rows are the interior ones, the first column kept
// odd for aligned column pairs).
struct fa_lean_geo_t {
    int col_lo;
    int col_hi;
    int row_lo;
    int row_hi;
};

static bool fa_lean_geo(int nx, int ny, fa_lean_geo_t *g)
{
    g->col_lo = 5;
    g->col_hi = nx - 2;
    g->row_lo = 4;
    g->row_hi = ny - 2;
    return g->col_hi > g->col_lo + 8 && g->row_hi > g->row_lo + 8;
}

template <int S>
__device__ __forceinline__ void fa_lean_update(float rhs, float bj, float aj,
    const fa_step_params_t &p, float &out, float &acc)
{
    if (S == 1) {
        acc = rhs;
        out = bj + p.dt_half * rhs;
    } else if (S == 2) {
        acc = aj + 2.0f * rhs;
        out = bj + p.dt_half * rhs;
    } else if (S == 3) {
        acc = aj + 2.0f * rhs;
        out = bj + p.dt * rhs;
    } else {
        out = bj + p.dt_over_6 * (aj + rhs);
    }
}

template <int NW, int S>
__global__ void __launch_bounds__(32 * NW)
fa_lean_stage(fa_lptr_t m, fa_step_params_t p, fa_lean_geo_t geo, int H)
{
    constexpr int R = FA_M_R;
    constexpr int NT = 32 * NW;
    constexpr int W = 2 * NT - 6;
    __shared__ __align__(16) float xb[R][2 * NT + 8];
    __shared__ __align__(16) float fd[R][2 * NT + 8];
    __shared__ __align__(16) float fa[R][2 * NT + 8];
    const int t = threadIdx.x;
    for (int q = t; q < R * (2 * NT + 8); q += NT) {
        (&xb[0][0])[q] = 0.0f;
        (&fd[0][0])[q] = 0.0f;
        (&fa[0][0])[q] = 0.0f;
    }
    __syncthreads();
    const int c0 = geo.col_lo + (int)blockIdx.x * W;     // odd
    const int y0 = geo.row_lo + (int)blockIdx.y * H;
    const int y1 = min(y0 + H, geo.row_hi);
    const int c_hi = min(c0 + W, geo.col_hi);
    const int c = c0 - 3 + 2 * t;                         // even
    const int ps = m.ps;
    // Prologue: carried rows [y0, y0 + 3) and y faces [y0 - 2, y0 + 1).
    float rc[2][3], ydc[2][3], yac[2][3];
    {
        float2 pr[6];
#pragma unroll
        for (int q = 0; q < 6; ++q) {
            pr[q] = fa_ld2(m.in + c + ps * (y0 - 3 + q));
        }
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            const float2 v = fa_ld2(m.vy + c + ps * (y0 - 2 + q));
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float s0 = e ? pr[q].y : pr[q].x;
                const float s1 = e ? pr[q + 1].y : pr[q + 1].x;
                const float s2 = e ? pr[q + 2].y : pr[q + 2].x;
                const float s3 = e ? pr[q + 3].y : pr[q + 3].x;
                ydc[e][q] = fa_hgrad_interior(s0, s1, s2, s3, p.inv_dy);
                yac[e][q] = (e ? v.y : v.x) * fa_interp_interior(s0, s1, s2, s3);
                rc[e][q] = s3;
            }
        }
    }
    const int K = (y1 - y0 + R - 1) / R;
    for (int k = 0; k < K; ++k) {
        const int b = y0 + k * R;       // output rows [b, b + R)
        // All global loads of the chunk first.
        float2 cur[R], vyl[R], vxl[R], bl[R], al[R];
#pragma unroll
        for (int q = 0; q < R; ++q) {
            cur[q] = fa_ld2(m.in + c + ps * (b + 3 + q));
            vyl[q] = fa_ld2(m.vy + c + ps * (b + 1 + q));
            vxl[q] = fa_ld2(m.vx + c + m.pf * (b + q));
            bl[q] = fa_ld2(m.base + c + ps * (b + q));
            al[q] = S > 1 ? fa_ld2(m.acc + c + ps * (b + q)) : make_float2(0.0f, 0.0f);
        }
        float w[2][R + 3];
        float yd[2][R + 3];
        float ya[2][R + 3];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                w[e][q] = rc[e][q];
                yd[e][q] = ydc[e][q];
                ya[e][q] = yac[e][q];
            }
#pragma unroll
            for (int q = 0; q < R; ++q) {
                w[e][q + 3] = e ? cur[q].y : cur[q].x;
            }
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                rc[e][q] = w[e][R + q];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            *reinterpret_cast<float2 *>(&xb[r][2 * t + 4]) = make_float2(w[0][r], w[1][r]);
        }
        __syncthreads();
#pragma unroll
        for (int q = 0; q < R; ++q) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                yd[e][q + 3] = fa_hgrad_interior(w[e][q], w[e][q + 1], w[e][q + 2],
                    w[e][q + 3], p.inv_dy);
                ya[e][q + 3] = (e ? vyl[q].y : vyl[q].x) * fa_interp_interior(w[e][q],
                    w[e][q + 1], w[e][q + 2], w[e][q + 3]);
            }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                ydc[e][q] = yd[e][R + q];
                yac[e][q] = ya[e][R + q];
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float sm = xb[r][2 * t + 3];
            const float2 sp = *reinterpret_cast<const float2 *>(&xb[r][2 * t + 6]);
            const float d0 = fa_hgrad_interior(sm, w[0][r], w[1][r], sp.x, p.inv_dx);
            const float d1 = fa_hgrad_interior(w[0][r], w[1][r], sp.x, sp.y, p.inv_dx);
            const float a0 = vxl[r].x * fa_interp_interior(sm, w[0][r], w[1][r], sp.x);
            const float a1 = vxl[r].y * fa_interp_interior(w[0][r], w[1][r], sp.x, sp.y);
            *reinterpret_cast<float2 *>(&fd[r][2 * t + 4]) = make_float2(d0, d1);
            *reinterpret_cast<float2 *>(&fa[r][2 * t + 4]) = make_float2(a0, a1);
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int j = b + r;
            const float2 dm = *reinterpret_cast<const float2 *>(&fd[r][2 * t + 2]);
            const float2 d0 = *reinterpret_cast<const float2 *>(&fd[r][2 * t + 4]);
            const float dp = fd[r][2 * t + 6];
            const float2 am = *reinterpret_cast<const float2 *>(&fa[r][2 * t + 2]);
            const float2 a0 = *reinterpret_cast<const float2 *>(&fa[r][2 * t + 4]);
            const float ap = fa[r][2 * t + 6];
            float diffx[2], advx[2];
            diffx[0] = fa_div_interior(dm.x, dm.y, d0.x, d0.y, p.inv_dx);
            diffx[1] = fa_div_interior(dm.y, d0.x, d0.y, dp, p.inv_dx);
            advx[0] = fa_div_interior(am.x, am.y, a0.x, a0.y, p.inv_dx);
            advx[1] = fa_div_interior(am.y, a0.x, a0.y, ap, p.inv_dx);
            float o[2], na[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float diffy = fa_div_interior(yd[e][r], yd[e][r + 1], yd[e][r + 2],
                    yd[e][r + 3], p.inv_dy);
                const float advy = fa_div_interior(ya[e][r], ya[e][r + 1], ya[e][r + 2],
                    ya[e][r + 3], p.inv_dy);
                const float rhs = (diffx[e] + diffy) - (advx[e] + advy);
                fa_lean_update<S>(rhs, e ? bl[r].y : bl[r].x, e ? al[r].y : al[r].x, p,
                    o[e], na[e]);
            }
            if (j < y1) {
                const bool o0 = c >= c0 && c < c_hi;
                const bool o1 = c + 1 >= c0 && c + 1 < c_hi;
                float *po = m.out + c + ps * j;
                float *pa = m.acc + c + ps * j;
                if (o0 && o1) {
                    *reinterpret_cast<float2 *>(po) = make_float2(o[0], o[1]);
                    if (S < 4) {
                        *reinterpret_cast<float2 *>(pa) = make_float2(na[0], na[1]);
                    }
                } else if (o0 || o1) {
                    const int e = o0 ? 0 : 1;
                    po[e] = o[e];
                    if (S < 4) {
                        pa[e] = na[e];
                    }
                }
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Boundary bands of "lean": tiles with the general operator rows
// (fa_tile_general on the padded layout), clipped to their band.  Bands:
// top rows [1, row_lo) and bottom rows [row_hi, ny] in 64 x 3 tiles, left
// columns [1, col_lo) and right columns [col_hi, nx] (between row_lo and
// row_hi) in 4 x 48 tiles; with an empty interior the whole domain is one
// band of 64 x 3 tiles.
struct fa_lean_bands_t {
    int n;              // number of bands
    int c_lo[4], c_hi[4], r_lo[4], r_hi[4];
    int tiles_x[4];
    int first[5];       // first block of each band
};

static fa_lean_bands_t fa_lean_bands(int nx, int ny, bool sides, int tx, int ty)
{
    fa_lean_bands_t b;
    fa_lean_geo_t g;
    int k = 0;
    const auto add = [&](int c_lo, int c_hi, int r_lo, int r_hi) {
        if (c_hi <= c_lo || r_hi <= r_lo) {
            return;
        }
        b.c_lo[k] = c_lo;
        b.c_hi[k] = c_hi;
        b.r_lo[k] = r_lo;
        b.r_hi[k] = r_hi;
        b.tiles_x[k] = (c_hi - c_lo + tx - 1) / tx;
        ++k;
    };
    if (fa_lean_geo(nx, ny, &g)) {
        if (sides) {
            add(1, g.col_lo, g.row_lo, g.row_hi);
            add(g.col_hi, nx + 1, g.row_lo, g.row_hi);
        } else {
            add(1, nx + 1, 1, g.row_lo);
            add(1, nx + 1, g.row_hi, ny + 1);
        }
    } else if (!sides) {
        add(1, nx + 1, 1, ny + 1);
    }
    b.n = k;
    b.first[0] = 0;
    for (int q = 0; q < k; ++q) {
        b.first[q + 1] = b.first[q] + b.tiles_x[q] * ((b.r_hi[q] - b.r_lo[q] + ty - 1) / ty);
    }
    for (int q = k; q < 4; ++q) {
        b.first[q + 1] = b.first[k];
    }
    return b;
}

template <int STAGE, int TX, int TY>
__device__ __forceinline__ void fa_lean_tile(const fa_lptr_t &m, const fa_step_params_t &p,
    const fa_lean_bands_t &bands, int blk)
{
    constexpr int SW = TX + 8;
    constexpr int SH = TY + 8;
    constexpr int FXW = TX + 5;
    constexpr int FYH = TY + 5;
    __shared__ float s[SH][SW];
    __shared__ float fxd[TY][FXW];
    __shared__ float fxa[TY][FXW];
    __shared__ float fyd[FYH][TX];
    __shared__ float fya[FYH][TX];
    int q = 0;
    while (q + 1 < bands.n && blk >= bands.first[q + 1]) {
        ++q;
    }
    const int k = blk - bands.first[q];
    const int i0 = bands.c_lo[q] + TX * (k % bands.tiles_x[q]);
    const int j0 = bands.r_lo[q] + TY * (k / bands.tiles_x[q]);
    const int nx = p.nx;
    const int ny = p.ny;
    const int sx = nx + 2;
    const int sy = ny + 2;
    const int ps = m.ps;
    const int lane = (int)threadIdx.x % TX;
    const int ty = (int)threadIdx.x / TX;
    const float inv_dx = p.inv_dx;
    const float inv_dy = p.inv_dy;
    const int gi = i0 + lane;
    // Points outside the domain are never used by the general rows.
    for (int lj = ty; lj < SH; lj += TY) {
        const int gj = j0 - 5 + lj;
        for (int li = lane; li < SW; li += TX) {
            const int gs = i0 - 5 + li;
            s[lj][li] = gs >= 0 && gs < sx && gj >= 0 && gj < sy ? m.in[gs + ps * gj] : 0.0f;
        }
    }
    __syncthreads();
    {
        const int lj = ty;
        const int gj = j0 + lj;
        for (int lf = lane; lf < FXW; lf += TX) {
            const int f = i0 - 4 + lf;
            if (f >= 0 && f <= nx && gj <= ny) {
                const fa_sline_t line = {&s[lj + 5][0], i0 - 5, 1};
                fxd[lj][lf] = 0.5f * fa_t_grad(line, sx, f, inv_dx);
                fxa[lj][lf] = m.vx[f + m.pf * gj] * fa_t_interp(line, sx, f);
            }
        }
    }
    for (int lg = ty; lg < FYH; lg += TY) {
        const int g = j0 - 4 + lg;
        if (g >= 0 && g <= ny && gi <= nx) {
            const fa_sline_t line = {&s[0][lane + 5], j0 - 5, SW};
            fyd[lg][lane] = 0.5f * fa_t_grad(line, sy, g, inv_dy);
            fya[lg][lane] = m.vy[gi + ps * g] * fa_t_interp(line, sy, g);
        }
    }
    __syncthreads();
    const int gj = j0 + ty;
    if (gi >= bands.c_hi[q] || gj >= bands.r_hi[q]) {
        return;
    }
    const fa_sline_t xd = {&fxd[ty][0], i0 - 4, 1};
    const fa_sline_t xa = {&fxa[ty][0], i0 - 4, 1};
    const fa_sline_t yd = {&fyd[0][lane], j0 - 4, TX};
    const fa_sline_t ya = {&fya[0][lane], j0 - 4, TX};
    const float diffx = fa_t_div(xd, nx + 1, gi - 1, inv_dx);
    const float diffy = fa_t_div(yd, ny + 1, gj - 1, inv_dy);
    const float advx = fa_t_div(xa, nx + 1, gi - 1, inv_dx);
    const float advy = fa_t_div(ya, ny + 1, gj - 1, inv_dy);
    const float rhs = (diffx + diffy) - (advx + advy);
    const int idx = gi + ps * gj;
    float o, na;
    fa_lean_update<STAGE>(rhs, m.base[idx], STAGE > 1 ? m.acc[idx] : 0.0f, p, o, na);
    m.out[idx] = o;
    if (STAGE < 4) {
        m.acc[idx] = na;
    }
}

// All boundary tiles of a stage in one grid of 192-thread blocks: 64 x 3
// tiles for the top and bottom bands, 4 x 48 tiles for the side bands.  The
// grid allows the interior kernel that follows it in the stream to launch
// as soon as all its blocks have started (programmatic dependent launch), so
// the short boundary blocks are scheduled first and the two grids overlap.
template <int S>
__global__ void __launch_bounds__(192)
fa_lean_frame(fa_lptr_t m, fa_step_params_t p, fa_lean_bands_t bands, fa_lean_bands_t sides)
{
    asm volatile("griddepcontrol.launch_dependents;");
    const int nb = bands.first[bands.n];
    if ((int)blockIdx.x < nb) {
        fa_lean_tile<S, 64, 3>(m, p, bands, (int)blockIdx.x);
    } else {
        fa_lean_tile<S, 4, 48>(m, p, sides, (int)blockIdx.x - nb);
    }
}

template <int NW, int S>
static void fa_launch_lean_stage(const fa_config_t *cfg, const fa_lptr_t &m, int H,
    cudaStream_t s)
{
    constexpr int W = 64 * NW - 6;
    const int nx = (int)cfg->nx;
    const int ny = (int)cfg->ny;
    const fa_step_params_t p = fa_step_params(cfg);
    const fa_lean_bands_t bands = fa_lean_bands(nx, ny, false, 64, 3);
    const fa_lean_bands_t sides = fa_lean_bands(nx, ny, true, 4, 48);
    fa_lean_geo_t g;
    const bool interior = fa_lean_geo(nx, ny, &g);
    fa_lean_frame<S><<<bands.first[bands.n] + sides.first[sides.n], 192, 0, s>>>(m, p, bands,
        sides);
    if (interior) {
        // The interior grid waits for the boundary grid only at its end
        // (griddepcontrol.wait), so that the next stage, ordered after it,
        // sees both.
        cudaLaunchConfig_t lc = {};
        lc.gridDim = dim3((g.col_hi - g.col_lo + W - 1) / W, (g.row_hi - g.row_lo + H - 1) / H,
            1);
        lc.blockDim = dim3(32 * NW, 1, 1);
        lc.dynamicSmemBytes = 0;
        lc.stream = s;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        lc.attrs = attr;
        lc.numAttrs = 1;
        fa_cuda_check(cudaLaunchKernelEx(&lc, fa_lean_stage<NW, S>, m, p, g, H),
            "lean launch");
    }
}

template <int NW>
static void fa_launch_lean(const fa_config_t *cfg, fa_device_t *d, int parity, int H,
    cudaStream_t s)
{
    fa_lptr_t m;
    m.vx = d->m.vx;
    m.vy = d->m.vy;
    m.ps = d->m.ps;
    m.pf = d->m.pf;
    m.acc = d->m.acc;
    float *base = d->m.base[parity];
    m.base = base;
    m.in = base;
    m.out = d->m.stage[0];
    fa_launch_lean_stage<NW, 1>(cfg, m, H, s);
    m.in = d->m.stage[0];
    m.out = d->m.stage[1];
    fa_launch_lean_stage<NW, 2>(cfg, m, H, s);
    m.in = d->m.stage[1];
    m.out = d->m.stage[0];
    fa_launch_lean_stage<NW, 3>(cfg, m, H, s);
    m.in = d->m.stage[0];
    m.out = d->m.base[parity ^ 1];
    fa_launch_lean_stage<NW, 4>(cfg, m, H, s);
}

// Launch one timestep.  Only the "pair" variant alternates base buffers by
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
        case 8048: fa_launch_regy<48, 8>(cfg, d, sp, s); break;
        case 4024: fa_launch_regy<24, 4>(cfg, d, sp, s); break;
        case 16048: fa_launch_regy<48, 16>(cfg, d, sp, s); break;
        case 8040: fa_launch_regy<40, 8>(cfg, d, sp, s); break;
        default: fa_die("unsupported regy --block/--tile-y combination");
        }
        break;
    }
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
    case FA_KERNEL_MARCH:
        switch (opt->block_x) {
        case 64: fa_launch_march<2>(cfg, d, parity, (int)opt->tile_y, s); break;
        case 96: fa_launch_march<3>(cfg, d, parity, (int)opt->tile_y, s); break;
        case 128: fa_launch_march<4>(cfg, d, parity, (int)opt->tile_y, s); break;
        default: fa_die("unsupported march --block (64, 96 or 128)");
        }
        break;
    case FA_KERNEL_LEAN:
        switch (opt->block_x) {
        case 64: fa_launch_lean<2>(cfg, d, parity, (int)opt->tile_y, s); break;
        case 96: fa_launch_lean<3>(cfg, d, parity, (int)opt->tile_y, s); break;
        case 128: fa_launch_lean<4>(cfg, d, parity, (int)opt->tile_y, s); break;
        default: fa_die("unsupported lean --block (64, 96 or 128)");
        }
        break;
    default:
        fa_die("unsupported kernel variant");
    }
}

static void fa_upload_state(const fa_config_t *cfg, const fa_fields_t *fields,
    fa_device_t *d)
{
    const size_t sb = fields->scalar_len * sizeof(float);
    if (d->m.alloc[0]) {
        // Both base buffers and the stage buffers start as base: fixed
        // boundary points are never written.
        const size_t row = ((size_t)cfg->nx + 2) * sizeof(float);
        float *dst[4] = {d->m.base[0], d->m.base[1], d->m.stage[0], d->m.stage[1]};
        for (int k = 0; k < 4; ++k) {
            if (dst[k]) {
                fa_cuda_check(cudaMemcpy2D(dst[k], (size_t)d->m.ps * sizeof(float),
                    fields->base, row, row, (size_t)cfg->ny + 2, cudaMemcpyHostToDevice),
                    "upload padded base");
            }
        }
    }
    fa_cuda_check(cudaMemcpy(d->base, fields->base, sb, cudaMemcpyHostToDevice), "upload base");
    if (d->base_alt) {
        fa_cuda_check(cudaMemcpy(d->base_alt, fields->base, sb, cudaMemcpyHostToDevice),
            "upload base_alt");
    }
    fa_cuda_check(cudaMemcpy(d->stage[0], fields->stage[0], sb, cudaMemcpyHostToDevice), "upload stage0");
    fa_cuda_check(cudaMemcpy(d->stage[1], fields->stage[1], sb, cudaMemcpyHostToDevice), "upload stage1");
    fa_cuda_check(cudaMemcpy(d->accum, fields->accum, sb, cudaMemcpyHostToDevice), "upload accum");
}

// Resolve --kernel auto and fill in the tuned launch geometry of each
// variant where --block/--tile-x/--tile-y were not given.  "auto" selects
// "lean" while the six full-size fields of the per-stage scheme (base, two
// stage buffers, accumulator, both velocities) fit in the L2 cache, and
// "march" (one fused kernel per timestep, about a fifth of the global traffic
// of "pair") once they do not.
static void fa_resolve_kernel(fa_options_t *opt, size_t scalar_bytes)
{
    if (opt->kernel == FA_KERNEL_AUTO) {
        int l2 = 0;
        fa_cuda_check(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0), "L2 size");
        opt->kernel = 6 * scalar_bytes <= (size_t)l2 ? FA_KERNEL_LEAN : FA_KERNEL_MARCH;
    }
    uint32_t bx = 32, by = 8, tx = 32, ty = 16;
    switch (opt->kernel) {
    case FA_KERNEL_SPLIT: by = 4; break;
    case FA_KERNEL_REGY:
    case FA_KERNEL_REGYF: ty = 24; break;
    case FA_KERNEL_PAIR: by = 6; tx = 58; ty = 12; break;
    case FA_KERNEL_MARCH: bx = 128; by = 1; tx = 230; ty = 64; break;
    case FA_KERNEL_LEAN:
        // Tuned at 2048^2 (96 threads, 32 rows) and 1024^2 (64, 24).
        bx = (uint64_t)opt->cfg.nx * opt->cfg.ny >= 2000000 ? 96 : 64;
        by = 1;
        tx = 2 * bx - 6;
        ty = bx == 96 ? 32 : 24;
        break;
    default: break;
    }
    if (!opt->block_supplied) {
        opt->block_x = bx;
        opt->block_y = by;
    }
    if (opt->tile_x == 0) {
        opt->tile_x = tx;
    }
    if (opt->tile_y == 0) {
        opt->tile_y = ty;
    }
    if (opt->kernel == FA_KERNEL_MARCH) {
        if (opt->block_x != 64 && opt->block_x != 96 && opt->block_x != 128) {
            fa_die("march supports --block 64, 96 or 128");
        }
        if (opt->tile_y % FA_M_R != 0 || opt->tile_y < 24) {
            fa_die("march requires --tile-y a multiple of 8, at least 24");
        }
        opt->block_y = 1;
        opt->tile_x = 2 * opt->block_x - 26;
        if (opt->cfg.ny < opt->tile_y + 12) {
            // A segment would hold the dependency cones of both y boundaries.
            fprintf(stderr, "[march] ny < tile-y + 12: using regyf\n");
            opt->kernel = FA_KERNEL_REGYF;
            opt->block_x = 32;
            opt->block_y = 8;
            opt->tile_x = 32;
            opt->tile_y = 24;
        }
    }
    if (opt->kernel == FA_KERNEL_LEAN) {
        if (opt->block_x != 64 && opt->block_x != 96 && opt->block_x != 128) {
            fa_die("lean supports --block 64, 96 or 128");
        }
        if (opt->tile_y % FA_M_R != 0 || opt->tile_y < 8) {
            fa_die("lean requires --tile-y a multiple of 8");
        }
        opt->block_y = 1;
        opt->tile_x = 2 * opt->block_x - 6;
    }
    if ((uint64_t)opt->block_x * opt->block_y > 1024) {
        fa_die("block exceeds 1024 threads");
    }
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
    if (opt->kernel == FA_KERNEL_PAIR) {
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
    if (opt->kernel == FA_KERNEL_MARCH || opt->kernel == FA_KERNEL_LEAN) {
        const size_t nx = opt->cfg.nx;
        const size_t ny = opt->cfg.ny;
        const size_t pad = FA_M_PAD;
        d.m.ps = (int)((nx + 2 + 2 * pad + 31) / 32 * 32);
        d.m.pf = (int)((nx + 1 + 2 * pad + 31) / 32 * 32);
        const size_t rows = ny + 2 + 2 * pad;
        const size_t sbytes = (size_t)d.m.ps * rows * sizeof(float);
        const size_t fbytes = (size_t)d.m.pf * rows * sizeof(float);
        const int nalloc = opt->kernel == FA_KERNEL_LEAN ? 7 : 4;
        for (int k = 0; k < nalloc; ++k) {
            const size_t bytes = k == 2 ? fbytes : sbytes;
            fa_cuda_check(cudaMalloc((void **)&d.m.alloc[k], bytes), "malloc padded");
            fa_cuda_check(cudaMemset(d.m.alloc[k], 0, bytes), "memset padded");
        }
        const size_t origin = pad + (size_t)d.m.ps * pad;
        d.m.base[0] = d.m.alloc[0] + origin;
        d.m.base[1] = d.m.alloc[1] + origin;
        d.m.vx = d.m.alloc[2] + pad + (size_t)d.m.pf * pad;
        d.m.vy = d.m.alloc[3] + origin;
        if (nalloc == 7) {
            d.m.stage[0] = d.m.alloc[4] + origin;
            d.m.stage[1] = d.m.alloc[5] + origin;
            d.m.acc = d.m.alloc[6] + origin;
        }
        fa_cuda_check(cudaMemcpy2D(d.m.vx, (size_t)d.m.pf * sizeof(float), fields->vx,
            (nx + 1) * sizeof(float), (nx + 1) * sizeof(float), ny + 2,
            cudaMemcpyHostToDevice), "upload padded vx");
        fa_cuda_check(cudaMemcpy2D(d.m.vy, (size_t)d.m.ps * sizeof(float), fields->vy,
            (nx + 2) * sizeof(float), (nx + 2) * sizeof(float), ny + 1,
            cudaMemcpyHostToDevice), "upload padded vy");
    }
    fa_upload_state(&opt->cfg, fields, &d);
    d.march_parts = (int)opt->frame_parts;

    cudaStream_t stream;
    fa_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    fa_cuda_check(cudaStreamCreateWithFlags(&d.side, cudaStreamNonBlocking), "side stream");
    for (int q = 0; q < 3; ++q) {
        fa_cuda_check(cudaStreamCreateWithFlags(&d.mside[q], cudaStreamNonBlocking),
            "side stream");
        fa_cuda_check(cudaEventCreateWithFlags(&d.mjoin[q], cudaEventDisableTiming),
            "join event");
    }
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
    fa_upload_state(&opt->cfg, fields, &d);
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
    if (opt->kernel == FA_KERNEL_MARCH || opt->kernel == FA_KERNEL_LEAN) {
        const size_t row = ((size_t)opt->cfg.nx + 2) * sizeof(float);
        fa_cuda_check(cudaMemcpy2DAsync(fields->base, row, d.m.base[opt->cfg.steps & 1u],
            (size_t)d.m.ps * sizeof(float), row, (size_t)opt->cfg.ny + 2,
            cudaMemcpyDeviceToHost, stream), "download base");
    } else {
        fa_cuda_check(cudaMemcpyAsync(fields->base, final_base, sb, cudaMemcpyDeviceToHost,
            stream), "download base");
    }
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
    for (int q = 0; q < 3; ++q) {
        cudaStreamDestroy(d.mside[q]);
        cudaEventDestroy(d.mjoin[q]);
    }
    cudaEventDestroy(d.fork);
    cudaEventDestroy(d.join);
    cudaFree(d.base);
    cudaFree(d.base_alt);
    for (int k = 0; k < 7; ++k) {
        cudaFree(d.m.alloc[k]);
    }
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
        "          [--kernel auto|split|fused|tiled|prefetch|regy|regyf|pair|march|lean] [--block x[,y]]\n"
        "          [--tile-x n] [--tile-y n] [--frame-parts n] [--graph on|off]\n",
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
    opt.kernel = FA_KERNEL_AUTO;
    opt.graph = true;
    opt.tile_x = 0;
    opt.tile_y = 0;
    opt.frame_parts = 2;

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
        } else if (strcmp(argv[i], "--frame-parts") == 0 && i + 1 < argc) {
            opt.frame_parts = fa_parse_u32_strict(argv[++i], "--frame-parts");
            if (opt.frame_parts < 1 || opt.frame_parts > 8) {
                fa_die("--frame-parts must be 1..8");
            }
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
        fa_resolve_kernel(&opt, fields.scalar_len * sizeof(float));
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
