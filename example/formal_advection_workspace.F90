module formal_advection_workspace
    use iso_fortran_env, only: real32, real64, int64
    use tensors_2D_m, only: scalar_2D_t, scalar_2D_initializer_i
    implicit none
    private
    public :: fa_config_t, fa_options_t, fa_fields_t, fa_parse_args
    public :: fa_init_fields, fa_advance_rk4, fa_advance_rk4_fused, fa_advance_rk4_fused2
    public :: fa_advance_rk4_regy
    public :: fa_write_csv, fa_reconstruct_scalar
    public :: fa_init_from_arrays, fa_write_final_raw
    public :: fa_validate_cfg, fa_validate_working_set, fa_read_raw_input
    public :: fa_now_seconds

    type :: fa_config_t
        integer :: nx = 51
        integer :: ny = 51
        integer :: steps = 500
        real(real32) :: dt = 1.0e-4_real32
    end type

    type :: fa_options_t
        type(fa_config_t) :: cfg
        logical :: dump_csv = .false.
        logical :: gpu_profile = .false.
        logical :: dt_supplied = .false.
        character(len=:), allocatable :: input_path
        character(len=:), allocatable :: output_path
    end type

    type :: fa_fields_t
        real(real32), allocatable :: base(:,:)
        real(real32), allocatable :: q(:,:)
        real(real32), allocatable :: acc(:,:)
        real(real32), allocatable :: vx(:,:)
        real(real32), allocatable :: vy(:,:)
        real(real32), allocatable :: diff_x(:,:)
        real(real32), allocatable :: adv_x(:,:)
        real(real32), allocatable :: diff_y(:,:)
        real(real32), allocatable :: adv_y(:,:)
    end type

    real(real32), allocatable, save :: fa_scalar_reconstruction_values(:,:)

    ! Cells along y that one iteration of the interior loop of
    ! fa_advance_rk4_regy computes.
    integer, parameter :: fa_regy_rows = 3

contains

    real(real64) function fa_now_seconds() result(now)
        integer(int64) :: count, rate
        call system_clock(count=count, count_rate=rate)
        if (rate <= 0) error stop "system_clock is unavailable"
        now = real(count, real64) / real(rate, real64)
    end function fa_now_seconds

    pure real(real32) function fa_center_coord(i, n, minv, maxv) result(x)
        integer, intent(in) :: i, n
        real(real32), intent(in) :: minv, maxv
        real(real32) :: dx
        dx = (maxv - minv) / real(n, real32)
        if (i == 1) then
            x = minv
        else if (i == n + 2) then
            x = maxv
        else
            x = minv + (real(i, real32) - 1.5_real32) * dx
        end if
    end function fa_center_coord

    pure real(real32) function fa_face_coord(i, n, minv, maxv) result(x)
        integer, intent(in) :: i, n
        real(real32), intent(in) :: minv, maxv
        real(real32) :: dx
        dx = (maxv - minv) / real(n, real32)
        x = minv + real(i - 1, real32) * dx
    end function fa_face_coord

    pure real(real32) function fa_velocity_x(x, y) result(v)
        real(real32), intent(in) :: x, y
        v = 10.0_real32 * sin(x) * cos(y)
    end function fa_velocity_x

    pure real(real32) function fa_velocity_y(x, y) result(v)
        real(real32), intent(in) :: x, y
        v = -10.0_real32 * cos(x) * sin(y)
    end function fa_velocity_y

    subroutine fa_die(message)
        character(len=*), intent(in) :: message
        write(*, '(a)') trim(message)
        error stop trim(message)
    end subroutine fa_die

    integer(int64) function fa_checked_mul(a, b, what) result(product)
        integer(int64), intent(in) :: a, b
        character(len=*), intent(in) :: what
        if (a < 0_int64 .or. b < 0_int64) call fa_die(trim(what) // " size must be non-negative")
        if (a > 0_int64) then
            if (b > huge(0_int64) / a) call fa_die(trim(what) // " size overflow")
        end if
        product = a * b
    end function fa_checked_mul

    integer(int64) function fa_checked_add(a, b, what) result(total)
        integer(int64), intent(in) :: a, b
        character(len=*), intent(in) :: what
        if (a < 0_int64 .or. b < 0_int64) call fa_die(trim(what) // " size must be non-negative")
        if (a > huge(0_int64) - b) call fa_die(trim(what) // " size overflow")
        total = a + b
    end function fa_checked_add

    subroutine fa_validate_working_set(nx, ny)
        integer, intent(in) :: nx, ny
        integer(int64) :: nx64, ny64, scalar_len, xface_len, yface_len
        integer(int64) :: scalar_bytes, xface_bytes, yface_bytes, total_bytes
        if (nx < 1 .or. ny < 1) call fa_die("cells must be positive")
        if (nx > huge(nx) - 2 .or. ny > huge(ny) - 2) then
            call fa_die("cells including halo size overflow")
        end if
        nx64 = int(nx, int64)
        ny64 = int(ny, int64)
        scalar_len = fa_checked_mul(nx64 + 2_int64, ny64 + 2_int64, &
            "scalar field")
        xface_len = fa_checked_mul(nx64 + 1_int64, ny64 + 2_int64, &
            "x-face field")
        yface_len = fa_checked_mul(nx64 + 2_int64, ny64 + 1_int64, &
            "y-face field")
        scalar_bytes = fa_checked_mul(scalar_len, 12_int64, "scalar buffers")
        xface_bytes = fa_checked_mul(xface_len, 12_int64, "x-face buffers")
        yface_bytes = fa_checked_mul(yface_len, 12_int64, "y-face buffers")
        total_bytes = fa_checked_add(scalar_bytes, xface_bytes, "working set")
        total_bytes = fa_checked_add(total_bytes, yface_bytes, "working set")
        if (total_bytes > 512_int64 * 1024_int64 * 1024_int64) then
            call fa_die("estimated working set exceeds 512 MiB")
        end if
    end subroutine fa_validate_working_set

    subroutine fa_validate_cfg(cfg, dt_supplied)
        use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
        type(fa_config_t), intent(inout) :: cfg
        logical, intent(in) :: dt_supplied
        real(real32) :: scale
        if (cfg%nx < 9 .or. cfg%ny < 9) then
            call fa_die("cells must be at least 9 in each dimension")
        end if
        if (cfg%steps < 1) then
            call fa_die("steps must be positive")
        end if
        if (.not. ieee_is_finite(cfg%dt) .or. .not. (cfg%dt > 0.0_real32)) then
            call fa_die("dt must be finite and positive")
        end if
        if (.not. dt_supplied) then
            scale = 51.0_real32 / real(max(cfg%nx, cfg%ny), real32)
            cfg%dt = 1.0e-4_real32 * scale * scale
        end if
    end subroutine fa_validate_cfg

    pure real(real32) function fa_scalar_init(x, y, x0, y0, sigma) result(s)
        real(real32), intent(in) :: x, y, x0, y0, sigma
        real(real32) :: dx, dy, r
        dx = x - x0
        dy = y - y0
        r = sqrt(dx * dx + dy * dy)
        s = exp(-((r * r) / (2.0_real32 * sigma * sigma)))
    end function fa_scalar_init

    subroutine fa_fill_analytic(fields, nx, ny, pi, x0, y0, sigma)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny
        real(real32), intent(in) :: pi, x0, y0, sigma
        integer :: i, j
        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            fields%base(i, j) = fa_scalar_init( &
                fa_center_coord(i, nx, -pi, pi), &
                fa_center_coord(j, ny, -pi, pi), x0, y0, sigma)
            fields%q(i, j) = fields%base(i, j)
            fields%acc(i, j) = 0.0_real32
        end do
        do concurrent (j = 1:ny + 2, i = 1:nx + 1)
            fields%vx(i, j) = fa_velocity_x( &
                fa_face_coord(i, nx, -pi, pi), &
                fa_center_coord(j, ny, -pi, pi))
        end do
        do concurrent (j = 1:ny + 1, i = 1:nx + 2)
            fields%vy(i, j) = fa_velocity_y( &
                fa_center_coord(i, nx, -pi, pi), &
                fa_face_coord(j, ny, -pi, pi))
        end do
    end subroutine fa_fill_analytic

    subroutine fa_read_line(unit, line, ios)
        integer, intent(in) :: unit
        character(len=:), allocatable, intent(out) :: line
        integer, intent(out) :: ios
        character(len=1) :: ch
        character(len=1024) :: buffer
        integer :: len_line

        buffer = ''
        len_line = 0
        ios = 0
        do
            read(unit, iostat=ios) ch
            if (ios /= 0) exit
            if (ch == new_line('a')) exit
            len_line = len_line + 1
            if (len_line > len(buffer)) then
                ios = 1
                exit
            end if
            buffer(len_line:len_line) = ch
        end do
        if (len_line > 0) then
            line = buffer(:len_line)
        else
            line = ''
        end if
    end subroutine fa_read_line

    subroutine fa_read_raw_input(path, fields, nx, ny, steps, dt)
        character(len=*), intent(in) :: path
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(out) :: nx, ny, steps
        real(real32), intent(out) :: dt
        integer :: unit, ios
        character(len=:), allocatable :: line
        character(len=16) :: key, mode
        character(len=512) :: msg
        type(fa_config_t) :: cfg

        open(newunit=unit, file=path, status="old", action="read", &
            access="stream", form="unformatted", iostat=ios, iomsg=msg)
        if (ios /= 0) call fa_die(trim(msg))

        call fa_read_line(unit, line, ios)
        if (ios /= 0 .or. trim(line) /= "FADV2D1") then
            call fa_die("invalid input header")
        end if
        call fa_read_line(unit, line, ios)
        if (ios /= 0) call fa_die("missing input dimensions")
        read(line, *, iostat=ios) cfg%nx, cfg%ny, cfg%steps, cfg%dt
        if (ios /= 0) call fa_die("cannot parse input dimensions")
        call fa_read_line(unit, line, ios)
        if (ios /= 0) call fa_die("missing input mode")
        read(line, *, iostat=ios) key, mode
        if (ios /= 0 .or. trim(key) /= "mode") then
            call fa_die("cannot parse input mode")
        end if
        if (trim(mode) /= "raw") then
            call fa_die("unknown input mode")
        end if

        call fa_validate_cfg(cfg, .true.)
        nx = cfg%nx
        ny = cfg%ny
        steps = cfg%steps
        dt = cfg%dt
        call fa_allocate_fields(fields, nx, ny)
        read(unit, iostat=ios) fields%base
        if (ios /= 0) call fa_die("cannot read raw base field")
        read(unit, iostat=ios) fields%vx
        if (ios /= 0) call fa_die("cannot read raw vx field")
        read(unit, iostat=ios) fields%vy
        if (ios /= 0) call fa_die("cannot read raw vy field")
        fields%q = fields%base
        fields%acc = 0.0_real32
        close(unit)
    end subroutine fa_read_raw_input

    pure real(real32) function fa_grad_face_x(q, nx, ny, i, j, inv_dx) result(g)
        real(real32), intent(in) :: q(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dx
        integer :: n
        n = nx + 2
        if (i == 1) then
            g = inv_dx * ((-352.0_real32 / 105.0_real32) * q(1, j) &
                + (35.0_real32 / 8.0_real32) * q(2, j) &
                + (-35.0_real32 / 24.0_real32) * q(3, j) &
                + (21.0_real32 / 40.0_real32) * q(4, j) &
                + (-5.0_real32 / 56.0_real32) * q(5, j))
        else if (i == 2) then
            g = inv_dx * ((16.0_real32 / 105.0_real32) * q(1, j) &
                + (-31.0_real32 / 24.0_real32) * q(2, j) &
                + (29.0_real32 / 24.0_real32) * q(3, j) &
                + (-3.0_real32 / 40.0_real32) * q(4, j) &
                + (1.0_real32 / 168.0_real32) * q(5, j))
        else if (i + 2 < n) then
            g = inv_dx * ((1.0_real32 / 24.0_real32) * q(i - 1, j) &
                + (-9.0_real32 / 8.0_real32) * q(i, j) &
                + (9.0_real32 / 8.0_real32) * q(i + 1, j) &
                + (-1.0_real32 / 24.0_real32) * q(i + 2, j))
        else if (i == n - 2) then
            g = inv_dx * ((-1.0_real32 / 168.0_real32) * q(n - 4, j) &
                + (3.0_real32 / 40.0_real32) * q(n - 3, j) &
                + (-29.0_real32 / 24.0_real32) * q(n - 2, j) &
                + (31.0_real32 / 24.0_real32) * q(n - 1, j) &
                + (-16.0_real32 / 105.0_real32) * q(n, j))
        else
            g = inv_dx * ((5.0_real32 / 56.0_real32) * q(n - 4, j) &
                + (-21.0_real32 / 40.0_real32) * q(n - 3, j) &
                + (35.0_real32 / 24.0_real32) * q(n - 2, j) &
                + (-35.0_real32 / 8.0_real32) * q(n - 1, j) &
                + (352.0_real32 / 105.0_real32) * q(n, j))
        end if
    end function fa_grad_face_x

    pure real(real32) function fa_grad_face_y(q, nx, ny, i, j, inv_dy) result(g)
        real(real32), intent(in) :: q(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dy
        integer :: n
        n = ny + 2
        if (j == 1) then
            g = inv_dy * ((-352.0_real32 / 105.0_real32) * q(i, 1) &
                + (35.0_real32 / 8.0_real32) * q(i, 2) &
                + (-35.0_real32 / 24.0_real32) * q(i, 3) &
                + (21.0_real32 / 40.0_real32) * q(i, 4) &
                + (-5.0_real32 / 56.0_real32) * q(i, 5))
        else if (j == 2) then
            g = inv_dy * ((16.0_real32 / 105.0_real32) * q(i, 1) &
                + (-31.0_real32 / 24.0_real32) * q(i, 2) &
                + (29.0_real32 / 24.0_real32) * q(i, 3) &
                + (-3.0_real32 / 40.0_real32) * q(i, 4) &
                + (1.0_real32 / 168.0_real32) * q(i, 5))
        else if (j + 2 < n) then
            g = inv_dy * ((1.0_real32 / 24.0_real32) * q(i, j - 1) &
                + (-9.0_real32 / 8.0_real32) * q(i, j) &
                + (9.0_real32 / 8.0_real32) * q(i, j + 1) &
                + (-1.0_real32 / 24.0_real32) * q(i, j + 2))
        else if (j == n - 2) then
            g = inv_dy * ((-1.0_real32 / 168.0_real32) * q(i, n - 4) &
                + (3.0_real32 / 40.0_real32) * q(i, n - 3) &
                + (-29.0_real32 / 24.0_real32) * q(i, n - 2) &
                + (31.0_real32 / 24.0_real32) * q(i, n - 1) &
                + (-16.0_real32 / 105.0_real32) * q(i, n))
        else
            g = inv_dy * ((5.0_real32 / 56.0_real32) * q(i, n - 4) &
                + (-21.0_real32 / 40.0_real32) * q(i, n - 3) &
                + (35.0_real32 / 24.0_real32) * q(i, n - 2) &
                + (-35.0_real32 / 8.0_real32) * q(i, n - 1) &
                + (352.0_real32 / 105.0_real32) * q(i, n))
        end if
    end function fa_grad_face_y

    pure real(real32) function fa_interp_face_x(q, nx, ny, i, j) result(v)
        real(real32), intent(in) :: q(:,:)
        integer, intent(in) :: nx, ny, i, j
        integer :: n
        n = nx + 2
        if (i == 1) then
            v = q(1, j)
        else if (i == 2) then
            v = (-16.0_real32 / 112.0_real32) * q(1, j) &
                + (70.0_real32 / 112.0_real32) * q(2, j) &
                + (70.0_real32 / 112.0_real32) * q(3, j) &
                + (-14.0_real32 / 112.0_real32) * q(4, j) &
                + (2.0_real32 / 112.0_real32) * q(5, j)
        else if (i + 2 < n) then
            v = (-7.0_real32 / 112.0_real32) * q(i - 1, j) &
                + (63.0_real32 / 112.0_real32) * q(i, j) &
                + (63.0_real32 / 112.0_real32) * q(i + 1, j) &
                + (-7.0_real32 / 112.0_real32) * q(i + 2, j)
        else if (i == n - 2) then
            v = (2.0_real32 / 112.0_real32) * q(n - 4, j) &
                + (-14.0_real32 / 112.0_real32) * q(n - 3, j) &
                + (70.0_real32 / 112.0_real32) * q(n - 2, j) &
                + (70.0_real32 / 112.0_real32) * q(n - 1, j) &
                + (-16.0_real32 / 112.0_real32) * q(n, j)
        else
            v = q(n, j)
        end if
    end function fa_interp_face_x

    pure real(real32) function fa_interp_face_y(q, nx, ny, i, j) result(v)
        real(real32), intent(in) :: q(:,:)
        integer, intent(in) :: nx, ny, i, j
        integer :: n
        n = ny + 2
        if (j == 1) then
            v = q(i, 1)
        else if (j == 2) then
            v = (-16.0_real32 / 112.0_real32) * q(i, 1) &
                + (70.0_real32 / 112.0_real32) * q(i, 2) &
                + (70.0_real32 / 112.0_real32) * q(i, 3) &
                + (-14.0_real32 / 112.0_real32) * q(i, 4) &
                + (2.0_real32 / 112.0_real32) * q(i, 5)
        else if (j + 2 < n) then
            v = (-7.0_real32 / 112.0_real32) * q(i, j - 1) &
                + (63.0_real32 / 112.0_real32) * q(i, j) &
                + (63.0_real32 / 112.0_real32) * q(i, j + 1) &
                + (-7.0_real32 / 112.0_real32) * q(i, j + 2)
        else if (j == n - 2) then
            v = (2.0_real32 / 112.0_real32) * q(i, n - 4) &
                + (-14.0_real32 / 112.0_real32) * q(i, n - 3) &
                + (70.0_real32 / 112.0_real32) * q(i, n - 2) &
                + (70.0_real32 / 112.0_real32) * q(i, n - 1) &
                + (-16.0_real32 / 112.0_real32) * q(i, n)
        else
            v = q(i, n)
        end if
    end function fa_interp_face_y

    pure real(real32) function fa_div_face_x(flux, nx, ny, i, j, inv_dx) result(d)
        real(real32), intent(in) :: flux(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dx
        integer :: n
        n = nx + 1
        if (i == 1) then
            d = inv_dx * ((-11.0_real32 / 12.0_real32) * flux(1, j) &
                + (17.0_real32 / 24.0_real32) * flux(2, j) &
                + (3.0_real32 / 8.0_real32) * flux(3, j) &
                + (-5.0_real32 / 24.0_real32) * flux(4, j) &
                + (1.0_real32 / 24.0_real32) * flux(5, j))
        else if (i + 1 < n) then
            d = inv_dx * ((1.0_real32 / 24.0_real32) * flux(i - 1, j) &
                + (-9.0_real32 / 8.0_real32) * flux(i, j) &
                + (9.0_real32 / 8.0_real32) * flux(i + 1, j) &
                + (-1.0_real32 / 24.0_real32) * flux(i + 2, j))
        else
            d = inv_dx * ((-1.0_real32 / 24.0_real32) * flux(n - 4, j) &
                + (5.0_real32 / 24.0_real32) * flux(n - 3, j) &
                + (-3.0_real32 / 8.0_real32) * flux(n - 2, j) &
                + (-17.0_real32 / 24.0_real32) * flux(n - 1, j) &
                + (11.0_real32 / 12.0_real32) * flux(n, j))
        end if
    end function fa_div_face_x

    pure real(real32) function fa_div_face_y(flux, nx, ny, i, j, inv_dy) result(d)
        real(real32), intent(in) :: flux(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dy
        integer :: n
        n = ny + 1
        if (j == 1) then
            d = inv_dy * ((-11.0_real32 / 12.0_real32) * flux(i, 1) &
                + (17.0_real32 / 24.0_real32) * flux(i, 2) &
                + (3.0_real32 / 8.0_real32) * flux(i, 3) &
                + (-5.0_real32 / 24.0_real32) * flux(i, 4) &
                + (1.0_real32 / 24.0_real32) * flux(i, 5))
        else if (j + 1 < n) then
            d = inv_dy * ((1.0_real32 / 24.0_real32) * flux(i, j - 1) &
                + (-9.0_real32 / 8.0_real32) * flux(i, j) &
                + (9.0_real32 / 8.0_real32) * flux(i, j + 1) &
                + (-1.0_real32 / 24.0_real32) * flux(i, j + 2))
        else
            d = inv_dy * ((-1.0_real32 / 24.0_real32) * flux(i, n - 4) &
                + (5.0_real32 / 24.0_real32) * flux(i, n - 3) &
                + (-3.0_real32 / 8.0_real32) * flux(i, n - 2) &
                + (-17.0_real32 / 24.0_real32) * flux(i, n - 1) &
                + (11.0_real32 / 12.0_real32) * flux(i, n))
        end if
    end function fa_div_face_y

    subroutine fa_allocate_fields(fields, nx, ny)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny
        call fa_validate_working_set(nx, ny)
        allocate(fields%base(1:nx+2, 1:ny+2))
        allocate(fields%q(1:nx+2, 1:ny+2))
        allocate(fields%acc(1:nx+2, 1:ny+2))
        allocate(fields%vx(1:nx+1, 1:ny+2))
        allocate(fields%vy(1:nx+2, 1:ny+1))
        allocate(fields%diff_x(1:nx+1, 1:ny+2))
        allocate(fields%adv_x(1:nx+1, 1:ny+2))
        allocate(fields%diff_y(1:nx+2, 1:ny+1))
        allocate(fields%adv_y(1:nx+2, 1:ny+1))
    end subroutine fa_allocate_fields

    subroutine fa_init_fields(fields, nx, ny, pi, x0, y0, sigma)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny
        real(real32), intent(in) :: pi, x0, y0, sigma
        call fa_allocate_fields(fields, nx, ny)
        call fa_fill_analytic(fields, nx, ny, pi, x0, y0, sigma)
    end subroutine fa_init_fields

    subroutine fa_init_from_arrays(fields, base, vx, vy)
        type(fa_fields_t), intent(inout) :: fields
        real(real32), intent(in) :: base(:,:), vx(:,:), vy(:,:)
        integer :: nx, ny

        nx = size(base, 1) - 2
        ny = size(base, 2) - 2
        call fa_allocate_fields(fields, nx, ny)
        fields%base = base
        fields%q = base
        fields%acc = 0.0_real32
        fields%vx = vx
        fields%vy = vy
        fields%diff_x = 0.0_real32
        fields%adv_x = 0.0_real32
        fields%diff_y = 0.0_real32
        fields%adv_y = 0.0_real32
    end subroutine fa_init_from_arrays

    subroutine fa_write_final_raw(path, fields, nx, ny, steps, dt)
        character(len=*), intent(in) :: path
        type(fa_fields_t), intent(in) :: fields
        integer, intent(in) :: nx, ny, steps
        real(real32), intent(in) :: dt
        integer :: unit
        character(len=256) :: header

        open(newunit=unit, file=path, access="stream", form="unformatted", &
            status="replace", action="write")
        write(unit) "FADV2D1" // new_line("a")
        write(header,'(i0,1x,i0,1x,i0,1x,es24.16)') nx, ny, steps, real(dt, real64)
        write(unit) trim(header) // new_line("a")
        write(unit) "mode raw-final" // new_line("a")
        write(unit) real(fields%base, kind=real32)
        close(unit)
    end subroutine fa_write_final_raw

    pure function fa_scalar_reconstruction_initializer(x, y) result(field)
        real, intent(in) :: x(:), y(:)
        real field(size(x), size(y))

        if (.not. allocated(fa_scalar_reconstruction_values)) then
            error stop "scalar reconstruction values are unavailable"
        end if
        field = real(fa_scalar_reconstruction_values, kind=kind(field))
    end function fa_scalar_reconstruction_initializer

    function fa_reconstruct_scalar(values, order, cells, x_min, x_max) result(scalar)
        real(real32), intent(in) :: values(:,:)
        integer, intent(in) :: order
        integer, intent(in) :: cells(:)
        real(real32), intent(in) :: x_min(:), x_max(:)
        type(scalar_2D_t) :: scalar
        procedure(scalar_2D_initializer_i), pointer :: initializer

        initializer => fa_scalar_reconstruction_initializer
        fa_scalar_reconstruction_values = values
        scalar = scalar_2D_t(initializer, order=order, cells=cells, x_min=x_min, x_max=x_max)
        deallocate(fa_scalar_reconstruction_values)
    end function fa_reconstruct_scalar

    subroutine fa_advance_rk4(fields, nx, ny, steps, dt, pi, diffusion)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny, steps
        real(real32), intent(in) :: dt, pi, diffusion
        integer :: step, stage, i, j
        real(real32) :: inv_dx, inv_dy, half_dt, dt_over_6, rhs, total
        real(real32), allocatable :: vx_local(:,:), vy_local(:,:)

        inv_dx = real(nx, real32) / (2.0_real32 * pi)
        inv_dy = real(ny, real32) / (2.0_real32 * pi)
        half_dt = 0.5_real32 * dt
        dt_over_6 = dt / 6.0_real32
        allocate(vx_local(1:nx + 1, 1:ny + 2))
        allocate(vy_local(1:nx + 2, 1:ny + 1))
        do concurrent (j = 1:ny + 2, i = 1:nx + 1)
            vx_local(i, j) = fields%vx(i, j)
        end do
        do concurrent (j = 1:ny + 1, i = 1:nx + 2)
            vy_local(i, j) = fields%vy(i, j)
        end do
        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            fields%q(i, j) = fields%base(i, j)
            fields%acc(i, j) = 0.0_real32
        end do

        do step = 1, steps
            do stage = 1, 4
                do concurrent (j = 1:ny + 2, i = 1:nx + 2)
                    if (i <= nx + 1) then
                        fields%diff_x(i, j) = diffusion * fa_grad_face_x( &
                            fields%q, nx, ny, i, j, inv_dx)
                        fields%adv_x(i, j) = vx_local(i, j) * fa_interp_face_x( &
                            fields%q, nx, ny, i, j)
                    end if
                    if (j <= ny + 1) then
                        fields%diff_y(i, j) = diffusion * fa_grad_face_y( &
                            fields%q, nx, ny, i, j, inv_dy)
                        fields%adv_y(i, j) = vy_local(i, j) * fa_interp_face_y( &
                            fields%q, nx, ny, i, j)
                    end if
                end do

                do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs, total)
                    rhs = (fa_div_face_x(fields%diff_x, nx, ny, i - 1, j, &
                        inv_dx) + fa_div_face_y(fields%diff_y, nx, ny, i, j - 1, &
                        inv_dy)) - (fa_div_face_x(fields%adv_x, nx, ny, i - 1, j, &
                        inv_dx) + fa_div_face_y(fields%adv_y, nx, ny, i, j - 1, &
                        inv_dy))
                    select case (stage)
                    case (1)
                        fields%acc(i, j) = rhs
                        fields%q(i, j) = fields%base(i, j) + half_dt * rhs
                    case (2)
                        fields%acc(i, j) = fields%acc(i, j) + 2.0_real32 * rhs
                        fields%q(i, j) = fields%base(i, j) + half_dt * rhs
                    case (3)
                        fields%acc(i, j) = fields%acc(i, j) + 2.0_real32 * rhs
                        fields%q(i, j) = fields%base(i, j) + dt * rhs
                    case default
                        total = fields%acc(i, j) + rhs
                        fields%base(i, j) = fields%base(i, j) + dt_over_6 * total
                        fields%q(i, j) = fields%base(i, j)
                    end select
                end do
            end do
        end do
        deallocate(vx_local)
        deallocate(vy_local)
    end subroutine fa_advance_rk4

    ! Face flux of the scalar q at x face (k, j): diffusive (diffusion * grad)
    ! or advective (v * interp), exactly as fa_advance_rk4 stores them in
    ! diff_x and adv_x.
    pure real(real32) function fa_flux_x(q, v, nx, ny, k, j, inv_dx, diffusion, advective) result(f)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, k, j
        real(real32), intent(in) :: inv_dx, diffusion
        logical, intent(in) :: advective
        if (advective) then
            f = v(k, j) * fa_interp_face_x(q, nx, ny, k, j)
        else
            f = diffusion * fa_grad_face_x(q, nx, ny, k, j, inv_dx)
        end if
    end function fa_flux_x

    pure real(real32) function fa_flux_y(q, v, nx, ny, i, k, inv_dy, diffusion, advective) result(f)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, i, k
        real(real32), intent(in) :: inv_dy, diffusion
        logical, intent(in) :: advective
        if (advective) then
            f = v(i, k) * fa_interp_face_y(q, nx, ny, i, k)
        else
            f = diffusion * fa_grad_face_y(q, nx, ny, i, k, inv_dy)
        end if
    end function fa_flux_y

    ! fa_div_face_x applied to the x face fluxes of q, computed on the fly.
    pure real(real32) function fa_fused_div_x(q, v, nx, ny, i, j, inv_dx, diffusion, advective) result(d)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dx, diffusion
        logical, intent(in) :: advective
        integer :: n
        n = nx + 1
        if (i == 1) then
            d = inv_dx * ((-11.0_real32 / 12.0_real32) * fa_flux_x(q, v, nx, ny, 1, j, inv_dx, diffusion, advective) &
                + (17.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, 2, j, inv_dx, diffusion, advective) &
                + (3.0_real32 / 8.0_real32) * fa_flux_x(q, v, nx, ny, 3, j, inv_dx, diffusion, advective) &
                + (-5.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, 4, j, inv_dx, diffusion, advective) &
                + (1.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, 5, j, inv_dx, diffusion, advective))
        else if (i + 1 < n) then
            d = inv_dx * ((1.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, i - 1, j, inv_dx, diffusion, advective) &
                + (-9.0_real32 / 8.0_real32) * fa_flux_x(q, v, nx, ny, i, j, inv_dx, diffusion, advective) &
                + (9.0_real32 / 8.0_real32) * fa_flux_x(q, v, nx, ny, i + 1, j, inv_dx, diffusion, advective) &
                + (-1.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, i + 2, j, inv_dx, diffusion, advective))
        else
            d = inv_dx * ((-1.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, n - 4, j, inv_dx, diffusion, advective) &
                + (5.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, n - 3, j, inv_dx, diffusion, advective) &
                + (-3.0_real32 / 8.0_real32) * fa_flux_x(q, v, nx, ny, n - 2, j, inv_dx, diffusion, advective) &
                + (-17.0_real32 / 24.0_real32) * fa_flux_x(q, v, nx, ny, n - 1, j, inv_dx, diffusion, advective) &
                + (11.0_real32 / 12.0_real32) * fa_flux_x(q, v, nx, ny, n, j, inv_dx, diffusion, advective))
        end if
    end function fa_fused_div_x

    ! fa_div_face_y applied to the y face fluxes of q, computed on the fly.
    pure real(real32) function fa_fused_div_y(q, v, nx, ny, i, j, inv_dy, diffusion, advective) result(d)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dy, diffusion
        logical, intent(in) :: advective
        integer :: n
        n = ny + 1
        if (j == 1) then
            d = inv_dy * ((-11.0_real32 / 12.0_real32) * fa_flux_y(q, v, nx, ny, i, 1, inv_dy, diffusion, advective) &
                + (17.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, 2, inv_dy, diffusion, advective) &
                + (3.0_real32 / 8.0_real32) * fa_flux_y(q, v, nx, ny, i, 3, inv_dy, diffusion, advective) &
                + (-5.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, 4, inv_dy, diffusion, advective) &
                + (1.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, 5, inv_dy, diffusion, advective))
        else if (j + 1 < n) then
            d = inv_dy * ((1.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, j - 1, inv_dy, diffusion, advective) &
                + (-9.0_real32 / 8.0_real32) * fa_flux_y(q, v, nx, ny, i, j, inv_dy, diffusion, advective) &
                + (9.0_real32 / 8.0_real32) * fa_flux_y(q, v, nx, ny, i, j + 1, inv_dy, diffusion, advective) &
                + (-1.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, j + 2, inv_dy, diffusion, advective))
        else
            d = inv_dy * ((-1.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, n - 4, inv_dy, diffusion, advective) &
                + (5.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, n - 3, inv_dy, diffusion, advective) &
                + (-3.0_real32 / 8.0_real32) * fa_flux_y(q, v, nx, ny, i, n - 2, inv_dy, diffusion, advective) &
                + (-17.0_real32 / 24.0_real32) * fa_flux_y(q, v, nx, ny, i, n - 1, inv_dy, diffusion, advective) &
                + (11.0_real32 / 12.0_real32) * fa_flux_y(q, v, nx, ny, i, n, inv_dy, diffusion, advective))
        end if
    end function fa_fused_div_y

    ! Right-hand side at interior cell (i, j), in the operation order of
    ! fa_advance_rk4: (div diff_x + div diff_y) - (div adv_x + div adv_y).
    pure real(real32) function fa_fused_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion) result(rhs)
        real(real32), intent(in) :: q(:,:), vx(:,:), vy(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dx, inv_dy, diffusion
        rhs = (fa_fused_div_x(q, vx, nx, ny, i - 1, j, inv_dx, diffusion, .false.) &
            + fa_fused_div_y(q, vy, nx, ny, i, j - 1, inv_dy, diffusion, .false.)) &
            - (fa_fused_div_x(q, vx, nx, ny, i - 1, j, inv_dx, diffusion, .true.) &
            + fa_fused_div_y(q, vy, nx, ny, i, j - 1, inv_dy, diffusion, .true.))
    end function fa_fused_rhs

    ! The method of fa_advance_rk4 with one `do concurrent` per RK stage: the
    ! face fluxes are computed where the divergence needs them, so no face
    ! arrays are stored.  Stage inputs alternate between two buffers (a stage
    ! cannot overwrite the field its neighbours still read): stage 1 reads
    ! base, stages 2-4 read q or q2, and stage 4 updates base directly.
    ! Boundary values are never written, so every buffer keeps those of base.
    subroutine fa_advance_rk4_fused(fields, nx, ny, steps, dt, pi, diffusion)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny, steps
        real(real32), intent(in) :: dt, pi, diffusion
        integer :: step, i, j
        real(real32) :: inv_dx, inv_dy, half_dt, dt_over_6, rhs
        real(real32), allocatable :: base(:,:), q(:,:), q2(:,:), acc(:,:)
        real(real32), allocatable :: vx(:,:), vy(:,:)

        inv_dx = real(nx, real32) / (2.0_real32 * pi)
        inv_dy = real(ny, real32) / (2.0_real32 * pi)
        half_dt = 0.5_real32 * dt
        dt_over_6 = dt / 6.0_real32
        allocate(base(1:nx + 2, 1:ny + 2))
        allocate(q(1:nx + 2, 1:ny + 2))
        allocate(q2(1:nx + 2, 1:ny + 2))
        allocate(acc(1:nx + 2, 1:ny + 2))
        allocate(vx(1:nx + 1, 1:ny + 2))
        allocate(vy(1:nx + 2, 1:ny + 1))
        do concurrent (j = 1:ny + 2, i = 1:nx + 1)
            vx(i, j) = fields%vx(i, j)
        end do
        do concurrent (j = 1:ny + 1, i = 1:nx + 2)
            vy(i, j) = fields%vy(i, j)
        end do
        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            base(i, j) = fields%base(i, j)
            q(i, j) = fields%base(i, j)
            q2(i, j) = fields%base(i, j)
            acc(i, j) = 0.0_real32
        end do

        do step = 1, steps
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused_rhs(base, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                acc(i, j) = rhs
                q(i, j) = base(i, j) + half_dt * rhs
            end do
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                acc(i, j) = acc(i, j) + 2.0_real32 * rhs
                q2(i, j) = base(i, j) + half_dt * rhs
            end do
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused_rhs(q2, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                acc(i, j) = acc(i, j) + 2.0_real32 * rhs
                q(i, j) = base(i, j) + dt * rhs
            end do
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                base(i, j) = base(i, j) + dt_over_6 * (acc(i, j) + rhs)
            end do
        end do

        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            fields%base(i, j) = base(i, j)
            fields%q(i, j) = base(i, j)
            fields%acc(i, j) = 0.0_real32
        end do
        deallocate(base, q, q2, acc, vx, vy)
    end subroutine fa_advance_rk4_fused

    ! The diffusive and advective flux at x face k from one set of
    ! loads: the rows of fa_grad_face_x and fa_interp_face_x, in
    ! their order.
    pure subroutine fa_face_fluxes_x(q, v, nx, ny, k, j, inv_dx, diffusion, fd, fa)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, k, j
        real(real32), intent(in) :: inv_dx, diffusion
        real(real32), intent(out) :: fd, fa
        real(real32) :: g, c
        integer :: n
        n = nx + 2
        if (k == 1) then
            g = inv_dx * ((-352.0_real32 / 105.0_real32) * q(1, j) &
                + (35.0_real32 / 8.0_real32) * q(2, j) &
                + (-35.0_real32 / 24.0_real32) * q(3, j) &
                + (21.0_real32 / 40.0_real32) * q(4, j) &
                + (-5.0_real32 / 56.0_real32) * q(5, j))
            c = q(1, j)
        else if (k == 2) then
            g = inv_dx * ((16.0_real32 / 105.0_real32) * q(1, j) &
                + (-31.0_real32 / 24.0_real32) * q(2, j) &
                + (29.0_real32 / 24.0_real32) * q(3, j) &
                + (-3.0_real32 / 40.0_real32) * q(4, j) &
                + (1.0_real32 / 168.0_real32) * q(5, j))
            c = (-16.0_real32 / 112.0_real32) * q(1, j) &
                + (70.0_real32 / 112.0_real32) * q(2, j) &
                + (70.0_real32 / 112.0_real32) * q(3, j) &
                + (-14.0_real32 / 112.0_real32) * q(4, j) &
                + (2.0_real32 / 112.0_real32) * q(5, j)
        else if (k + 2 < n) then
            g = inv_dx * ((1.0_real32 / 24.0_real32) * q(k - 1, j) &
                + (-9.0_real32 / 8.0_real32) * q(k, j) &
                + (9.0_real32 / 8.0_real32) * q(k + 1, j) &
                + (-1.0_real32 / 24.0_real32) * q(k + 2, j))
            c = (-7.0_real32 / 112.0_real32) * q(k - 1, j) &
                + (63.0_real32 / 112.0_real32) * q(k, j) &
                + (63.0_real32 / 112.0_real32) * q(k + 1, j) &
                + (-7.0_real32 / 112.0_real32) * q(k + 2, j)
        else if (k == n - 2) then
            g = inv_dx * ((-1.0_real32 / 168.0_real32) * q(n - 4, j) &
                + (3.0_real32 / 40.0_real32) * q(n - 3, j) &
                + (-29.0_real32 / 24.0_real32) * q(n - 2, j) &
                + (31.0_real32 / 24.0_real32) * q(n - 1, j) &
                + (-16.0_real32 / 105.0_real32) * q(n, j))
            c = (2.0_real32 / 112.0_real32) * q(n - 4, j) &
                + (-14.0_real32 / 112.0_real32) * q(n - 3, j) &
                + (70.0_real32 / 112.0_real32) * q(n - 2, j) &
                + (70.0_real32 / 112.0_real32) * q(n - 1, j) &
                + (-16.0_real32 / 112.0_real32) * q(n, j)
        else
            g = inv_dx * ((5.0_real32 / 56.0_real32) * q(n - 4, j) &
                + (-21.0_real32 / 40.0_real32) * q(n - 3, j) &
                + (35.0_real32 / 24.0_real32) * q(n - 2, j) &
                + (-35.0_real32 / 8.0_real32) * q(n - 1, j) &
                + (352.0_real32 / 105.0_real32) * q(n, j))
            c = q(n, j)
        end if
        fd = diffusion * g
        fa = v(k, j) * c
    end subroutine fa_face_fluxes_x

    ! The diffusive and advective flux at y face k from one set of
    ! loads: the rows of fa_grad_face_y and fa_interp_face_y, in
    ! their order.
    pure subroutine fa_face_fluxes_y(q, v, nx, ny, i, k, inv_dy, diffusion, fd, fa)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, i, k
        real(real32), intent(in) :: inv_dy, diffusion
        real(real32), intent(out) :: fd, fa
        real(real32) :: g, c
        integer :: n
        n = ny + 2
        if (k == 1) then
            g = inv_dy * ((-352.0_real32 / 105.0_real32) * q(i, 1) &
                + (35.0_real32 / 8.0_real32) * q(i, 2) &
                + (-35.0_real32 / 24.0_real32) * q(i, 3) &
                + (21.0_real32 / 40.0_real32) * q(i, 4) &
                + (-5.0_real32 / 56.0_real32) * q(i, 5))
            c = q(i, 1)
        else if (k == 2) then
            g = inv_dy * ((16.0_real32 / 105.0_real32) * q(i, 1) &
                + (-31.0_real32 / 24.0_real32) * q(i, 2) &
                + (29.0_real32 / 24.0_real32) * q(i, 3) &
                + (-3.0_real32 / 40.0_real32) * q(i, 4) &
                + (1.0_real32 / 168.0_real32) * q(i, 5))
            c = (-16.0_real32 / 112.0_real32) * q(i, 1) &
                + (70.0_real32 / 112.0_real32) * q(i, 2) &
                + (70.0_real32 / 112.0_real32) * q(i, 3) &
                + (-14.0_real32 / 112.0_real32) * q(i, 4) &
                + (2.0_real32 / 112.0_real32) * q(i, 5)
        else if (k + 2 < n) then
            g = inv_dy * ((1.0_real32 / 24.0_real32) * q(i, k - 1) &
                + (-9.0_real32 / 8.0_real32) * q(i, k) &
                + (9.0_real32 / 8.0_real32) * q(i, k + 1) &
                + (-1.0_real32 / 24.0_real32) * q(i, k + 2))
            c = (-7.0_real32 / 112.0_real32) * q(i, k - 1) &
                + (63.0_real32 / 112.0_real32) * q(i, k) &
                + (63.0_real32 / 112.0_real32) * q(i, k + 1) &
                + (-7.0_real32 / 112.0_real32) * q(i, k + 2)
        else if (k == n - 2) then
            g = inv_dy * ((-1.0_real32 / 168.0_real32) * q(i, n - 4) &
                + (3.0_real32 / 40.0_real32) * q(i, n - 3) &
                + (-29.0_real32 / 24.0_real32) * q(i, n - 2) &
                + (31.0_real32 / 24.0_real32) * q(i, n - 1) &
                + (-16.0_real32 / 105.0_real32) * q(i, n))
            c = (2.0_real32 / 112.0_real32) * q(i, n - 4) &
                + (-14.0_real32 / 112.0_real32) * q(i, n - 3) &
                + (70.0_real32 / 112.0_real32) * q(i, n - 2) &
                + (70.0_real32 / 112.0_real32) * q(i, n - 1) &
                + (-16.0_real32 / 112.0_real32) * q(i, n)
        else
            g = inv_dy * ((5.0_real32 / 56.0_real32) * q(i, n - 4) &
                + (-21.0_real32 / 40.0_real32) * q(i, n - 3) &
                + (35.0_real32 / 24.0_real32) * q(i, n - 2) &
                + (-35.0_real32 / 8.0_real32) * q(i, n - 1) &
                + (352.0_real32 / 105.0_real32) * q(i, n))
            c = q(i, n)
        end if
        fd = diffusion * g
        fa = v(i, k) * c
    end subroutine fa_face_fluxes_y

    ! fa_div_face_x of the diffusive (dd) and the advective (da) face
    ! fluxes of q, each face computed once for both.
    pure subroutine fa_fused2_div_x(q, v, nx, ny, i, j, inv_dx, diffusion, dd, da)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dx, diffusion
        real(real32), intent(out) :: dd, da
        real(real32) :: fd(5), fa(5)
        integer :: n
        n = nx + 1
        if (i == 1) then
            call fa_face_fluxes_x(q, v, nx, ny, 1, j, inv_dx, diffusion, fd(1), fa(1))
            call fa_face_fluxes_x(q, v, nx, ny, 2, j, inv_dx, diffusion, fd(2), fa(2))
            call fa_face_fluxes_x(q, v, nx, ny, 3, j, inv_dx, diffusion, fd(3), fa(3))
            call fa_face_fluxes_x(q, v, nx, ny, 4, j, inv_dx, diffusion, fd(4), fa(4))
            call fa_face_fluxes_x(q, v, nx, ny, 5, j, inv_dx, diffusion, fd(5), fa(5))
            dd = inv_dx * ((-11.0_real32 / 12.0_real32) * fd(1) &
                + (17.0_real32 / 24.0_real32) * fd(2) &
                + (3.0_real32 / 8.0_real32) * fd(3) &
                + (-5.0_real32 / 24.0_real32) * fd(4) &
                + (1.0_real32 / 24.0_real32) * fd(5))
            da = inv_dx * ((-11.0_real32 / 12.0_real32) * fa(1) &
                + (17.0_real32 / 24.0_real32) * fa(2) &
                + (3.0_real32 / 8.0_real32) * fa(3) &
                + (-5.0_real32 / 24.0_real32) * fa(4) &
                + (1.0_real32 / 24.0_real32) * fa(5))
        else if (i + 1 < n) then
            call fa_face_fluxes_x(q, v, nx, ny, i - 1, j, inv_dx, diffusion, fd(1), fa(1))
            call fa_face_fluxes_x(q, v, nx, ny, i, j, inv_dx, diffusion, fd(2), fa(2))
            call fa_face_fluxes_x(q, v, nx, ny, i + 1, j, inv_dx, diffusion, fd(3), fa(3))
            call fa_face_fluxes_x(q, v, nx, ny, i + 2, j, inv_dx, diffusion, fd(4), fa(4))
            dd = inv_dx * ((1.0_real32 / 24.0_real32) * fd(1) &
                + (-9.0_real32 / 8.0_real32) * fd(2) &
                + (9.0_real32 / 8.0_real32) * fd(3) &
                + (-1.0_real32 / 24.0_real32) * fd(4))
            da = inv_dx * ((1.0_real32 / 24.0_real32) * fa(1) &
                + (-9.0_real32 / 8.0_real32) * fa(2) &
                + (9.0_real32 / 8.0_real32) * fa(3) &
                + (-1.0_real32 / 24.0_real32) * fa(4))
        else
            call fa_face_fluxes_x(q, v, nx, ny, n - 4, j, inv_dx, diffusion, fd(1), fa(1))
            call fa_face_fluxes_x(q, v, nx, ny, n - 3, j, inv_dx, diffusion, fd(2), fa(2))
            call fa_face_fluxes_x(q, v, nx, ny, n - 2, j, inv_dx, diffusion, fd(3), fa(3))
            call fa_face_fluxes_x(q, v, nx, ny, n - 1, j, inv_dx, diffusion, fd(4), fa(4))
            call fa_face_fluxes_x(q, v, nx, ny, n, j, inv_dx, diffusion, fd(5), fa(5))
            dd = inv_dx * ((-1.0_real32 / 24.0_real32) * fd(1) &
                + (5.0_real32 / 24.0_real32) * fd(2) &
                + (-3.0_real32 / 8.0_real32) * fd(3) &
                + (-17.0_real32 / 24.0_real32) * fd(4) &
                + (11.0_real32 / 12.0_real32) * fd(5))
            da = inv_dx * ((-1.0_real32 / 24.0_real32) * fa(1) &
                + (5.0_real32 / 24.0_real32) * fa(2) &
                + (-3.0_real32 / 8.0_real32) * fa(3) &
                + (-17.0_real32 / 24.0_real32) * fa(4) &
                + (11.0_real32 / 12.0_real32) * fa(5))
        end if
    end subroutine fa_fused2_div_x

    ! fa_div_face_y of the diffusive (dd) and the advective (da) face
    ! fluxes of q, each face computed once for both.
    pure subroutine fa_fused2_div_y(q, v, nx, ny, i, j, inv_dy, diffusion, dd, da)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dy, diffusion
        real(real32), intent(out) :: dd, da
        real(real32) :: fd(5), fa(5)
        integer :: n
        n = ny + 1
        if (j == 1) then
            call fa_face_fluxes_y(q, v, nx, ny, i, 1, inv_dy, diffusion, fd(1), fa(1))
            call fa_face_fluxes_y(q, v, nx, ny, i, 2, inv_dy, diffusion, fd(2), fa(2))
            call fa_face_fluxes_y(q, v, nx, ny, i, 3, inv_dy, diffusion, fd(3), fa(3))
            call fa_face_fluxes_y(q, v, nx, ny, i, 4, inv_dy, diffusion, fd(4), fa(4))
            call fa_face_fluxes_y(q, v, nx, ny, i, 5, inv_dy, diffusion, fd(5), fa(5))
            dd = inv_dy * ((-11.0_real32 / 12.0_real32) * fd(1) &
                + (17.0_real32 / 24.0_real32) * fd(2) &
                + (3.0_real32 / 8.0_real32) * fd(3) &
                + (-5.0_real32 / 24.0_real32) * fd(4) &
                + (1.0_real32 / 24.0_real32) * fd(5))
            da = inv_dy * ((-11.0_real32 / 12.0_real32) * fa(1) &
                + (17.0_real32 / 24.0_real32) * fa(2) &
                + (3.0_real32 / 8.0_real32) * fa(3) &
                + (-5.0_real32 / 24.0_real32) * fa(4) &
                + (1.0_real32 / 24.0_real32) * fa(5))
        else if (j + 1 < n) then
            call fa_face_fluxes_y(q, v, nx, ny, i, j - 1, inv_dy, diffusion, fd(1), fa(1))
            call fa_face_fluxes_y(q, v, nx, ny, i, j, inv_dy, diffusion, fd(2), fa(2))
            call fa_face_fluxes_y(q, v, nx, ny, i, j + 1, inv_dy, diffusion, fd(3), fa(3))
            call fa_face_fluxes_y(q, v, nx, ny, i, j + 2, inv_dy, diffusion, fd(4), fa(4))
            dd = inv_dy * ((1.0_real32 / 24.0_real32) * fd(1) &
                + (-9.0_real32 / 8.0_real32) * fd(2) &
                + (9.0_real32 / 8.0_real32) * fd(3) &
                + (-1.0_real32 / 24.0_real32) * fd(4))
            da = inv_dy * ((1.0_real32 / 24.0_real32) * fa(1) &
                + (-9.0_real32 / 8.0_real32) * fa(2) &
                + (9.0_real32 / 8.0_real32) * fa(3) &
                + (-1.0_real32 / 24.0_real32) * fa(4))
        else
            call fa_face_fluxes_y(q, v, nx, ny, i, n - 4, inv_dy, diffusion, fd(1), fa(1))
            call fa_face_fluxes_y(q, v, nx, ny, i, n - 3, inv_dy, diffusion, fd(2), fa(2))
            call fa_face_fluxes_y(q, v, nx, ny, i, n - 2, inv_dy, diffusion, fd(3), fa(3))
            call fa_face_fluxes_y(q, v, nx, ny, i, n - 1, inv_dy, diffusion, fd(4), fa(4))
            call fa_face_fluxes_y(q, v, nx, ny, i, n, inv_dy, diffusion, fd(5), fa(5))
            dd = inv_dy * ((-1.0_real32 / 24.0_real32) * fd(1) &
                + (5.0_real32 / 24.0_real32) * fd(2) &
                + (-3.0_real32 / 8.0_real32) * fd(3) &
                + (-17.0_real32 / 24.0_real32) * fd(4) &
                + (11.0_real32 / 12.0_real32) * fd(5))
            da = inv_dy * ((-1.0_real32 / 24.0_real32) * fa(1) &
                + (5.0_real32 / 24.0_real32) * fa(2) &
                + (-3.0_real32 / 8.0_real32) * fa(3) &
                + (-17.0_real32 / 24.0_real32) * fa(4) &
                + (11.0_real32 / 12.0_real32) * fa(5))
        end if
    end subroutine fa_fused2_div_y

    ! fa_fused_rhs with each face flux pair computed once.
    pure real(real32) function fa_fused2_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion) result(rhs)
        real(real32), intent(in) :: q(:,:), vx(:,:), vy(:,:)
        integer, intent(in) :: nx, ny, i, j
        real(real32), intent(in) :: inv_dx, inv_dy, diffusion
        real(real32) :: dxd, dxa, dyd, dya
        call fa_fused2_div_x(q, vx, nx, ny, i - 1, j, inv_dx, diffusion, dxd, dxa)
        call fa_fused2_div_y(q, vy, nx, ny, i, j - 1, inv_dy, diffusion, dyd, dya)
        rhs = (dxd + dyd) - (dxa + dya)
    end function fa_fused2_rhs

    ! fa_advance_rk4_fused with fa_fused2_rhs.
    subroutine fa_advance_rk4_fused2(fields, nx, ny, steps, dt, pi, diffusion)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny, steps
        real(real32), intent(in) :: dt, pi, diffusion
        integer :: step, i, j
        real(real32) :: inv_dx, inv_dy, half_dt, dt_over_6, rhs
        real(real32), allocatable :: base(:,:), q(:,:), q2(:,:), acc(:,:)
        real(real32), allocatable :: vx(:,:), vy(:,:)

        inv_dx = real(nx, real32) / (2.0_real32 * pi)
        inv_dy = real(ny, real32) / (2.0_real32 * pi)
        half_dt = 0.5_real32 * dt
        dt_over_6 = dt / 6.0_real32
        allocate(base(1:nx + 2, 1:ny + 2))
        allocate(q(1:nx + 2, 1:ny + 2))
        allocate(q2(1:nx + 2, 1:ny + 2))
        allocate(acc(1:nx + 2, 1:ny + 2))
        allocate(vx(1:nx + 1, 1:ny + 2))
        allocate(vy(1:nx + 2, 1:ny + 1))
        do concurrent (j = 1:ny + 2, i = 1:nx + 1)
            vx(i, j) = fields%vx(i, j)
        end do
        do concurrent (j = 1:ny + 1, i = 1:nx + 2)
            vy(i, j) = fields%vy(i, j)
        end do
        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            base(i, j) = fields%base(i, j)
            q(i, j) = fields%base(i, j)
            q2(i, j) = fields%base(i, j)
            acc(i, j) = 0.0_real32
        end do

        do step = 1, steps
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused2_rhs(base, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                acc(i, j) = rhs
                q(i, j) = base(i, j) + half_dt * rhs
            end do
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused2_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                acc(i, j) = acc(i, j) + 2.0_real32 * rhs
                q2(i, j) = base(i, j) + half_dt * rhs
            end do
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused2_rhs(q2, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                acc(i, j) = acc(i, j) + 2.0_real32 * rhs
                q(i, j) = base(i, j) + dt * rhs
            end do
            do concurrent (j = 2:ny + 1, i = 2:nx + 1) local(rhs)
                rhs = fa_fused2_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                base(i, j) = base(i, j) + dt_over_6 * (acc(i, j) + rhs)
            end do
        end do

        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            fields%base(i, j) = base(i, j)
            fields%q(i, j) = base(i, j)
            fields%acc(i, j) = 0.0_real32
        end do
        deallocate(base, q, q2, acc, vx, vy)
    end subroutine fa_advance_rk4_fused2

    ! The diffusive and advective flux at x face k of an interior face
    ! (2 < k < nx): the interior rows of fa_face_fluxes_x, in their order.
    pure subroutine fa_face_fluxes_x_interior(q, v, k, j, inv_dx, diffusion, fd, fa)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: k, j
        real(real32), intent(in) :: inv_dx, diffusion
        real(real32), intent(out) :: fd, fa
        real(real32) :: g, c
        g = inv_dx * ((1.0_real32 / 24.0_real32) * q(k - 1, j) &
            + (-9.0_real32 / 8.0_real32) * q(k, j) &
            + (9.0_real32 / 8.0_real32) * q(k + 1, j) &
            + (-1.0_real32 / 24.0_real32) * q(k + 2, j))
        c = (-7.0_real32 / 112.0_real32) * q(k - 1, j) &
            + (63.0_real32 / 112.0_real32) * q(k, j) &
            + (63.0_real32 / 112.0_real32) * q(k + 1, j) &
            + (-7.0_real32 / 112.0_real32) * q(k + 2, j)
        fd = diffusion * g
        fa = v(k, j) * c
    end subroutine fa_face_fluxes_x_interior

    ! The diffusive and advective flux at y face k of an interior face
    ! (2 < k < ny): the interior rows of fa_face_fluxes_y, in their order.
    pure subroutine fa_face_fluxes_y_interior(q, v, i, k, inv_dy, diffusion, fd, fa)
        real(real32), intent(in) :: q(:,:), v(:,:)
        integer, intent(in) :: i, k
        real(real32), intent(in) :: inv_dy, diffusion
        real(real32), intent(out) :: fd, fa
        real(real32) :: g, c
        g = inv_dy * ((1.0_real32 / 24.0_real32) * q(i, k - 1) &
            + (-9.0_real32 / 8.0_real32) * q(i, k) &
            + (9.0_real32 / 8.0_real32) * q(i, k + 1) &
            + (-1.0_real32 / 24.0_real32) * q(i, k + 2))
        c = (-7.0_real32 / 112.0_real32) * q(i, k - 1) &
            + (63.0_real32 / 112.0_real32) * q(i, k) &
            + (63.0_real32 / 112.0_real32) * q(i, k + 1) &
            + (-7.0_real32 / 112.0_real32) * q(i, k + 2)
        fd = diffusion * g
        fa = v(i, k) * c
    end subroutine fa_face_fluxes_y_interior

    ! fa_fused2_rhs of the fa_regy_rows cells (i, j0), ..., (i, j0 +
    ! fa_regy_rows - 1) of the interior, where every face and divergence
    ! uses the interior rows (5 <= i <= nx - 2, 5 <= j0, j0 + fa_regy_rows
    ! - 1 <= ny - 2). The y faces between the cells are computed once for
    ! all of them; every value is computed as in fa_fused2_rhs.
    pure subroutine fa_regy_rhs(q, vx, vy, i, j0, inv_dx, inv_dy, diffusion, rhs)
        real(real32), intent(in) :: q(:,:), vx(:,:), vy(:,:)
        integer, intent(in) :: i, j0
        real(real32), intent(in) :: inv_dx, inv_dy, diffusion
        real(real32), intent(out) :: rhs(fa_regy_rows)
        real(real32) :: yd(fa_regy_rows + 3), ya(fa_regy_rows + 3)
        real(real32) :: fd(4), fa(4), dxd, dxa, dyd, dya
        integer :: r, k, j
        do r = 1, fa_regy_rows + 3
            call fa_face_fluxes_y_interior(q, vy, i, j0 - 3 + r, inv_dy, diffusion, yd(r), ya(r))
        end do
        do r = 1, fa_regy_rows
            j = j0 + r - 1
            do k = 1, 4
                call fa_face_fluxes_x_interior(q, vx, i - 3 + k, j, inv_dx, diffusion, fd(k), fa(k))
            end do
            dxd = inv_dx * ((1.0_real32 / 24.0_real32) * fd(1) &
                + (-9.0_real32 / 8.0_real32) * fd(2) &
                + (9.0_real32 / 8.0_real32) * fd(3) &
                + (-1.0_real32 / 24.0_real32) * fd(4))
            dxa = inv_dx * ((1.0_real32 / 24.0_real32) * fa(1) &
                + (-9.0_real32 / 8.0_real32) * fa(2) &
                + (9.0_real32 / 8.0_real32) * fa(3) &
                + (-1.0_real32 / 24.0_real32) * fa(4))
            dyd = inv_dy * ((1.0_real32 / 24.0_real32) * yd(r) &
                + (-9.0_real32 / 8.0_real32) * yd(r + 1) &
                + (9.0_real32 / 8.0_real32) * yd(r + 2) &
                + (-1.0_real32 / 24.0_real32) * yd(r + 3))
            dya = inv_dy * ((1.0_real32 / 24.0_real32) * ya(r) &
                + (-9.0_real32 / 8.0_real32) * ya(r + 1) &
                + (9.0_real32 / 8.0_real32) * ya(r + 2) &
                + (-1.0_real32 / 24.0_real32) * ya(r + 3))
            rhs(r) = (dxd + dyd) - (dxa + dya)
        end do
    end subroutine fa_regy_rhs

    ! Cell t of the frame of fa_advance_rk4_regy: the cells (i, j),
    ! 2 <= i <= nx + 1, 2 <= j <= ny + 1, outside the interior blocks
    ! 5 <= i <= nx - 2, 5 <= j < jtop. First the rows (j < 5 or j >= jtop)
    ! along x, then the three columns at each side of the block rows.
    pure subroutine fa_regy_frame_cell(t, nx, ny, jtop, i, j)
        integer, intent(in) :: t, nx, ny, jtop
        integer, intent(out) :: i, j
        integer :: s, n_row_cells, c
        n_row_cells = (min(3, ny) + max(0, ny + 2 - max(jtop, 5))) * nx
        if (t <= n_row_cells) then
            s = (t - 1) / nx
            i = 2 + (t - 1 - s * nx)
            if (s < 3) then
                j = 2 + s
            else
                j = max(jtop, 5) + s - 3
            end if
        else
            s = t - 1 - n_row_cells
            c = s / (jtop - 5)
            j = 5 + (s - c * (jtop - 5))
            if (c < 3) then
                i = 2 + c
            else
                i = nx - 4 + c
            end if
        end if
    end subroutine fa_regy_frame_cell

    ! fa_advance_rk4_fused2 with the interior cells, whose faces and
    ! divergences use only the interior operator rows, computed in blocks
    ! of fa_regy_rows cells along y that share their y faces (fa_regy_rhs),
    ! without any branch over the boundary rows. The frame of cells near
    ! the boundaries runs fa_fused2_rhs in the same loop, in iterations
    ! after those of the interior blocks (jb >= nb), so that it runs
    ! alongside the interior rather than after it. Only fields%base is
    ! written back: the stage buffers and the accumulator are the
    ! routine's own, and fields%q and fields%acc are left as they are.
    subroutine fa_advance_rk4_regy(fields, nx, ny, steps, dt, pi, diffusion)
        type(fa_fields_t), intent(inout) :: fields
        integer, intent(in) :: nx, ny, steps
        real(real32), intent(in) :: dt, pi, diffusion
        integer :: step, i, ii, j, jb, r, t, nb, nw, nf, jtop, n_frame
        real(real32) :: inv_dx, inv_dy, half_dt, dt_over_6, rhs
        real(real32) :: rhs_block(fa_regy_rows)
        real(real32), allocatable :: base(:,:), q(:,:), q2(:,:), acc(:,:)
        real(real32), allocatable :: vx(:,:), vy(:,:)

        inv_dx = real(nx, real32) / (2.0_real32 * pi)
        inv_dy = real(ny, real32) / (2.0_real32 * pi)
        half_dt = 0.5_real32 * dt
        dt_over_6 = dt / 6.0_real32
        ! Interior blocks: i = 5:nx-2 (nw columns), j = 5:jtop-1 in nb
        ! blocks of rows; then nf rows of nw frame cells.
        if (nx >= 7) then
            nb = max(0, ny - 6) / fa_regy_rows
        else
            nb = 0
        end if
        jtop = 5 + nb * fa_regy_rows
        if (nb > 0) then
            nw = nx - 6
            n_frame = (3 + ny + 2 - jtop) * nx + 6 * (jtop - 5)
        else
            nw = nx
            n_frame = nx * ny
        end if
        nf = (n_frame + nw - 1) / nw
        allocate(base(1:nx + 2, 1:ny + 2))
        allocate(q(1:nx + 2, 1:ny + 2))
        allocate(q2(1:nx + 2, 1:ny + 2))
        allocate(acc(1:nx + 2, 1:ny + 2))
        allocate(vx(1:nx + 1, 1:ny + 2))
        allocate(vy(1:nx + 2, 1:ny + 1))
        do concurrent (j = 1:ny + 2, i = 1:nx + 1)
            vx(i, j) = fields%vx(i, j)
        end do
        do concurrent (j = 1:ny + 1, i = 1:nx + 2)
            vy(i, j) = fields%vy(i, j)
        end do
        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            base(i, j) = fields%base(i, j)
            q(i, j) = fields%base(i, j)
            q2(i, j) = fields%base(i, j)
            acc(i, j) = 0.0_real32
        end do

        do step = 1, steps
            do concurrent (ii = 1:nw, jb = 0:nb + nf - 1) local(rhs_block, rhs, r, i, j, t)
                if (jb < nb) then
                    i = ii + 4
                    call fa_regy_rhs(base, vx, vy, i, 5 + jb * fa_regy_rows, inv_dx, inv_dy, diffusion, rhs_block)
                    do r = 1, fa_regy_rows
                        j = 5 + jb * fa_regy_rows + r - 1
                        acc(i, j) = rhs_block(r)
                        q(i, j) = base(i, j) + half_dt * rhs_block(r)
                    end do
                else
                    t = (jb - nb) * nw + ii
                    if (t <= n_frame) then
                        call fa_regy_frame_cell(t, nx, ny, jtop, i, j)
                        rhs = fa_fused2_rhs(base, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                        acc(i, j) = rhs
                        q(i, j) = base(i, j) + half_dt * rhs
                    end if
                end if
            end do
            do concurrent (ii = 1:nw, jb = 0:nb + nf - 1) local(rhs_block, rhs, r, i, j, t)
                if (jb < nb) then
                    i = ii + 4
                    call fa_regy_rhs(q, vx, vy, i, 5 + jb * fa_regy_rows, inv_dx, inv_dy, diffusion, rhs_block)
                    do r = 1, fa_regy_rows
                        j = 5 + jb * fa_regy_rows + r - 1
                        acc(i, j) = acc(i, j) + 2.0_real32 * rhs_block(r)
                        q2(i, j) = base(i, j) + half_dt * rhs_block(r)
                    end do
                else
                    t = (jb - nb) * nw + ii
                    if (t <= n_frame) then
                        call fa_regy_frame_cell(t, nx, ny, jtop, i, j)
                        rhs = fa_fused2_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                        acc(i, j) = acc(i, j) + 2.0_real32 * rhs
                        q2(i, j) = base(i, j) + half_dt * rhs
                    end if
                end if
            end do
            do concurrent (ii = 1:nw, jb = 0:nb + nf - 1) local(rhs_block, rhs, r, i, j, t)
                if (jb < nb) then
                    i = ii + 4
                    call fa_regy_rhs(q2, vx, vy, i, 5 + jb * fa_regy_rows, inv_dx, inv_dy, diffusion, rhs_block)
                    do r = 1, fa_regy_rows
                        j = 5 + jb * fa_regy_rows + r - 1
                        acc(i, j) = acc(i, j) + 2.0_real32 * rhs_block(r)
                        q(i, j) = base(i, j) + dt * rhs_block(r)
                    end do
                else
                    t = (jb - nb) * nw + ii
                    if (t <= n_frame) then
                        call fa_regy_frame_cell(t, nx, ny, jtop, i, j)
                        rhs = fa_fused2_rhs(q2, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                        acc(i, j) = acc(i, j) + 2.0_real32 * rhs
                        q(i, j) = base(i, j) + dt * rhs
                    end if
                end if
            end do
            do concurrent (ii = 1:nw, jb = 0:nb + nf - 1) local(rhs_block, rhs, r, i, j, t)
                if (jb < nb) then
                    i = ii + 4
                    call fa_regy_rhs(q, vx, vy, i, 5 + jb * fa_regy_rows, inv_dx, inv_dy, diffusion, rhs_block)
                    do r = 1, fa_regy_rows
                        j = 5 + jb * fa_regy_rows + r - 1
                        base(i, j) = base(i, j) + dt_over_6 * (acc(i, j) + rhs_block(r))
                    end do
                else
                    t = (jb - nb) * nw + ii
                    if (t <= n_frame) then
                        call fa_regy_frame_cell(t, nx, ny, jtop, i, j)
                        rhs = fa_fused2_rhs(q, vx, vy, nx, ny, i, j, inv_dx, inv_dy, diffusion)
                        base(i, j) = base(i, j) + dt_over_6 * (acc(i, j) + rhs)
                    end if
                end if
            end do
        end do

        do concurrent (j = 1:ny + 2, i = 1:nx + 2)
            fields%base(i, j) = base(i, j)
        end do
        deallocate(base, q, q2, acc, vx, vy)
    end subroutine fa_advance_rk4_regy

    subroutine fa_write_csv(path, fields, nx, ny)
        character(len=*), intent(in) :: path
        type(fa_fields_t), intent(in) :: fields
        integer, intent(in) :: nx, ny
        integer :: unit, i, j, ios
        character(len=512) :: message
        real(real32) :: pi

        pi = acos(-1.0_real32)

        open(newunit=unit, file=path, status="replace", action="write", &
            form="formatted", iostat=ios, iomsg=message)
        if (ios /= 0) error stop message
        write(unit, '(a)') 'x,y,scalar'
        do j = 1, ny + 2
            do i = 1, nx + 2
                write(unit, '(f0.9, ",", f0.9, ",", es0.8)') &
                    fa_center_coord(i, nx, -pi, pi), &
                    fa_center_coord(j, ny, -pi, pi), &
                    fields%base(i, j)
            end do
            write(unit, *)
        end do
        close(unit)
    end subroutine fa_write_csv

    subroutine fa_parse_args(opts)
        type(fa_options_t), intent(inout) :: opts
        integer :: argc, i
        character(len=256) :: arg, value

        argc = command_argument_count()
        i = 1
        do while (i <= argc)
            call get_command_argument(i, arg)
            select case (trim(arg))
            case ("--cells")
                i = i + 1
                call get_command_argument(i, value)
                read(value, *) opts%cfg%nx, opts%cfg%ny
            case ("--steps")
                i = i + 1
                call get_command_argument(i, value)
                read(value, *) opts%cfg%steps
            case ("--dt")
                i = i + 1
                call get_command_argument(i, value)
                read(value, *) opts%cfg%dt
                opts%dt_supplied = .true.
            case ("--gpu-profile")
                opts%gpu_profile = .true.
            case ("--input")
                i = i + 1
                call get_command_argument(i, value)
                opts%input_path = trim(value)
            case ("--dump", "--csv")
                opts%dump_csv = .true.
                if (i < argc) then
                    call get_command_argument(i + 1, value)
                    if (value(1:1) /= '-') then
                        opts%output_path = trim(value)
                        i = i + 1
                    end if
                end if
            case ("--output")
                i = i + 1
                call get_command_argument(i, value)
                opts%output_path = trim(value)
                opts%dump_csv = .true.
            case ("--help", "-h")
                write(*, '(a)') "usage: fortran_workspace [--cells nx,ny] " // &
                    "[--steps n] [--dt dt] [--input path] " // &
                    "[--dump [csv]] [--output path] [--gpu-profile]"
                stop 0
            case default
                if (arg(1:1) /= '-') then
                    ! Positional numeric convenience: nx ny steps dt.
                    if (i == 1 .and. argc >= 4) then
                        read(arg, *) opts%cfg%nx
                        call get_command_argument(2, value)
                        read(value, *) opts%cfg%ny
                        call get_command_argument(3, value)
                        read(value, *) opts%cfg%steps
                        call get_command_argument(4, value)
                        read(value, *) opts%cfg%dt
                        opts%dt_supplied = .true.
                        i = 4
                    else
                        error stop "unknown argument"
                    end if
                else
                    error stop "unknown argument"
                end if
            end select
            i = i + 1
        end do
        if (.not. allocated(opts%output_path)) then
            opts%output_path = "build/fortran/final.csv"
        end if
    end subroutine fa_parse_args

end module formal_advection_workspace
