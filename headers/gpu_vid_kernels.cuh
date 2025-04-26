__global__ void rgb_to_grayscale_kernel(unsigned char *rgb, float *gray, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (idx < total) {
        int rgb_idx = idx * 3;
        unsigned char r = rgb[rgb_idx + 0];
        unsigned char g = rgb[rgb_idx + 1];
        unsigned char b = rgb[rgb_idx + 2];

        gray[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

float sobel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

float sobel_y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

__global__ void gpu_conv2d_kernel(float const *d_N_ptr, float const *d_F_ptr, float *d_P_ptr, int n_rows, int n_cols, int filter_radius) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_row < n_rows && out_col < n_cols) {
        float p_val = 0.0f;

        for (int f_row = 0; f_row < 2 * filter_radius + 1; f_row++) {
            for (int f_col = 0; f_col < 2 * filter_radius + 1; f_col++) {
                int in_row = out_row + (f_row - filter_radius);
                int in_col = out_col + (f_col - filter_radius);

                if (in_row >= 0 && in_row < n_rows && in_col >= 0 && in_col < n_cols) {
                    float pixel = d_N_ptr[in_row * n_cols + in_col];
                    float filter = d_F_ptr[f_row * (2 * filter_radius + 1) + f_col];
                    p_val += filter * pixel;
                }
            }
        }
        d_P_ptr[out_row * n_cols + out_col] = p_val;
    }
}