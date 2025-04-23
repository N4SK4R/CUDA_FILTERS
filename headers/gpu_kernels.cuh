__global__ void gpu_sepia(float *r_in, float *g_in, float *b_in,float *r_out, float *g_out, float *b_out,int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x < width && y < height) {
        float r0 = r_in[i], g0 = g_in[i], b0 = b_in[i];
        r_out[i] = fminf(r0 * 0.393f + g0 * 0.769f + b0 * 0.189f, 1.0f);
        g_out[i] = fminf(r0 * 0.349f + g0 * 0.686f + b0 * 0.168f, 1.0f);
        b_out[i] = fminf(r0 * 0.272f + g0 * 0.534f + b0 * 0.131f, 1.0f);
    }
}

__global__ void gpu_invert(float *r_in, float *g_in, float *b_in,float *r_out, float *g_out, float *b_out,int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * width + x;

    if (x < width && y < height) {
        r_out[i] = 1.0f - r_in[i];
        g_out[i] = 1.0f - g_in[i];
        b_out[i] = 1.0f - b_in[i];
    }
}

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

__global__ void rgb_to_grayscale(const float *r, const float *g, const float *b,float *out_r, float *out_g, float *out_b,int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int i = y * width + x;
    float gray = 0.299f * r[i] + 0.587f * g[i] + 0.114f * b[i];

    out_r[i] = gray;
    out_g[i] = gray;
    out_b[i] = gray;
}
