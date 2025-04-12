#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define FILTER_RADIUS 1

__global__ void gpu_conv2d_kernel(float const *d_N_ptr, float const *d_F_ptr, float *d_P_ptr, int n_rows, int n_cols) {
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_row < n_rows && out_col < n_cols) {
        float p_val = 0.0f;

        for (int f_row = 0; f_row < 2 * FILTER_RADIUS + 1; f_row++) {
            for (int f_col = 0; f_col < 2 * FILTER_RADIUS + 1; f_col++) {
                int in_row = out_row + (f_row - FILTER_RADIUS);
                int in_col = out_col + (f_col - FILTER_RADIUS);

                if (in_row >= 0 && in_row < n_rows && in_col >= 0 && in_col < n_cols) {
                    float pixel = d_N_ptr[in_row * n_cols + in_col];
                    float filter = d_F_ptr[f_row * (2 * FILTER_RADIUS + 1) + f_col];
                    p_val += filter * pixel;
                }
            }
        }
        d_P_ptr[out_row * n_cols + out_col] = p_val;
    }
}

void draw_grayscale(Display *display, Window win, GC gc, Visual *visual, int depth, unsigned char *data, int w, int h, int x_offset) {
    XImage *img = XCreateImage(display, visual, depth, ZPixmap, 0, NULL, w, h, 32, 0);
    img->data = (char *)malloc(w * h * 4);
    for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
        unsigned char v = data[y * w + x];
        unsigned int pixel = (v << 16) | (v << 8) | v;
        ((unsigned int *)img->data)[y * w + x] = pixel;
    }
    XPutImage(display, win, gc, img, 0, 0, x_offset, 0, w, h);
    XDestroyImage(img);
}

int main() {

    int width, height, channels;
    unsigned char *img_data = stbi_load("test.jpg", &width, &height, &channels, 1);
    if (!img_data) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }

    size_t size = width * height * sizeof(float);
    float *h_input = (float *)malloc(size);
    for (int i = 0; i < width * height; ++i)
        h_input[i] = img_data[i] / 255.0f;

    // Blur kernel
    float h_filter[9] = {
        1.f/9, 1.f/9, 1.f/9,
        1.f/9, 1.f/9, 1.f/9,
        1.f/9, 1.f/9, 1.f/9
    };

    float *d_input, *d_output, *d_filter;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_filter, sizeof(h_filter));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(ceil((width)/(float)16),ceil((height)/(float)16));
    gpu_conv2d_kernel<<<grid, block>>>(d_input, d_filter, d_output, height, width);
    cudaDeviceSynchronize();

    float *h_output = (float *)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    unsigned char *out_img = (unsigned char *)malloc(width * height);
    for (int i = 0; i < width * height; ++i)
        out_img[i] = (unsigned char)(fminf(fmaxf(h_output[i], 0.0f), 1.0f) * 255.0f);

    // Display using X11
    Display *display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Cannot open X display\n");
        return 1;
    }

    int screen = DefaultScreen(display);
    Visual *visual = DefaultVisual(display, screen);
    int depth = DefaultDepth(display, screen);
    int win_width = 2 * width;
    int win_height = height;

    Window win = XCreateSimpleWindow(display, RootWindow(display, screen),
                                     10, 10, win_width, win_height, 1,
                                     BlackPixel(display, screen),
                                     WhitePixel(display, screen));
    XMapWindow(display, win);
    XFlush(display);

    GC gc = XCreateGC(display, win, 0, NULL);
    draw_grayscale(display, win, gc, visual, depth, img_data, width, height, 0);
    draw_grayscale(display, win, gc, visual, depth, out_img, width, height, width);
    XFlush(display);

    sleep(10);

    stbi_image_free(img_data);
    free(h_input);
    free(h_output);
    free(out_img);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    XDestroyWindow(display, win);
    XCloseDisplay(display);

}
