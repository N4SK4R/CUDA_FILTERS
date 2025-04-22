#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.cuh"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define FILTER_RADIUS 2

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

void draw_rgb(Display *display, Window win, GC gc, Visual *visual, int depth, unsigned char *rgb_data, int w, int h, int x_offset) {
    XImage *img = XCreateImage(display, visual, depth, ZPixmap, 0, NULL, w, h, 32, 0);
    img->data = (char *)malloc(w * h * 4);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i = (y * w + x) * 3;
            unsigned char r = rgb_data[i];
            unsigned char g = rgb_data[i + 1];
            unsigned char b = rgb_data[i + 2];
            unsigned int pixel = (r << 16) | (g << 8) | b;
            ((unsigned int *)img->data)[y * w + x] = pixel;
        }
    }
    XPutImage(display, win, gc, img, 0, 0, x_offset, 0, w, h);
    XDestroyImage(img);
}


int main() {
    int width, height, channels;
    unsigned char *img_data = stbi_load("images/test2.jpg", &width, &height, &channels, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }

    size_t size = width * height * sizeof(float);
    float *h_r = (float *)malloc(size);
    float *h_g = (float *)malloc(size);
    float *h_b = (float *)malloc(size);

    for (int i = 0; i < width * height; ++i) {
        h_r[i] = img_data[3 * i + 0] / 255.0f;
        h_g[i] = img_data[3 * i + 1] / 255.0f;
        h_b[i] = img_data[3 * i + 2] / 255.0f;
    }

    float h_filter[25] = {
        1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
        1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
        1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
        1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,
        1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25
    };

    float *d_r_in, *d_g_in, *d_b_in;
    float *d_r_out, *d_g_out, *d_b_out;
    float *d_filter;

    cudaMalloc(&d_r_in, size); cudaMalloc(&d_r_out, size);
    cudaMalloc(&d_g_in, size); cudaMalloc(&d_g_out, size);
    cudaMalloc(&d_b_in, size); cudaMalloc(&d_b_out, size);
    cudaMalloc(&d_filter, sizeof(h_filter));

    cudaMemcpy(d_r_in, h_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_in, h_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    gpu_conv2d_kernel<<<grid, block>>>(d_r_in, d_filter, d_r_out, height, width);
    gpu_conv2d_kernel<<<grid, block>>>(d_g_in, d_filter, d_g_out, height, width);
    gpu_conv2d_kernel<<<grid, block>>>(d_b_in, d_filter, d_b_out, height, width);
    cudaDeviceSynchronize();

    float *h_r_out = (float *)malloc(size);
    float *h_g_out = (float *)malloc(size);
    float *h_b_out = (float *)malloc(size);

    cudaMemcpy(h_r_out, d_r_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g_out, d_g_out, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_out, d_b_out, size, cudaMemcpyDeviceToHost);

    unsigned char *out_rgb = (unsigned char *)malloc(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        out_rgb[3 * i + 0] = (unsigned char)(fminf(fmaxf(h_r_out[i], 0.0f), 1.0f) * 255.0f);
        out_rgb[3 * i + 1] = (unsigned char)(fminf(fmaxf(h_g_out[i], 0.0f), 1.0f) * 255.0f);
        out_rgb[3 * i + 2] = (unsigned char)(fminf(fmaxf(h_b_out[i], 0.0f), 1.0f) * 255.0f);
    }

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

    XSelectInput(display, win, StructureNotifyMask);
    XMapWindow(display, win);
    for (;;) {
        XEvent e;
        XNextEvent(display, &e);
        if (e.type == MapNotify) break;
    }

    GC gc = XCreateGC(display, win, 0, NULL);
    draw_rgb(display, win, gc, visual, depth, img_data, width, height, 0);
    draw_rgb(display, win, gc, visual, depth, out_rgb, width, height, width);
    XFlush(display);

    sleep(10);

    stbi_image_free(img_data);
    free(h_r); free(h_g); free(h_b);
    free(h_r_out); free(h_g_out); free(h_b_out);
    free(out_rgb);
    cudaFree(d_r_in); cudaFree(d_g_in); cudaFree(d_b_in);
    cudaFree(d_r_out); cudaFree(d_g_out); cudaFree(d_b_out);
    cudaFree(d_filter);
    XDestroyWindow(display, win);
    XCloseDisplay(display);

    return 0;
}
