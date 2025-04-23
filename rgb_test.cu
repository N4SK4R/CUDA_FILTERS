#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.cuh"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "headers/filters.cuh"
#include "headers/gpu_kernels.cuh"

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

int main(int argc, char **argv) {

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_path>\n", argv[0]);
        return 1;
    }

    int width, height, channels;
    unsigned char *img_data = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!img_data) {
        fprintf(stderr, "Failed to load image.\n");
        return 1;
    }

    pthread_t cli;
    pthread_create(&cli, NULL, cli_thread, NULL);

    size_t size = width * height * sizeof(float);
    float *h_r = (float *)malloc(size);
    float *h_g = (float *)malloc(size);
    float *h_b = (float *)malloc(size);

    for (int i = 0; i < width * height; ++i) {
        h_r[i] = img_data[3 * i + 0] / 255.0f;
        h_g[i] = img_data[3 * i + 1] / 255.0f;
        h_b[i] = img_data[3 * i + 2] / 255.0f;
    }

    float *d_r_in, *d_g_in, *d_b_in;
    float *d_r_out, *d_g_out, *d_b_out;

    cudaMalloc(&d_r_in, size); cudaMalloc(&d_r_out, size);
    cudaMalloc(&d_g_in, size); cudaMalloc(&d_g_out, size);
    cudaMalloc(&d_b_in, size); cudaMalloc(&d_b_out, size);

    cudaMemcpy(d_r_in, h_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_in, h_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_in, h_b, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(ceil((width)/(float)16),ceil((height)/(float)16));

    float *h_r_out = (float *)malloc(size);
    float *h_g_out = (float *)malloc(size);
    float *h_b_out = (float *)malloc(size);
    unsigned char *out_rgb = (unsigned char *)malloc(width * height * 3);
    
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
    GC gc = XCreateGC(display, win, 0, NULL);
    
    while (1) {

        if (update_requested==2) break;

        if (update_requested) {

            pthread_mutex_lock(&filter_lock);
            int r = filter_radius;
            int type = filter_type;
            update_requested = 0;
            pthread_mutex_unlock(&filter_lock);

            int fsize = (2 * r + 1);
            int filter_len = fsize * fsize;
            float *h_filter = (float *)malloc(filter_len * sizeof(float));
            choose_filter(h_filter, r, type);

            float *d_filter;
            cudaMalloc(&d_filter, filter_len * sizeof(float));
            cudaMemcpy(d_filter, h_filter, filter_len * sizeof(float), cudaMemcpyHostToDevice);

            if (filter_type == FILTER_SEPIA) 
            gpu_sepia<<<grid, block>>>(d_r_in, d_g_in, d_b_in, d_r_out, d_g_out, d_b_out, width, height);

            else if (filter_type == FILTER_GREY) 
            rgb_to_grayscale<<<grid, block>>>(d_r_in, d_g_in, d_b_in, d_r_out, d_g_out, d_b_out, width, height);

            else if (filter_type == FILTER_INVERT) 
            gpu_invert<<<grid, block>>>(d_r_in, d_g_in, d_b_in, d_r_out, d_g_out, d_b_out, width, height);
             
            else 
            {
                gpu_conv2d_kernel<<<grid, block>>>(d_r_in, d_filter, d_r_out, height, width, r);
                gpu_conv2d_kernel<<<grid, block>>>(d_g_in, d_filter, d_g_out, height, width, r);
                gpu_conv2d_kernel<<<grid, block>>>(d_b_in, d_filter, d_b_out, height, width, r);
            }
            cudaDeviceSynchronize();

            cudaMemcpy(h_r_out, d_r_out, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_g_out, d_g_out, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_b_out, d_b_out, size, cudaMemcpyDeviceToHost);

            for (int i = 0; i < width * height; ++i) {
                out_rgb[3 * i + 0] = (unsigned char)(fminf(fmaxf(h_r_out[i], 0.0f), 1.0f) * 255.0f);
                out_rgb[3 * i + 1] = (unsigned char)(fminf(fmaxf(h_g_out[i], 0.0f), 1.0f) * 255.0f);
                out_rgb[3 * i + 2] = (unsigned char)(fminf(fmaxf(h_b_out[i], 0.0f), 1.0f) * 255.0f);
            }

            draw_rgb(display, win, gc, visual, depth, img_data, width, height, 0);
            draw_rgb(display, win, gc, visual, depth, out_rgb, width, height, width);
            
            XFlush(display);
            free(h_filter);
            cudaFree(d_filter);

        }

        usleep(100000);
    }

    stbi_image_free(img_data);
    free(h_r); free(h_g); free(h_b);
    free(h_r_out); free(h_g_out); free(h_b_out);
    free(out_rgb);
    
    cudaFree(d_r_in); cudaFree(d_g_in); cudaFree(d_b_in);
    cudaFree(d_r_out); cudaFree(d_g_out); cudaFree(d_b_out);
    
    XDestroyWindow(display, win);
    XCloseDisplay(display);

    return 0;
}
