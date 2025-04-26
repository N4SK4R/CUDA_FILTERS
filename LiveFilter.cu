#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.cuh"
#include "headers/gpu_vid_kernels.cuh"

#define WIDTH 640
#define HEIGHT 480
#define DEVICE "/dev/video0"

struct buffer {
    void   *start;
    size_t length;
};

int main() {
    int fd = open(DEVICE, O_RDWR);
    if (fd < 0) { perror("open"); return 1; }

    // Set video format
    struct v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    ioctl(fd, VIDIOC_S_FMT, &fmt);

    // 60 FPS
    struct v4l2_streamparm parm = {0};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = 60; 
    if (ioctl(fd, VIDIOC_S_PARM, &parm) < 0) perror("VIDIOC_S_PARM");
    
    // Request buffer
    struct v4l2_requestbuffers req = {0};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(fd, VIDIOC_REQBUFS, &req);

    // Map buffer
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;
    ioctl(fd, VIDIOC_QUERYBUF, &buf);

    struct buffer buffer;
    buffer.length = buf.length;
    buffer.start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

    ioctl(fd, VIDIOC_QBUF, &buf);
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(fd, VIDIOC_STREAMON, &type);

    // Setup X11
    Display *dpy = XOpenDisplay(NULL);
    Window win = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, WIDTH, HEIGHT, 0, 0, 0);
    XMapWindow(dpy, win);
    XFlush(dpy);
    GC gc = XCreateGC(dpy, win, 0, NULL);

    int threads = 256;
    int blocks = (WIDTH * HEIGHT + threads - 1) / threads;

    unsigned char *d_rgb;
    float *d_gray;

    cudaMalloc(&d_rgb, WIDTH * HEIGHT * 3);
    cudaMalloc(&d_gray, WIDTH * HEIGHT * sizeof(float));

    float *d_sobel_x, *d_sobel_y;
    cudaMalloc(&d_sobel_x, 9 * sizeof(float));
    cudaMalloc(&d_sobel_y, 9 * sizeof(float));
    cudaMemcpy(d_sobel_x, sobel_x, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sobel_y, sobel_y, 9 * sizeof(float), cudaMemcpyHostToDevice);

    float *d_sobel_x_out, *d_sobel_y_out;
    cudaMalloc(&d_sobel_x_out, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc(&d_sobel_y_out, WIDTH * HEIGHT * sizeof(float));

    float *sobel_x_host = (float *)malloc(WIDTH * HEIGHT * sizeof(float));
    float *sobel_y_host = (float *)malloc(WIDTH * HEIGHT * sizeof(float));

    XImage *img = XCreateImage(dpy, DefaultVisual(dpy, 0), 24, ZPixmap, 0,
                               (char*)malloc(WIDTH * HEIGHT * sizeof(int)),
                               WIDTH, HEIGHT, 32, 0);
    
    while (1) {
        
        ioctl(fd, VIDIOC_DQBUF, &buf);
    
        int w, h, channels;
        unsigned char *rgb = stbi_load_from_memory((unsigned char*)buffer.start, buf.bytesused, &w, &h, &channels, 3);
        if (rgb && w == WIDTH && h == HEIGHT) {

            cudaMemcpy(d_rgb, rgb, WIDTH * HEIGHT * 3, cudaMemcpyHostToDevice);
    
            rgb_to_grayscale_kernel<<<blocks, threads>>>(d_rgb, d_gray, WIDTH, HEIGHT);
            cudaDeviceSynchronize();

            dim3 block(16, 16);
            dim3 grid(ceil((WIDTH)/(float)16),ceil((HEIGHT)/(float)16));

            gpu_conv2d_kernel<<<grid, block>>>(d_gray, d_sobel_x, d_sobel_x_out, HEIGHT, WIDTH, 1);
            gpu_conv2d_kernel<<<grid, block>>>(d_gray, d_sobel_y, d_sobel_y_out, HEIGHT, WIDTH, 1);
            cudaDeviceSynchronize();

            cudaMemcpy(sobel_x_host, d_sobel_x_out, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(sobel_y_host, d_sobel_y_out, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    
            for (int i = 0; i < WIDTH * HEIGHT; i++) {
                float gx = sobel_x_host[i];
                float gy = sobel_y_host[i];
                float magnitude = sqrtf(gx * gx + gy * gy);
                magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);
                unsigned char edge_val = (unsigned char)magnitude;
                ((unsigned int *)img->data)[i] = (edge_val << 16) | (edge_val << 8) | edge_val;
            }
    
            XPutImage(dpy, win, gc, img, 0, 0, 0, 0, WIDTH, HEIGHT);
            XFlush(dpy);
    
            stbi_image_free(rgb);
        }
    
        ioctl(fd, VIDIOC_QBUF, &buf);
    }
    
    // Cleanup
    cudaFree(d_rgb);
    cudaFree(d_gray);
    XDestroyImage(img);

}
