# CUDA_FILTERS⚡  

GPU-accelerated image filtering project using CUDA on WSL2. This project implements both **convolutional** and **non-convolutional** filters to enhance and transform images efficiently using parallel processing.

---

**Convolutional Filters** (Matrix-based transformations)  
   - Gaussian Blur  
   - Edge Detection (Sobel, Laplacian)   
   - Emboss  

**Non-Convolutional Filters** (Pixel-wise transformations)  
   - Grayscale and Sepia Conversions  
   - Brightness & Contrast Adjustments (Orton)  
   - Inversion (Negative)  

![optimize](https://github.com/user-attachments/assets/8bb1d296-bf2e-42c8-b3bd-fe379700c475)


## 🚀 Setting Up CUDA on WSL  

You need CUDA installed on **WSL 2 (Ubuntu)** with a compatible NVIDIA GPU.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run
```

`Configure Environment Variables`
```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

Instead of relying on heavy GUI frameworks, this project uses the **X11** Graphics API to render processed images directly onto the screen.
```bash
nvcc Filters.cu -lX11 -o filter_cli
```
---
### 🎥 Live Webcam Filtering with CUDA on WSL

Real-time CUDA filters directly to webcam input inside WSL2 (Ubuntu).  
It uses low-level **V4L2** API for interfacing with `/dev/video0`, **ioctl()** system calls for device control

Connect USB Camera to WSL
https://learn.microsoft.com/en-us/windows/wsl/connect-usb 

```powershell
usbipd list
usbipd attach --wsl --busid <your-busid>
```

Make sure to **Re-compile** the WSL kernel with the required camera drivers for `/dev/video0` to show up

```bash
nvcc LiveFilter.cu -lX11 -o camera
```
