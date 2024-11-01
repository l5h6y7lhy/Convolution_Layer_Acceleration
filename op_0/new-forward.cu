#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Optimizations: #0, #4

// Define constant memory for device kernel
#define Map_out_L1 4
#define Channel_L1 1
#define Map_out_L2 16
#define Channel_L2 4
#define MASK_WIDTH 7
#define SEG_SIZE 60
static __constant__ float Mc1[Map_out_L1][Channel_L1][MASK_WIDTH][MASK_WIDTH];
static __constant__ float Mc2[Map_out_L2][Channel_L2][MASK_WIDTH][MASK_WIDTH];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // Insert your GPU convolution kernel code here
    int w = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (w < Batch * Map_out * Height_out * Width_out) {
      int b = w / (Map_out * Height_out * Width_out);
      w -= (b * Map_out * Height_out * Width_out);
      int m = w / (Height_out * Width_out);
      w -= (m * Height_out * Width_out);
      int h = w / Width_out;
      w -= (h * Width_out);

      float r = 0.0;

      for (int c = 0; c < Channel; c++) {
        for (int p = 0; p < K; p++) {
          for (int q = 0; q < K; q++) {
            float k = (Channel > 1) ? Mc2[m][c][p][q] : Mc1[m][c][p][q];
            r += (in_4d(b, c, h+p, w+q) * k);
          }
        }
      }

      out_4d(b, m, h, w) = r;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


// Calculate the number of remaining images after c epoches
__host__ int r(int c, const int Batch)
{
  return min(SEG_SIZE, Batch - (c * SEG_SIZE));
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // Copy Layer 1 or 2's kernel to constant memory
    int masksize = Map_out * Channel * K * K * sizeof(float);
    if (Channel > 1) {
      cudaMemcpyToSymbol(Mc2, host_mask, masksize);
    }
    else {
      cudaMemcpyToSymbol(Mc1, host_mask, masksize);
    }

    // Commonly used variables and macros
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int c = 0;
    int total = ceil(Batch*1.0/SEG_SIZE);

    #define input_size(i) i * Channel * Height * Width
    #define output_size(i) i * Map_out * Height_out * Width_out

    // Start the three streams
    cudaStream_t trans_in, compute, trans_out, tmp;
    cudaStreamCreate(&trans_in);
    cudaStreamCreate(&compute);
    cudaStreamCreate(&trans_out);

    // Set up the input and output memory for the three streams
    float *input_ti;
    float *output_ti;
    float *input_c;
    float *output_c;
    float *input_to;
    float *output_to;
    float *dtmp;

    cudaMalloc((void **)&input_ti, input_size(SEG_SIZE)*sizeof(float));
    cudaMalloc((void **)&output_ti, output_size(SEG_SIZE)*sizeof(float));
    cudaMalloc((void **)&input_c, input_size(SEG_SIZE)*sizeof(float));
    cudaMalloc((void **)&output_c, output_size(SEG_SIZE)*sizeof(float));
    cudaMalloc((void **)&input_to, input_size(SEG_SIZE)*sizeof(float));
    cudaMalloc((void **)&output_to, output_size(SEG_SIZE)*sizeof(float));

    if (total <= 1 ) {
      // When the dataset is small
      cudaMemcpyAsync(input_to, &host_input[0], input_size(Batch)*sizeof(float), cudaMemcpyHostToDevice, trans_out);
      conv_forward_kernel<<<ceil(output_size(Batch)/96.0),96,0,trans_out>>>(output_to, input_to,
                                                                            NULL, Batch,
                                                                            Map_out, Channel, Height,
                                                                            Width, K);
      cudaMemcpyAsync(((void*) &host_output[0]), output_to, output_size(Batch)*sizeof(float), cudaMemcpyDeviceToHost, trans_out);
    }
    else {
      // Start the first two streams
      cudaMemcpyAsync(input_to, &host_input[0], input_size(SEG_SIZE)*sizeof(float), cudaMemcpyHostToDevice, trans_out);
      c++;
      cudaMemcpyAsync(input_c, &host_input[input_size(c*SEG_SIZE)], input_size(r(c, Batch)) * sizeof(float),
                      cudaMemcpyHostToDevice, compute);
      conv_forward_kernel<<<ceil(output_size(SEG_SIZE)/96.0),96,0,trans_out>>>(output_to, input_to,
                                                                               NULL, SEG_SIZE,
                                                                               Map_out, Channel, Height,
                                                                               Width, K);
      c++;

      while (c < total) {
        // One stream transfers its data into GPU
        // One stream does its computation work
        // One stream transfers its data out of GPU
        cudaMemcpyAsync(input_ti, &host_input[input_size(c*SEG_SIZE)], input_size(r(c, Batch)) * sizeof(float),
                        cudaMemcpyHostToDevice, trans_in);
        conv_forward_kernel<<<ceil(output_size(SEG_SIZE)/96.0),96,0,compute>>>(output_c, input_c,
                                                                               NULL, SEG_SIZE,
                                                                               Map_out, Channel, Height,
                                                                               Width, K);
        cudaMemcpyAsync(((void*) &host_output[output_size((c-2)*SEG_SIZE)]), output_to, output_size(SEG_SIZE)*sizeof(float),
                        cudaMemcpyDeviceToHost, trans_out);

        // Swap the three streams and their memory pointers
        tmp = compute;
        compute = trans_in;
        trans_in = trans_out;
        trans_out = tmp;

        dtmp = input_c;
        input_c = input_ti;
        input_ti = input_to;
        input_to = dtmp;

        dtmp = output_c;
        output_c = output_ti;
        output_ti = output_to;
        output_to = dtmp;

        c++;
      }

      // Wrap up the computation and transferring work for two streams
      conv_forward_kernel<<<ceil(output_size(r(c-1,Batch))/96.0),96,0,compute>>>(output_c, input_c,
                                                                                 NULL, r(c-1, Batch),
                                                                                 Map_out, Channel, Height,
                                                                                 Width, K);
      cudaMemcpyAsync(((void*) &host_output[output_size((c-2)*SEG_SIZE)]), output_to, output_size(SEG_SIZE)*sizeof(float), cudaMemcpyDeviceToHost, trans_out);
      cudaMemcpyAsync(((void*) &host_output[output_size((c-1)*SEG_SIZE)]), output_c, output_size(r(c-1, Batch)) * sizeof(float),
                      cudaMemcpyDeviceToHost, compute);
    }

    // Make sure all of the three streams have finished all their work in the queue
    cudaStreamSynchronize(trans_out);
    cudaStreamSynchronize(compute);
    cudaStreamSynchronize(trans_in);

    cudaStreamDestroy(trans_out);
    cudaStreamDestroy(compute);
    cudaStreamDestroy(trans_in);

    // Free memory space
    cudaFree(input_ti);
    cudaFree(output_ti);
    cudaFree(input_c);
    cudaFree(output_c);
    cudaFree(input_to);
    cudaFree(output_to);

    #undef input_size
    #undef output_size
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    return;
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
