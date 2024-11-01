#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

// Optimizations: #2, #4

#define Map_out_L1 4
#define Channel_L1 1
#define Map_out_L2 16
#define Channel_L2 4
#define MASK_WIDTH 7
#define TILE_WIDTH 16

static __constant__ float Mc1[Map_out_L1][Channel_L1*MASK_WIDTH*MASK_WIDTH];
static __constant__ float Mc2[Map_out_L2][Channel_L2*MASK_WIDTH*MASK_WIDTH];

__global__ void unroll_kernel(float *output, const float *input, const int Batch,
                              const int Channel, const int Height, const int Width,
                              const int K, const int start)
{
  const int Height_out = Height - K + 1;
  const int Width_out = Width - K + 1;

  int w = (blockIdx.x * blockDim.x) + threadIdx.x;

  #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
  #define unroll_3d(i2, i1, i0) output[(i2) * (Channel * K * K * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]

  if (w < Batch * Height_out * Width_out) {
    int b = w / (Height_out * Width_out);
    w -= (b * Height_out * Width_out);
    int h = w / Width_out;
    w -= (h * Width_out);

    for (int c = 0; c < Channel; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          int col = (h * Width_out) + w;
          int row = (c * K * K) + (p * K) + q;
          unroll_3d(b, row, col) = in_4d(start+b, c, h + p, w + q);
        }
      }
    }
  }

  #undef in_4d
  #undef unroll_3d
}

__global__ void matrixMultiplyShared(float *output, const float *input,
                                     const int Batch, const int Map_out,
                                     const int Channel, const int Height,
                                     const int Width, const int K, const int start)
{
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int b = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0.0;

  const int inputSize = Channel*K*K;
  const int Height_out = Height - K + 1;
  const int Width_out = Width - K + 1;
  const int outputSize = Height_out * Width_out;

  #define input_3d(i2, i1, i0) input[(i2) * (Channel * K * K * Height_out * Width_out) + (i1) * (Height_out * Width_out) + i0]
  #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]

  for (int m = 0; m < ceil((1.0*inputSize)/TILE_WIDTH); ++m) {
    int subCol = (m * TILE_WIDTH) + tx;
    int subRow = (m * TILE_WIDTH) + ty;

    if (Row < Map_out && subCol < inputSize) {
      subTileA[ty][tx] = (Channel > 1) ? Mc2[Row][subCol] : Mc1[Row][subCol];
    }
    else {subTileA[ty][tx] = 0.0;}

    if (subRow < inputSize && Col < outputSize) {
      subTileB[ty][tx] = input_3d(b, subRow, Col);
    }
    else {subTileB[ty][tx] = 0.0;}

    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }

  if (Row < Map_out && Col < outputSize) {
    int h = Col / Width_out;
    int w = Col % Width_out;
    out_4d(start+b, Row, h, w) = Pvalue;
  }

  #undef input_3d
  #undef out_4d
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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputsize = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void **) device_output_ptr, outputsize);

    int inputsize = Batch * Channel * Height * Width * sizeof(float);
    cudaMalloc((void **) device_input_ptr, inputsize);
    cudaMemcpy(*device_input_ptr, host_input, inputsize, cudaMemcpyHostToDevice);

    // Copy Layer 1 or 2's kernel to constant memory
    int masksize = Map_out * Channel * K * K * sizeof(float);
    if (Channel > 1) {
      cudaMemcpyToSymbol(Mc2, host_mask, masksize);
    }
    else {
      cudaMemcpyToSymbol(Mc1, host_mask, masksize);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Allocate memory space for the unrolled matrix
    float *unroll = NULL;
    int unroll_size = 1000 * Channel * K * K * Height_out * Width_out * sizeof(float);
    cudaMalloc((void **)&unroll, unroll_size);

    for (int i = 0; i < ceil(Batch/1000.0); i++) {
      int start = i * 1000;
      int mini_b = min(1000, Batch - start);

      // Initialize grid and block dimensions here
      int numPixels = mini_b * Height_out * Width_out;
      unroll_kernel<<<ceil(numPixels/96.0),96>>>(unroll, device_input, Batch, Channel, Height, Width, K, start);
      cudaDeviceSynchronize();

      dim3 DimGrid(ceil(Height_out * Width_out * 1.0 / TILE_WIDTH),
                   ceil(Map_out * 1.0 / TILE_WIDTH),
                   mini_b);
      dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
      matrixMultiplyShared<<<DimGrid,DimBlock>>>(device_output, unroll, Batch,
                                                 Map_out, Channel, Height, Width,
                                                 K, start);
      cudaDeviceSynchronize();
    }

    cudaFree(unroll);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
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
