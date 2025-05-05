#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>

#define BLOCK_SIZE 16  // using 16*16 arch

// The cuda kernel
__global__ void conv2d_kernel(float* input,float* filter, float* output, int H, int W, int R) {
//H = Height, W = Width , R = Filter size

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int P = (R-1) / 2; // Zero-padding size
  float sum = 0.0f; // setting the value to 0 for sum
  
  if (row < H && col < W) {
      for (int i = 0; i < R; i++) {
          for (int j = 0; j < R; j++) {
              int imgX = col + j - P;
              int imgY = row + i - P;
              
              if (imgX >= 0 && imgX < W && imgY >= 0 && imgY < H) {
                  sum += input[imgY * W + imgX] * filter[i * R + j];
              }
          }
      }
      output[row * W + col] = sum;
  }
}


// Function to read input values and allocate host memory
void readInputFile(const char* filename, float** data, int* H, int* W) {
  FILE* file = fopen(filename, "r");
  //first and second row reads Height and width
  fscanf(file, "%d %d", H, W);
   //allocate memory for the H*W size in host and read data values from input.txt
  *data = (float*)malloc((*H) * (*W) * sizeof(float));
  for (int i = 0; i < (*H) * (*W); i++) {
      fscanf(file, "%f", &((*data)[i]));
  }
  fclose(file);
}

// Function to read filter values and allocate host memory
void readFilterFile(const char* filename, float** filter, int* R) {
  FILE* file = fopen(filename, "r");
  //first row reads the filter size for R * R
  fscanf(file, "%d", R);
    //Allocate memory for R*R in host and read the filter values
  *filter = (float*)malloc((*R) * (*R) * sizeof(float));
  for (int i = 0; i < (*R) * (*R); i++) {
      fscanf(file, "%f", &((*filter)[i]));
  }
  fclose(file);
}



int main(int argc, char *argv[]) {
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess; 
    
  // Read the inputs from command line
  if (argc != 3) 
  {
    printf("Usage: %s <input.txt> <filter.txt>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  float *h_input, *h_filter, *h_output;
  float *d_input, *d_filter, *d_output;
  int H, W, R;

  //reads input and filters values
  readInputFile(argv[1], &h_input, &H, &W);
  readFilterFile(argv[2], &h_filter, &R);

  //allocate memory in the host
  h_output = (float*) malloc(H * W * sizeof(float));

  if (!h_input || !h_filter || !h_output) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  // Allocate/move data using cudaMallocManaged

    // Allocate device memory

    cudaMallocManaged(&d_input, H * W * sizeof(float));
    cudaMallocManaged(&d_filter, R * R * sizeof(float));
    cudaMallocManaged(&d_output, H * W * sizeof(float));


    // Create CUDA events...
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
      // Copy data to device
      err = cudaMemcpy(d_input, h_input, H * W * sizeof(float), cudaMemcpyHostToDevice);
      
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

      err = cudaMemcpy(d_filter, h_filter, R * R * sizeof(float), cudaMemcpyHostToDevice);

      if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to copy filter image from host to device (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }
  

  // Launch the kernel
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Launch the kernel and take timestamps before and after
  cudaEventRecord(start);
  conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, H, W, R);
  cudaEventRecord(stop);

  err = cudaGetLastError();


  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch conv2d kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
 


   // Copy result back to host
  err =  cudaMemcpy(h_output, d_output, H * W * sizeof(float), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy output image from device to host (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  // Extract the timing information
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Print the output
    // Print output
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
          printf("%.3f \n", h_output[i * W + j]);
      }
  }

  //Kernel execution time
  // std::cout<< "GPU kernel took: " <<milliseconds << " milliseconds to execute\n";
  // Clean up the memory
    // Free memory
    free(h_input); free(h_filter); free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input); cudaFree(d_filter); cudaFree(d_output);

  return 0;
}