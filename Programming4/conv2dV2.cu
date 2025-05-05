#include <iostream>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

#define BLOCK_SIZE 16

__global__ void conv2d_kernel(float *input, float *filter, float *output, int H, int W, int N, int K) {
    extern __shared__ float tile[];

    int Hout = H - 2;
    int Wout = W - 2;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int w_out = tx + blockIdx.x * blockDim.x;
    int h_out = ty + blockIdx.y * blockDim.y;

    int n = blockIdx.z / K;
    int k = blockIdx.z % K;

    int shared_W = BLOCK_SIZE + 2;

    // Load input tile into shared memory
    for (int i = ty; i < BLOCK_SIZE + 2; i += blockDim.y) {
        for (int j = tx; j < BLOCK_SIZE + 2; j += blockDim.x) {
            int row = blockIdx.y * BLOCK_SIZE + i;
            int col = blockIdx.x * BLOCK_SIZE + j;

            if (row < H && col < W) {
                tile[i * shared_W + j] = input[n * H * W + row * W + col];
            } else {
                tile[i * shared_W + j] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (h_out < Hout && w_out < Wout && n < N && k < K) {
        float sum = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sum += tile[(ty + i) * shared_W + (tx + j)] *
                       filter[k * 9 + i * 3 + j];
            }
        }

        int out_index = n * K * Hout * Wout + k * Hout * Wout + h_out * Wout + w_out;
        output[out_index] = sum;
    }
}


int main(int argc, char *argv[])
{
// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess; 
// Read the inputs from command line
    if (argc != 3)
    {
        printf("Usage: %s <input.txt> <filter.txt>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
// Open the input and filter files
    FILE *fp_input = fopen(argv[1], "r");
    FILE *fp_filter = fopen(argv[2], "r");

    if (!fp_input || !fp_filter)
    {
        std::cerr << "Error: Unable to open input or filter file.\n";
        return EXIT_FAILURE;
    }
//Info about the input and filter files
    int inputHeight, inputWidth, N;
    int filterHeight, K;
//First 3 values of input Input Image Height, input Image Height and Number of input images:
    fscanf(fp_input, "%d%d%d", &inputHeight, &inputWidth, &N);
    //First 2 values of filter Filter Height(same as width ) and  Number of filters:
    fscanf(fp_filter, "%d%d", &filterHeight, &K);

    //Total size of Input images would be number of images * height * width
    int inputSize = N * inputHeight * inputWidth;
    //Total size of filter would be number of filter * height * width
    int filterSize = K * filterHeight * filterHeight;
    
    //Padding size is 1 for 3x3 filter
    int padding = 1;
    int paddedHeight = inputHeight + 2 * padding;
    int paddedWidth = inputWidth + 2 * padding;
    int paddedSize = N * paddedHeight * paddedWidth;

    float *d_input, *d_filter, *d_output;
    int outputHeight = paddedHeight - 2;
    int outputWidth = paddedWidth - 2;
      int outputSize = N * K * outputHeight * outputWidth;

    float *inputImage = (float *)malloc(inputSize * sizeof(float));
    float *filterValues = (float *)malloc(filterSize * sizeof(float));
    float *h_output = (float *)malloc(outputSize * sizeof(float));

    for (int i = 0; i < inputSize; i++)
        fscanf(fp_input, "%f", &inputImage[i]);

    for (int i = 0; i < filterSize; i++)
        fscanf(fp_filter, "%f", &filterValues[i]);

    fclose(fp_input);
    fclose(fp_filter);



    float *paddedImage = (float *)malloc(paddedSize * sizeof(float));

     //Check if allocation was correctly done
    if (!inputImage || !filterValues || !h_output || !paddedImage) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
    } 

    for (int n = 0; n < N; n++)
    {
        for (int i = 0; i < inputHeight; i++)
        {
            for (int j = 0; j < inputWidth; j++)
            {
                paddedImage[n * paddedHeight * paddedWidth + (i + padding) * paddedWidth + (j + padding)] =
                    inputImage[n * inputHeight * inputWidth + i * inputWidth + j];
            }
        }
    }


    cudaMalloc(&d_input, paddedSize * sizeof(float));
    cudaMalloc(&d_filter, filterSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));

      // Copy data to device
    err = cudaMemcpy(d_input, paddedImage, paddedSize * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_filter, filterValues, filterSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      {
          fprintf(stderr, "Failed to copy filter image from host to device (error code %s)!\n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
      }


    int Wout = paddedWidth - 2;
    int Hout = paddedHeight - 2;

      // Launch the kernel
    // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 gridDim((Wout + BLOCK_SIZE-1) / BLOCK_SIZE, (Hout + BLOCK_SIZE-1) / BLOCK_SIZE, N * K);

    int sharedMemSize = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(float);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((Wout + BLOCK_SIZE - 1) / BLOCK_SIZE, (Hout + BLOCK_SIZE - 1) / BLOCK_SIZE, N * K);
    
    conv2d_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_filter, d_output, paddedHeight, paddedWidth, N, K);

    // conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, paddedHeight, paddedWidth, N, K);

        err = cudaGetLastError();
        if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch conv2d kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
 
      // Copy result back to host
    err = cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy output image from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Print the output
    for (int n = 0; n < N; n++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int i = 0; i < outputHeight; i++)
            {
                for (int j = 0; j < outputWidth; j++)
                {
                    int idx = n * K * outputHeight * outputWidth +
                              k * outputHeight * outputWidth +
                              i * outputWidth + j;
                    std::cout << std::fixed << std::setprecision(3) << h_output[idx] << " ";
                }
                std::cout << "\n";
            }
        }
    }

    // Cleanup
    free(inputImage);
    free(filterValues);
    free(paddedImage);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
