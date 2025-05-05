
#include <iostream>
#include <cuda.h>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

#define BLOCK_SIZE 16  // using 16*16 arch

// The cuda kernel
__global__ void conv2d_kernel(float *input, float *filter, float *output, int H, int W, int N, int K) {
    int Hout = H - 2;
    int Wout = W - 2;

    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int k = blockIdx.z % K;
    int n = blockIdx.z / K;

    if (n < N && k < K && h < Hout && w < Wout) {
        float sum = 0.0f; // setting the value to 0 for sum
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                sum += input[n * H * W + (h + 1 + i) * W + (w + 1 + j)] *
                       filter[k * 3 * 3 + (i + 1) * 3 + (j + 1)];
            }
        }
        int out_index = n * K * Hout * Wout + k * Hout * Wout + h * Wout + w;
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
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((Wout + BLOCK_SIZE-1) / BLOCK_SIZE, (Hout + BLOCK_SIZE-1) / BLOCK_SIZE, N * K);

    conv2d_kernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, paddedHeight, paddedWidth, N, K);

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
