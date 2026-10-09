/*
 * Copyright (c) 2026 University of Arkansas.
 * Department of Computer Science and Computer Engineering.
 * All rights reserved.
 *
 * Course: GPU Programming
 * Instructor: Abhi Rajagopala (abhi@uark.edu)
 *
 * Permission to use, copy, modify, and distribute this software for 
 * educational and academic purposes without fee is hereby granted, 
 * provided that this copyright notice appears in all copies.
 */
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
// The call is evaluated once and its status is stored, so it is safe to wrap
// calls such as cudaMallocManaged(...) with it.
#define cudaCheckError(call) { \
    cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status_) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

/* Image layout (both kernels)
 * - Images are stored row-major, one pixel after another.
 * - RGBA image : 4 bytes per pixel in the order R, G, B, A (channel stride = 4)
 * - Grayscale  : 1 byte per pixel
 * - Each thread processes exactly one pixel.
 */



/* TO DO: Complete the RGBA to Grayscale kernel (runs on GPU, launched from CPU)
 * - Add the kernel definition (input RGBA image, output grayscale image, width, height)
 * - Calculate the global x (column) and y (row) indices
 * - Add a boundary check that covers both dimensions
 * - Calculate the input pixel index and multiply by the channel stride
 *   (4 bytes per pixel for RGBA) to get the byte offset of the pixel
 * - Access the R, G, and B values (ignore A)
 * - Calculate gray = 0.299*R + 0.587*G + 0.114*B
 * - Calculate the output index (1 byte per pixel, so no channel stride)
 * - Write the output as an unsigned char
*/
__global__
void colorToGrey(const char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  //calculate x and y 
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int inputIndex = (y * width + x) * 4; //row major order 4 bytes per pixel

        unsigned char r = input[inputIndex];
        unsigned char g = input[inputIndex + 1];
        unsigned char b = input[inputIndex + 2];

        float gray = 0.299f * r + 0.587f * g + 0.114f * b;

        int outputIndex = y * width + x;   // greyscale has only one byte per pixel so no 

        output[outputIndex] = static_cast<unsigned char>(gray);   
    }
}


/* TO DO: Complete the palette function (runs on GPU, called from the Ironbow kernel)
 * - This is NOT a kernel. It is a device function called by another GPU function,
 *   so use the appropriate qualifier.
 * - Input : gray value (unsigned char, 0-255)
 * - Output: r, g, b, a (unsigned char). Return them through output parameters
 *   (pointers or references), since a function can return only one value.
 * - Calculate r, g, b as follows:
 *     gray  0 -  63 : r = 0,                 g = 0,                 b = gray * 4
 *     gray 64 - 127 : r = (gray - 64) * 4,   g = 0,                 b = 255 - (gray - 64) * 4
 *     gray 128 - 191: r = 255,               g = (gray - 128) * 4,  b = 0
 *     gray 192 - 255: r = 255,               g = 255,               b = (gray - 192) * 4
 * - Calculate a: 0 if gray < 15, otherwise 220
 * - Note: intermediate expressions are promoted to int. Make sure results stay
 *   within 0-255 before storing them in unsigned char.
*/

__device__
void palette(unsigned char gray, unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a)
{
    if (gray < 64)
    {
        r = 0;
        g = 0;
        b = gray * 4;
    }
    else if (gray < 128)
    {
        r = (gray - 64) * 4;
        g = 0;
        b = 255 - (gray - 64) * 4;
    }
    else if (gray < 192)
    {
        r = 255;
        g = (gray - 128) * 4;
        b = 0;
    }
    else
    {
        r = 255;
        g = 255;
        b = (gray - 192) * 4;
    }

    a = (gray < 15) ? 0 : 220;
}


/* TO DO: Complete the Ironbow kernel (runs on GPU, launched from CPU)
 * - Add the kernel definition (input grayscale image, input background RGBA image,
 *   output ironbow RGBA image, width, height)
 * - Calculate the global x (column) and y (row) indices
 * - Add a boundary check that covers both dimensions
 * - Calculate the pixel index (1 byte per pixel in the grayscale image)
 * - Access the input grayscale value
 * - Call the palette function with the grayscale value to get r, g, b, a
 * - Calculate the output byte offset (pixel index * channel stride of 4)
 * - Undergraduate: write r, g, b, a to the output
 *
 * Graduate only - Alpha blending with the background (Porter-Duff "over")
 * - Foreground = the palette output (r, g, b, a) computed above.
 *   Background = the pixel at the same position in the background RGBA image.
 * - Convert both alpha values to the range 0.0-1.0 (divide by 255) before
 *   blending. Leave r, g, b in the range 0-255.
 * - Output alpha:
 *     out_a = Foreground_a + Background_a * (1 - Foreground_a)
 * - If out_a is 0: output r, g, b, and a are all 0.
 * - Otherwise:
 *     out_r = (Foreground_r * Foreground_a
 *              + Background_r * Background_a * (1 - Foreground_a)) / out_a
 *     Repeat the same calculation for g and b.
 * - Convert out_a back to the range 0-255 (multiply by 255). Clamp all
 *   outputs to 0-255 and round to the nearest integer before storing them
 *   as unsigned char.
 * - Write the blended r, g, b, a to the output.
*/


__global__
void ironBow( const unsigned char* grayscale, const unsigned char* background, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int pixelIndex = y * width + x;

        unsigned char gray = grayscale[pixelIndex];

        unsigned char r, g, b, a;
        palette(gray, r, g, b, a);

        int outputIndex = pixelIndex * 4;

        output[outputIndex]     = r;
        output[outputIndex + 1] = g;
        output[outputIndex + 2] = b;
        output[outputIndex + 3] = a;
    }
}

// Host function
int main(){

	//Image dimensions (NOTE: These are matched to sample input)
	const unsigned int width = 1920;
	const unsigned int height = 1080;

	/*TO DO: Calculate the memory size in bytes for the two image types.
	 * Use the variable names below, since they are used later in the file
	 * read and write code:
	 *   rgba_bytes : size of one RGBA image (4 bytes per pixel)
	 *   gray_bytes : size of one grayscale image (1 byte per pixel)
	*/

    size_t rgba_bytes = width * height * 4;
    size_t gray_bytes = width * height;

	//Memory pointers for input and output images
    unsigned char *foregnd, *backgnd, *grayscale, *ironbow;

    /*TO DO: Allocate Unified memory for
	 * Input RGBA foreground image  (rgba_bytes)
	 * Input RGBA background image  (rgba_bytes)
	 * Output Grayscale image       (gray_bytes)
	 * Output Ironbow RGBA image    (rgba_bytes)
	 * Use the cudaCheckError macro to check each allocation.
	*/

    cudaCheckError(cudaMallocManaged(&foregnd, rgba_bytes));
    cudaCheckError(cudaMallocManaged(&backgnd, rgba_bytes));
    cudaCheckError(cudaMallocManaged(&grayscale, gray_bytes));
    cudaCheckError(cudaMallocManaged(&ironbow, rgba_bytes));


    //Read foreground images
	std::ifstream foreImg("foreground.raw", std::ios::binary);
    if (!foreImg) {
        std::cerr << "Error: Could not open foreground.raw!" << std::endl;
        cudaFree(foregnd); cudaFree(backgnd); cudaFree(grayscale); cudaFree(ironbow);
        return -1;
    }
    foreImg.read(reinterpret_cast<char*>(foregnd), rgba_bytes);
    foreImg.close();

    //Read background images
	std::ifstream backImg("background.raw", std::ios::binary);
    if (!backImg) {
        std::cerr << "Error: Could not open background.raw!" << std::endl;
        cudaFree(foregnd); cudaFree(backgnd); cudaFree(grayscale); cudaFree(ironbow);
        return -1;
    }
    backImg.read(reinterpret_cast<char*>(backgnd), rgba_bytes);
    backImg.close();

    /* TO DO: Define 2D Block and Grid dimensions
     * - Each thread processes one pixel (NOTE: one pixel, not one byte/value)
     * - Choose a 2D block size (threads per block must not exceed 1024)
     * - Round the number of blocks up in each dimension so every pixel is covered.
     *   Note that 1080 is not evenly divisible by common block sizes such as 16.
    */

    dim3 blockSize(16, 16);

    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);



	// Declare a variable to hold CUDA error status for the checks below
	cudaError_t err;

	/* TO DO: Launch the RGBA to Grayscale kernel and wait for the GPU to finish
	 * Arguments: Foreground image, Grayscale image, width, height
	 * Check for a launch error immediately after the launch (cudaGetLastError),
	 * then wait with cudaDeviceSynchronize and check its return value as well.
	*/

    colorToGrey<<<gridSize, blockSize>>>(foregnd, grayscale,width,height);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());


	/* TO DO: Launch the Ironbow kernel and wait for the GPU to finish
	 * Arguments: Grayscale image, Foreground image, Ironbow image, width, height
	 * Check for a launch error immediately after the launch (cudaGetLastError),
	 * then wait with cudaDeviceSynchronize and check its return value as well.
	*/

    ironBow<<<gridSize, blockSize>>>(grayscale,backgnd,ironbow,width,height);

    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());


    // Final check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Image processing kernels executed successfully!" << std::endl;
    }

    //Save output images
    std::ofstream grayImg("gray.raw", std::ios::binary);
    if (grayImg) {
        grayImg.write(reinterpret_cast<char*>(grayscale), gray_bytes);
        grayImg.close();
        std::cout << "Saved output to gray.raw" << std::endl;
    }

    std::ofstream ironImg("ironbow.raw", std::ios::binary);
    if (ironImg) {
        ironImg.write(reinterpret_cast<char*>(ironbow), rgba_bytes);
        ironImg.close();
        std::cout << "Saved output to ironbow.raw" << std::endl;
    }

    //Cleanup Unified Memory
    cudaFree(backgnd);
    cudaFree(foregnd);
    cudaFree(grayscale);
    cudaFree(ironbow);

	return 0;
}
