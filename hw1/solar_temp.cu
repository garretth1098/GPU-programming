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

//Required Library
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <stdlib.h>

// CUDA Kernel: Computes both elevation and azimuth for the full year
// The arguments includes: 
// Input: Total number of minutes, Latitude
// Output: Elevation, Azimuth

//TO DO: Create kernel definition
__global__
void calculateSolarGPU(int n, float latitude, float* elevation, float* azimuth)
{
	//TO DO: Calculate Index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) 
    {
        //convert lattitude to radians 
        float lat_rad = latitude * (float)M_PI / 180.0f;

	    //Index provides the Minutes.
	    float total_hours = (float)idx / 60.0f;
        
        // Day: Hours/24
        float day = total_hours / 24.0f;
        
        // Hour of the day: Total Hours % 24
        float hour = fmodf(total_hours, 24.0f);
        
        /******* Elevation Calculation *********/
        
        // Calculate Declination = 0.409 * sine(2 * PI * (Day - 80) / 365)
        float declination = 0.409f * sinf(2.0f * (float)M_PI * (day - 80.0f) / 365.0f);
        
        // Calculate Hour Angle = (Hour - 12) * 2 * PI / 24
        float hour_angle = (hour - 12.0f) * 2.0f * (float)M_PI / 24.0f;

        // Calculate Sine of Elevation = sine(latitude) * sine(Declination) + cosine(latitude) * cosine(Declination) * cosine(Hour Angle)
        float sin_elevation = sinf(lat_rad) * sinf(declination) + cosf(lat_rad) * cosf(declination) * cosf(hour_angle);
        
        // Calculate Elevation as Sine inverse (Clamped elevation -1, 1) * 180/PI
        float clamped_sin_el = sin_elevation;
        if (clamped_sin_el < -1.0f) clamped_sin_el = -1.0f;
        if (clamped_sin_el > 1.0f) clamped_sin_el = 1.0f;
        
        // Copy final elevation value to output
        elevation[idx] = asinf(clamped_sin_el) * 180.0f / (float)M_PI;
    }

        
        /******* Grad Students Only: ********/
        /*******Azimuth Calculation *********/
        //Calculate Azimuth
        // Equation of Time correction
        /*Because Earth's orbit is elliptical and tilted, 
        true solar time does not match standard clock time uniformly. 
        This discrepancy (the Equation of Time) shifts the sun's apparent 
        position by up to 16 minutes depending on the season.
        */
        //TO DO: Calculate Angle = 2 * PI * (Day - 81)/365
        
        //TO DO: Calculate Equation of Time minutes = 9.87 * sine(2*Angle) - 7.53 * cosine(Angle) - 1.5 * sine(Angle)
        
        //TO DO: Calculate Corrected Hour = Hour of the day + (Equation of Time minutes /60)
        
        //TO DO: Calculate Hour Angle = (Corrected Hour - 12) * 2 * PI/24

        //TO DO: Calculate Sine of Elevation = sine(latitude) * sine(Declination) + cosine(latitude) * cosine(Declination) * cosine(Hour Angle)
        
        
        //TO DO: Cosine Azimuth = sine(declination) - sine(latitude) * sine elevation/(cosine(latitude) * cosine(sine inverse (Clamped elevation -1, 1))
        
        //TO DO: Azimuth = Cosine inverse(cosine azimuth clamped -1, 1) * 180/PI
        //TO DO: If Hour Angle > 0 => Azimuth = 360 - Azimuth

		//TO DO: Copy final azimuth to output

    
}

//CPU Implementation of the Solar model
//The math is identical to the GPU kernel
// The arguments includes: 
// Input: Total number of minutes, Latitude
// Output: Elevation, Azimuth
	//Use a loop over Total number of minutes to calculate elevation (for all) and azimuth (Grad students)
void calculateSolarCPU(int n, float latitude, float* elevation, float* azimuth) 
{
    // Convert Latitude to radians
    float lat_rad = latitude * (float)M_PI / 180.0f;

    for (int i = 0; i < n; i++) {
        // Index provides the Minutes.
        // Total Hours: Minutes/60
        float total_hours = (float)i / 60.0f;
        
        // Day: Hours/24
        float day = total_hours / 24.0f;
        
        // Hour of the day: Total Hours % 24
        float hour = fmodf(total_hours, 24.0f);
        
        /******* Elevation Calculation *********/
        
        // Calculate Declination = 0.409 * sine(2 * PI * (Day - 80) / 365)
        float declination = 0.409f * sinf(2.0f * (float)M_PI * (day - 80.0f) / 365.0f);
        
        // Calculate Hour Angle = (Hour - 12) * 2 * PI / 24
        float hour_angle = (hour - 12.0f) * 2.0f * (float)M_PI / 24.0f;

        // Calculate Sine of Elevation = sine(latitude) * sine(Declination) + cosine(latitude) * cosine(Declination) * cosine(Hour Angle)
        float sin_elevation = sinf(lat_rad) * sinf(declination) + cosf(lat_rad) * cosf(declination) * cosf(hour_angle);
        
        // Calculate Elevation as Sine inverse (Clamped elevation -1, 1) * 180/PI
        float clamped_sin_el = sin_elevation;
        if (clamped_sin_el < -1.0f) clamped_sin_el = -1.0f;
        if (clamped_sin_el > 1.0f) clamped_sin_el = 1.0f;
        
        // Copy final elevation value to output
        elevation[i] = asinf(clamped_sin_el) * 180.0f / (float)M_PI;
    }
}

int main() {
	//Total minutes in a year
    int n = 525600;
    
	//Size in bytes for data type float
    size_t bytes = n * sizeof(float);
    
    //Latitude
    float latitude = 36.0f; 

    //TO DO: Allocate Unified Memory for elevation and azimuth
    // Name your pointers elevation and azimuth
    float* elevation;
    float* azimuth;

    cudaMallocManaged(&elevation, bytes);
    cudaMallocManaged(&azimuth, bytes);
    
    //TO DO: Create variable for block size and grid size
    const unsigned int numThreads = 256;
    const unsigned int numBlocks = (n + numThreads - 1) / numThreads;

    // Launch kernel
    
    calculateSolarGPU<<<numBlocks,numThreads>>>(n,latitude,elevation,azimuth);

    // Check for launch errors
    cudaGetLastError();

    // Synchronize to ensure GPU is done
    cudaDeviceSynchronize();

    // Print results 
    // Note: Minute 0 is midnight, so elevation should be negative (night), not 0.
    printf("Sample Results across the Year (Latitude: %.1f):\n", latitude);
    printf("Minute 0 (Jan 1, 00:00)  -> Elevation: %.2f°, Azimuth: %.2f°\n", elevation[0], azimuth[0]);
    
    // Check midday on day 180 (approx June 29, solar noon)
    int midday_idx = (180 * 1440) + 720;
    printf("Midday Summer (Day 180)  -> Elevation: %.2f°, Azimuth: %.2f°\n", elevation[midday_idx], azimuth[midday_idx]);

	
	//Save output to file (Grad Students: Create a graph)

	std::ofstream file("solar_output.csv");
	file << "Minute,Elevation,Azimuth\n";
	for (int i = 0; i < n; i++) {
			file << i << "," << elevation[i] << "," << azimuth[i] << "\n";

	}
	file.close();

    // TO DO: Cleanup memory
    cudaFree(elevation);
    cudaFree(azimuth);

	
	//CPU Implementation - Will be completed in class
    float* elevation_cpu = new float[n];
    float* azimuth_cpu = new float[n];
    
    clock_t start_time = clock();

    // Call CPU function
    calculateSolarCPU(n, latitude, elevation_cpu, azimuth_cpu);

    clock_t end_time = clock();
    double cpu_elapsed_ms = 1000.0 * (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    std::printf("CPU Execution Time: %.2f ms\n\n", cpu_elapsed_ms);

	
    return 0;
}


