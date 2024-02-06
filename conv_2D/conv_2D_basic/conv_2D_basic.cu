#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#define INPUT_SIZE_X 1024
#define INPUT_SIZE_Y 1024
#define KERNEL_SIZE_X 7
#define KERNEL_SIZE_Y 7
#define OUTPUT_SIZE_X 1024
#define OUTPUT_SIZE_Y 1024
#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8
#define GRID_DIM_X OUTPUT_SIZE_X/BLOCK_DIM_X
#define GRID_DIM_Y OUTPUT_SIZE_Y/BLOCK_DIM_Y

using namespace std;
using namespace chrono;
__global__ void conv_2D_basic(float *kernel, float *in_data, float *out_data, int kernel_width, int data_width)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0;
    int N_start_point_i = i - (kernel_width/2);
    int N_start_point_j = j - (kernel_width/2);


    for(int ki = 0; ki<kernel_width; ki++)
    {
        for(int kj = 0; kj<kernel_width; kj++)
        {
            int ni = N_start_point_i + ki;
            int nj = N_start_point_j + kj;
            if(ni>=0 && ni < data_width && nj>=0 && nj<data_width)
                Pvalue += in_data[ni*data_width + nj] * kernel[ki*kernel_width + kj];    
        }
    }

    
    out_data[i*data_width+j] = Pvalue;
    //out_data[0] = 6;
}




int main()
{
	//HOST DEVICE MEMORY ALLOCATION
	float *input_array  = (float*)malloc(INPUT_SIZE_X*INPUT_SIZE_Y*sizeof(float));
	float *kernel       = (float*)malloc(KERNEL_SIZE_X*KERNEL_SIZE_Y*sizeof(float));
	float *output_array = (float*)malloc(OUTPUT_SIZE_X*OUTPUT_SIZE_Y*sizeof(float));


	//CUDA DEVICE MEMORY ALLOCATION
	float *d_input_array, *d_kernel, *d_output_array;
	cudaMalloc(reinterpret_cast<void **>(&d_input_array), INPUT_SIZE_X*INPUT_SIZE_Y*sizeof(float));
	cudaMalloc(reinterpret_cast<void **>(&d_kernel), KERNEL_SIZE_X*KERNEL_SIZE_Y*sizeof(float));
	cudaMalloc(reinterpret_cast<void **>(&d_output_array), OUTPUT_SIZE_X*OUTPUT_SIZE_Y*sizeof(float));
	
	
    //SET NUMBER FOR INPUT AND KERNEL ARRAY
    for(int i=0; i<INPUT_SIZE_X; i++)
    {
        for(int j=0; j<INPUT_SIZE_Y; j++)
            input_array[i*INPUT_SIZE_Y + j] = (float)(rand()%10);
    }

    for(int i=0; i<KERNEL_SIZE_X; i++)
    {
        for(int j=0; j<KERNEL_SIZE_Y; j++)
		  kernel[i*KERNEL_SIZE_Y + j] = (float)(rand()%5);
    }

	auto start_grid = high_resolution_clock::now();
    //SEND VALUES OF KERNEL AND INPUT ARRAY TO CUDA DEVICE
	cudaMemcpy(d_input_array, input_array, INPUT_SIZE_X*INPUT_SIZE_Y*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_kernel, kernel, KERNEL_SIZE_X*KERNEL_SIZE_Y*sizeof(float),cudaMemcpyHostToDevice);


    
    //SET CUDA DEVICE PARAMETARS
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
	dim3 gridDim(GRID_DIM_X, GRID_DIM_Y);

    //RUN CUDA KERNEL

	conv_2D_basic<<< gridDim,blockDim >>>(d_kernel, d_input_array, d_output_array, KERNEL_SIZE_X,INPUT_SIZE_X);


    //COPY CODE FROM CUDA DEVICE TO HOST DEVICE
       cudaMemcpy(output_array, d_output_array, OUTPUT_SIZE_X*OUTPUT_SIZE_Y*sizeof(float), cudaMemcpyDeviceToHost); 
    
	auto end_grid = high_resolution_clock::now();
	for(int i=0; i<3; i++)
    {
    	for(int j=0; j<3;j++)
    	   printf( "in[%d][%d] = %lf\n", i ,j , input_array[i*INPUT_SIZE_Y+j]);
    }

    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3;j++)
           printf( "kernel[%d][%d] = %lf\n", i ,j , kernel[i*KERNEL_SIZE_Y + j]);
    }
    	
    for(int i=0; i<3; i++)
    {
        for(int j=0; j<3;j++)
           printf( "output[%d][%d] = %lf\n", i ,j , output_array[i*OUTPUT_SIZE_Y + j]);
    }
    	
    
    
   // printf( "out[0][0] = %lf\n", output_array[0]);
        


    
    cudaFree(d_input_array);
    cudaFree(d_kernel);
    cudaFree(d_output_array);

    free(input_array);
    free(kernel);
    free(output_array);

    auto elapsed = duration_cast<microseconds>(end_grid - start_grid);
   // printf("GRID TIME : %lf ms\n\n",elapsed.count());
    cout<<"Elapesd time : "<<elapsed.count()<<" us "<<endl;
    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    /*printf("N                 = %d\n", INPUT_SIZE);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");
    */
}
