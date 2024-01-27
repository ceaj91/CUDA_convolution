#include <stdio.h>
#include <cuda.h>
#define INPUT_SIZE 1024
#define KERNEL_SIZE 9
#define OUTPUT_SIZE 1024
#define TILE_SIZE 256

__constant__ float d_kernel[KERNEL_SIZE];

__global__ void conv_1D_tiled_halo_elements(float *in_data, float *out_data, int kernel_width, int data_width)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float in_data_ds[TILE_SIZE + KERNEL_SIZE - 1];

    int n = kernel_width/2;

    int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
    if(threadIdx.x >= blockDim.x - n)
    {
        in_data_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : in_data[halo_index_left];
    }


    in_data_ds[n + threadIdx.x] = in_data[blockIdx.x * blockDim.x + threadIdx.x];


    int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
    if(threadIdx.x < n)
    {
        in_data_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= data_width) ? 0 : in_data[halo_index_right];
    }

    __syncthreads();

    float Pvalue = 0;

    for(int j=0; j<kernel_width; j++)
    {
    	Pvalue += in_data_ds[threadIdx.x + j] * d_kernel[j];
    }

    out_data[i] = Pvalue;
}




int main()
{

	float *input_array  = (float*)malloc(INPUT_SIZE*sizeof(float));
	float *kernel       = (float*)malloc(KERNEL_SIZE*sizeof(float));
	float *output_array = (float*)malloc(OUTPUT_SIZE*sizeof(float));

	float *d_input_array, *d_output_array;
	cudaMalloc(&d_input_array, INPUT_SIZE*sizeof(float));
    //cudaMalloc(&d_kernel, KERNEL_SIZE*sizeof(float));
    cudaMalloc(&d_output_array, OUTPUT_SIZE*sizeof(float));

    for(int i=0; i<INPUT_SIZE; i++)
    {
        input_array[i] = (float)(rand()%10);
    }

    for(int i=0; i<KERNEL_SIZE; i++)
    {
        kernel[i] = (float)(rand()%5);
    }

    cudaMemcpy(d_input_array, input_array, INPUT_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_SIZE*sizeof(float));


	int thr_per_blk = TILE_SIZE;
	int blk_in_grid = ceil( float(OUTPUT_SIZE) / thr_per_blk );
	conv_1D_tiled_halo_elements<<< blk_in_grid, thr_per_blk >>>(d_input_array, d_output_array, KERNEL_SIZE,INPUT_SIZE);

	cudaMemcpy(output_array, d_output_array, OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i=0; i<INPUT_SIZE; i++)
    {
    	
    	printf( "in[%d] = %lf\n", i, input_array[i]);
        
    }
    for(int i=0; i<KERNEL_SIZE; i++)
    {
    	
    	printf( "kernel[%d] = %lf\n", i, kernel[i]);
        
    }
	for(int i=0; i<OUTPUT_SIZE; i++)
    {
    	printf( "out[%d] = %lf\n", i, output_array[i]);
       
    }

    

    free(input_array);
    //free(kernel);
    free(output_array);

    cudaFree(d_input_array);
    cudaFree(d_kernel);
    cudaFree(d_output_array);

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", INPUT_SIZE);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");
}