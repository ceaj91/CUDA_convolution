#include <stdio.h>
#include <iostream>
#include <chrono>
#define INPUT_SIZE_X 3840
#define INPUT_SIZE_Y 2160
#define KERNEL_SIZE_X 3
#define KERNEL_SIZE_Y 3
#define OUTPUT_SIZE_X 3840
#define OUTPUT_SIZE_Y 2160

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define GRID_DIM_X OUTPUT_SIZE_X/BLOCK_DIM_Y
#define GRID_DIM_Y OUTPUT_SIZE_Y/BLOCK_DIM_Y

#define TILE_SIZE_X BLOCK_DIM_X
#define TILE_SIZE_Y BLOCK_DIM_Y
using namespace std;
using namespace chrono;

__constant__ float d_kernel[KERNEL_SIZE_X * KERNEL_SIZE_Y];


__global__ void conv_2D_constant_mem( float *in_data, float *out_data, int kernel_width, int data_width_x, int data_width_y)
{
    
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float in_data_s[TILE_SIZE_X * TILE_SIZE_Y];
	if(i< data_width_y && j < data_width_x)
		in_data_s[threadIdx.x * TILE_SIZE_Y + threadIdx.y] = in_data[i*data_width_x + j]; 
	__syncthreads();

	int This_tile_start_point_X = blockIdx.x * blockDim.x;
	int This_tile_start_point_Y = blockIdx.y * blockDim.y;

	int Next_tile_start_point_X = (blockIdx.x + 1) * blockDim.x;
	int Next_tile_start_point_Y = (blockIdx.y + 1) * blockDim.y;

	float Pvalue = 0;

	int N_start_point_i = i - (kernel_width/2);
	int N_start_point_j = j - (kernel_width/2);
	
	if(i < data_width_y && j < data_width_x)
	{
		for(int ki = 0; ki<kernel_width; ki++)
		{
			for(int kj = 0; kj<kernel_width; kj++)
			{
				int ni = N_start_point_i + ki;
				int nj = N_start_point_j + kj;

				if(ni>=0 && ni < data_width_y && nj>=0 && nj<data_width_x)
				{
					if((ni >= This_tile_start_point_X) && (ni<Next_tile_start_point_X) && (nj >= This_tile_start_point_Y) && (nj < Next_tile_start_point_Y))
						Pvalue += in_data_s[(threadIdx.x - (kernel_width/2) + ki)*TILE_SIZE_Y + (threadIdx.y - (kernel_width/2) + kj)] * d_kernel[ki*kernel_width + kj];    
					else
						Pvalue += in_data[ni*data_width_x + nj] * d_kernel[ki*kernel_width + kj];
						//if(i == 0 && j==1024)
						//	printf(" Pvalue_d += in_data[%d][%d] * d_kern[%d][%d] ---> %lf += %lf * %lf\n",ni, nj, ki, kj, Pvalue, in_data[ni*data_width_x + nj] , d_kernel[ki*kernel_width + kj]);
				}
			}
		}


		out_data[i*data_width_x+j] = Pvalue;
	}	
}

void conv_2D_seq(float **kernel, float **in_data, float **out_data, int kernel_width, int data_width_x, int data_width_y)
{
        float Pvalue = 0;
        for(int out_pix_i = 0; out_pix_i<data_width_y; out_pix_i++ )
        {

                for(int out_pix_j = 0; out_pix_j<data_width_x; out_pix_j++ )
                {
                        Pvalue=0;

                        for(int kernel_pix_i = 0; kernel_pix_i<kernel_width; kernel_pix_i++ )
                        {

                                for(int kernel_pix_j = 0; kernel_pix_j<kernel_width; kernel_pix_j++ )
                                {
                                        int i_index = out_pix_i - (kernel_width/2) + kernel_pix_i;
                                        int j_index = out_pix_j - (kernel_width/2) + kernel_pix_j;
                                        if(i_index >= 0 && i_index < data_width_y && j_index >= 0 && j_index < data_width_x)
					{
                                                Pvalue += in_data[i_index][j_index] * kernel[kernel_pix_i][kernel_pix_j];
						if(out_pix_i == 0 && out_pix_j == 1024)
					//	{
							printf(" Pvalue += ref_in[%d][%d] * ref_kern[%d][%d] ---> %lf += %lf * %lf\n",i_index, j_index, kernel_pix_i, kernel_pix_j, Pvalue, in_data[i_index][j_index] , kernel[kernel_pix_i][kernel_pix_j]);
			

					}
				}		
                        }
                        out_data[out_pix_i][out_pix_j] = Pvalue;
                }
        }

}



int main()
{
	//HOST DEVICE MEMORY ALLOCATION
	float *input_array  = (float*)malloc(INPUT_SIZE_X*INPUT_SIZE_Y*sizeof(float));
	float *kernel       = (float*)malloc(KERNEL_SIZE_X*KERNEL_SIZE_Y*sizeof(float));
	float *output_array = (float*)malloc(OUTPUT_SIZE_X*OUTPUT_SIZE_Y*sizeof(float));


	//CUDA DEVICE MEMORY ALLOCATION
	float *d_input_array, *d_output_array;
	cudaMalloc(reinterpret_cast<void **>(&d_input_array), INPUT_SIZE_X*INPUT_SIZE_Y*sizeof(float));
	//cudaMalloc(&d_kernel, KERNEL_SIZE_X*KERNEL_SIZE_Y*sizeof(float));
	cudaMalloc(reinterpret_cast<void **>(&d_output_array), OUTPUT_SIZE_X*OUTPUT_SIZE_Y*sizeof(float));

	//INPUT FOR SEQ PART
	float **seq_input = new float*[INPUT_SIZE_Y];
        for (int i = 0; i < INPUT_SIZE_Y; i++) {
                seq_input[i] = new float[INPUT_SIZE_X];
        }
	float **seq_kernel = new float*[KERNEL_SIZE_Y];
        for (int i = 0; i < KERNEL_SIZE_Y; i++) {
                seq_kernel[i] = new float[KERNEL_SIZE_X];
        }
	float **seq_output = new float*[OUTPUT_SIZE_Y];
        for (int i = 0; i < OUTPUT_SIZE_Y; i++) {
                seq_output[i] = new float[OUTPUT_SIZE_X];
        }


	//generate numbers for input matrices
	for(int i=0; i<INPUT_SIZE_Y; i++)
	{
		for(int j=0; j<INPUT_SIZE_X; j++)
		{
		    float a = (float)(rand()%10);
		    input_array[i*INPUT_SIZE_X + j] = a;
		    seq_input[i][j] = a;
		}
	}

	//generate numbers for kernel
	for(int i=0; i<KERNEL_SIZE_Y; i++)
	{
		for(int j=0; j<KERNEL_SIZE_X; j++)
		{
		    float a = (float)(rand()%5);
		    kernel[i*KERNEL_SIZE_X + j] = a;
		    seq_kernel[i][j] = a;
		}
	}

	auto start_d = high_resolution_clock::now();
	//SEND VALUES OF KERNEL AND INPUT ARRAY TO CUDA DEVICE
	cudaMemcpy(d_input_array, input_array, INPUT_SIZE_X*INPUT_SIZE_Y*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_SIZE_X*KERNEL_SIZE_Y*sizeof(float));



	//SET CUDA DEVICE PARAMETARS
	//dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y);
	//dim3 gridDim(ceil(INPUT_SIZE_X/BLOCK_DIM_X), ceil(INPUT_SIZE_Y/BLOCK_DIM_Y));

	dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y);
	//dim3 gridDim(ceil(INPUT_SIZE_X/BLOCK_DIM_X), ceil(INPUT_SIZE_Y/BLOCK_DIM_Y));
	dim3 gridDim(ceil((float)OUTPUT_SIZE_Y/(float)BLOCK_DIM_Y),ceil((float)OUTPUT_SIZE_X/(float)BLOCK_DIM_X));
	//RUN CUDA KERNEL

	conv_2D_constant_mem<<< gridDim, blockDim >>>(d_input_array, d_output_array, KERNEL_SIZE_X,INPUT_SIZE_X,INPUT_SIZE_Y);


	//COPY CODE FROM CUDA DEVICE TO HOST DEVICE
	cudaMemcpy(output_array, d_output_array, OUTPUT_SIZE_X*OUTPUT_SIZE_Y*sizeof(float), cudaMemcpyDeviceToHost); 

	

	auto end_d = high_resolution_clock::now();

	//-----------------------------------------SEQ PART OF 2D CONVOLUTIO--------------------------------

	auto start_h = high_resolution_clock::now();
	conv_2D_seq(seq_kernel,seq_input,seq_output,KERNEL_SIZE_X, INPUT_SIZE_X, INPUT_SIZE_Y);
	auto end_h = high_resolution_clock::now();

	for(int i=0; i < OUTPUT_SIZE_Y; i++)
	{
		for(int j=0; j < OUTPUT_SIZE_X; j++)
		{
			if(seq_output[i][j] != output_array[i*OUTPUT_SIZE_X + j])
			{
				printf("Calculation missmatch!!! \n");
				printf("i = %d   j = %d\n",i,j);
				printf("seq_output[%d][%d] != output_array[%d] ---> %lf != %lf\n", i, j, i*OUTPUT_SIZE_X + j,seq_output[i][j],output_array[i*OUTPUT_SIZE_X + j] );
				return -1;
			}
		}
	}


	cudaFree(d_input_array);
	//cudaFree(d_kernel);
	cudaFree(d_output_array);



	free(input_array);
	free(kernel);
	free(output_array);

	//printf("aaa \n");
        for (int i = 0; i < INPUT_SIZE_Y; i++) {
                free(seq_input[i]);
        }

        for (int i = 0; i < KERNEL_SIZE_Y; i++) {
                free(seq_kernel[i]);
        }

        for (int i = 0; i < OUTPUT_SIZE_Y; i++) {
                free(seq_output[i]);
        }

	free(seq_input);
	free(seq_kernel);
	free(seq_output);

	auto elapsed_d = duration_cast<microseconds>(end_d - start_d);
	auto elapsed_h = duration_cast<microseconds>(end_h - start_h);
	float acc = (float)(elapsed_h.count())/(float)(elapsed_d.count());
	cout<<"Elapesd CUDA device time : "<<elapsed_d.count()<<" us "<<endl;
	cout<<"Elapesd HOST device time : "<<elapsed_h.count()<<" us "<<endl;
	cout<<"Acceleration : "<<acc<<"x"<<endl;
	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	/*printf("N                 = %d\n", INPUT_SIZE);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");
	*/
}
