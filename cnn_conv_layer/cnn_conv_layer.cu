#include <stdio.h>
#include <iostream>
#include <chrono>
#define INPUT_SIZE_X 32
#define INPUT_SIZE_Y 32
#define INPUT_CHANNEL_SIZE 32
#define INPUT_TOTAL_SIZE INPUT_SIZE_X * INPUT_SIZE_Y * INPUT_CHANNEL_SIZE

#define KERNEL_SIZE_X 3
#define KERNEL_SIZE_Y 3
#define NUM_OF_KERNELS 32
#define KERNEL_CHANNEL_SIZE INPUT_CHANNEL_SIZE

#define OUTPUT_SIZE_X 32
#define OUTPUT_SIZE_Y 32
#define OUTPUT_CHANNEL_SIZE NUM_OF_KERNELS
#define OUTPUT_TOTAL_SIZE OUTPUT_SIZE_X * OUTPUT_SIZE_Y * NUM_OF_KERNELS

#define BLOCK_DIM_X 8
#define BLOCK_DIM_Y 8

#define GRID_DIM_X OUTPUT_SIZE_X/BLOCK_DIM_Y
#define GRID_DIM_Y OUTPUT_SIZE_Y/BLOCK_DIM_Y

#define TILE_SIZE_X BLOCK_DIM_X
#define TILE_SIZE_Y BLOCK_DIM_Y
#define TILE_SIZE_Z INPUT_CHANNEL_SIZE

#define KERNEL_TOTAL_SIZE INPUT_CHANNEL_SIZE * KERNEL_SIZE_X * KERNEL_SIZE_Y * NUM_OF_KERNELS 
using namespace std;
using namespace chrono;

__constant__ float d_cnn_kernel[KERNEL_TOTAL_SIZE];


__global__ void cnn_conv_layer( float *in_data, float *out_data, int kernel_width, int data_width_x,int data_width_y)
{
    
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = threadIdx.z;
	


	__shared__ float in_data_s[TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z];
	//for(int channel = 0 ; channel < INPUT_CHANNEL_SIZE; channel++)
	in_data_s[threadIdx.z * TILE_SIZE_Y * TILE_SIZE_X + threadIdx.x * TILE_SIZE_X + threadIdx.y] = in_data[k * data_width_x * data_width_y + i * data_width_x + j]; 
	__syncthreads();

	int This_tile_start_point_X = blockIdx.x * blockDim.x;
	int This_tile_start_point_Y = blockIdx.y * blockDim.y;
	//int This_tile_start_point_K = blockIdx.z * blockDim.z;

	int Next_tile_start_point_X = (blockIdx.x + 1) * blockDim.x;
	int Next_tile_start_point_Y = (blockIdx.y + 1) * blockDim.y;
	//int This_tile_start_point_K = (blockIdx.z + 1 ) * blockDim.z;

	float Pvalue = 0;

	int N_start_point_i = i - (kernel_width/2);
	int N_start_point_j = j - (kernel_width/2);
	//start convolution 

	if(i < data_width_y && j < data_width_x)
	{
		for(int ch = 0 ; ch < INPUT_CHANNEL_SIZE; ch++ )
		{
			for(int ki = 0; ki<kernel_width; ki++)
			{
				for(int kj = 0; kj<kernel_width; kj++)
				{
					int ni = N_start_point_i + ki;
					int nj = N_start_point_j + kj;
					if (ni>=0 && ni < data_width_y && nj>=0 && nj<data_width_x)	
					{
						if((ni >= This_tile_start_point_X) && (ni<Next_tile_start_point_X) && (nj >= This_tile_start_point_Y) && (nj < Next_tile_start_point_Y))
							Pvalue += in_data_s[ch * (TILE_SIZE_X * TILE_SIZE_Y) + (threadIdx.x - (kernel_width/2) + ki)*TILE_SIZE_Y + (threadIdx.y - (kernel_width/2) + kj)] * d_cnn_kernel[threadIdx.z * (kernel_width*kernel_width*INPUT_CHANNEL_SIZE) + ch*(kernel_width*kernel_width) +  ki*kernel_width + kj];    
						else
							Pvalue += in_data[ch * (data_width_x*data_width_y) + ni*data_width_x + nj] * d_cnn_kernel[threadIdx.z * (kernel_width*kernel_width*INPUT_CHANNEL_SIZE) + ch*(kernel_width*kernel_width) +  ki*kernel_width + kj];
						


					//	if(i == 0 && j==0 && k ==0)
					//		printf(" Pvalue_d += in_data[%d][%d][%d] * d_kernel[%d][%d][%d][%d] ---> %lf += %lf * %lf\n",ch ,ni, nj, threadIdx.z,ch, ki, kj, Pvalue, in_data[ch * (data_width_x*data_width_y) + ni*data_width_x + nj] , d_cnn_kernel[threadIdx.z * (kernel_width*kernel_width*INPUT_CHANNEL_SIZE) + ch*(kernel_width*kernel_width) +  ki*kernel_width + kj] );

	//					if(i == 0 && j==0 && k ==0)
	//					{
	//						printf(" Pvalue_d += in_data[%d][%d][%d] * d_kernel[%d][%d][%d][%d] ---> %lf += %lf * %lf\n",ch ,ni, nj, threadIdx.z,ch, ki, kj, Pvalue, in_data_s[ch * (TILE_SIZE_X * TILE_SIZE_Y) + (threadIdx.x - (kernel_width/2) + ki)*TILE_SIZE_Y + (threadIdx.y - (kernel_width/2) + kj)]  , d_cnn_kernel[threadIdx.z * (kernel_width*kernel_width*INPUT_CHANNEL_SIZE) + ch*(kernel_width*kernel_width) +  ki*kernel_width + kj] );
	//						printf("threadIdx.x = %d\tthreadIdx.y = %d\tthreadIdx.z = %d\n",threadIdx.x,threadIdx.y, threadIdx.z);
	//					}

					}
					
				}
			}
		}
		out_data[k*(data_width_x * data_width_y) + i*data_width_x+j] = Pvalue;
	}
		
}




int main()
{
	//HOST DEVICE MEMORY ALLOCATION
	float *input_array  = (float*)malloc(INPUT_TOTAL_SIZE*sizeof(float));
	float *kernel   = (float*)malloc(KERNEL_TOTAL_SIZE*sizeof(float));
	float *output_array = (float*)malloc(OUTPUT_TOTAL_SIZE*sizeof(float));
	float *expected_output = (float*)malloc(OUTPUT_TOTAL_SIZE*sizeof(float));

	//CUDA DEVICE MEMORY ALLOCATION
	float *d_input_array, *d_output_array;
	cudaMalloc(reinterpret_cast<void **>(&d_input_array), INPUT_TOTAL_SIZE*sizeof(float));
	cudaMalloc(reinterpret_cast<void **>(&d_output_array), OUTPUT_TOTAL_SIZE*sizeof(float));

	//INPUT FOR SEQ PART
        float ***seq_input = new float**[INPUT_CHANNEL_SIZE];
        for (int i = 0; i < INPUT_CHANNEL_SIZE; i++) {
                seq_input[i] = new float*[INPUT_SIZE_Y];
                for (int j = 0; j < INPUT_SIZE_Y; j++) {
                        seq_input[i][j] = new float[INPUT_SIZE_X];
                }
        }

        //reference kernel array
        float ****seq_kernel = new float***[NUM_OF_KERNELS];
	for(int k_id = 0; k_id < NUM_OF_KERNELS; k_id++)
	{
		seq_kernel[k_id] = new float**[INPUT_CHANNEL_SIZE];
		for (int ch_id = 0; ch_id < INPUT_CHANNEL_SIZE; ch_id++) 
		{
			seq_kernel[k_id][ch_id] = new float*[KERNEL_SIZE_Y];
			for (int i = 0; i < KERNEL_SIZE_Y; i++) 
			{
				seq_kernel[k_id][ch_id][i] = new float[KERNEL_SIZE_X];
			}
		}
        }
	
        float ***seq_output = new float**[OUTPUT_CHANNEL_SIZE];
	for(int ch_id = 0 ; ch_id < OUTPUT_CHANNEL_SIZE; ch_id++)
	{
		seq_output[ch_id] = new float*[OUTPUT_SIZE_Y];
		for (int i = 0; i < OUTPUT_SIZE_Y; i++) 
		{
			seq_output[ch_id][i] = new float[OUTPUT_SIZE_X];
		}
	}
	

	//SET NUMBER FOR INPUT AND KERNEL ARRAY
	for(int ch=0; ch < INPUT_CHANNEL_SIZE; ch++)
	{
		for(int i=0; i < INPUT_SIZE_Y; i++)
		{
			for(int j = 0; j < INPUT_SIZE_X;j++)
			{
				float a=(float)(rand()%10);
		    		input_array[ch * INPUT_SIZE_X*INPUT_SIZE_Y + i * INPUT_SIZE_X + j] = a;
				seq_input[ch][i][j] = a;	
			}
		}
	}

	for(int kernel_id=0; kernel_id < NUM_OF_KERNELS; kernel_id++)
	{
		for(int ch=0; ch < KERNEL_CHANNEL_SIZE; ch++)
		{
			for(int i = 0; i < KERNEL_SIZE_Y; i++)
			{
				for(int j = 0; j < KERNEL_SIZE_X ; j++)
				{
					float a=(float)(rand()%5);
					kernel[kernel_id * (KERNEL_CHANNEL_SIZE * KERNEL_SIZE_X * KERNEL_SIZE_Y) + ch * (KERNEL_SIZE_X*KERNEL_SIZE_Y) + i * KERNEL_SIZE_X + j] = a;
					seq_kernel[kernel_id][ch][i][j] = a;
				}
			}
		}
	}

	auto start_d = high_resolution_clock::now();
	//SEND VALUES OF KERNEL AND INPUT ARRAY TO CUDA DEVICE
	cudaMemcpy(d_input_array, input_array, INPUT_TOTAL_SIZE*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpyToSymbol(d_cnn_kernel, kernel, KERNEL_TOTAL_SIZE*sizeof(float));



	//SET CUDA DEVICE PARAMETARS
	dim3 blockDim(BLOCK_DIM_X,BLOCK_DIM_Y,NUM_OF_KERNELS);
	dim3 gridDim(ceil((float)OUTPUT_SIZE_X/(float)BLOCK_DIM_X),ceil((float)OUTPUT_SIZE_Y/(float)BLOCK_DIM_Y),1);
	

	//RUN CUDA KERNEL

	printf("poziv device uredjaja\n");
	cnn_conv_layer<<< gridDim, blockDim >>>(d_input_array, d_output_array, KERNEL_SIZE_X,INPUT_SIZE_X,INPUT_SIZE_Y);
	printf("kraj uredjaja\n");


//COPY CODE FROM CUDA DEVICE TO HOST DEVICE
	cudaMemcpy(output_array, d_output_array, OUTPUT_TOTAL_SIZE*sizeof(float), cudaMemcpyDeviceToHost); 

	auto end_d = high_resolution_clock::now();


// --------------------------------------------------------------------------------------
// ------------------------------- 3D CONVOLUTION SEQ -----------------------------------
// --------------------------------------------------------------------------------------
	auto start_h = high_resolution_clock::now();
	for(int kernel_id = 0 ; kernel_id < NUM_OF_KERNELS; kernel_id++)
	{
		for(int out_i=0; out_i < OUTPUT_SIZE_Y ; out_i++)
		{
			for(int out_j=0; out_j < OUTPUT_SIZE_X ; out_j++)
			{
				float Pvalue=0;
				float Pvalue_d=0;
				int N_start_point_i = out_i - (KERNEL_SIZE_X/2);
				int N_start_point_j = out_j - (KERNEL_SIZE_Y/2);

				for(int ch = 0; ch < INPUT_CHANNEL_SIZE; ch++)
				{
					for(int ki=0; ki < KERNEL_SIZE_Y ; ki++)
					{
						for(int kj=0; kj < KERNEL_SIZE_X; kj++)
						{
							int ni = N_start_point_i + ki;
							int nj = N_start_point_j + kj;
							if (ni>=0 && ni < INPUT_SIZE_X && nj>=0 && nj<INPUT_SIZE_Y)	
							{
								Pvalue += seq_input[ch][ni][nj] * seq_kernel[kernel_id][ch][ki][kj];
							//	Pvalue_d +=input_array[ch*INPUT_SIZE_X*INPUT_SIZE_Y + ni*INPUT_SIZE_Y + nj] * kernel[kernel_id*INPUT_CHANNEL_SIZE*KERNEL_SIZE_X*KERNEL_SIZE_Y + ch * KERNEL_SIZE_X*KERNEL_SIZE_Y + ki*KERNEL_SIZE_Y +kj];
								if(out_i == 0 && out_j == 0 &&  kernel_id == 0)
								{
		//							printf(" Pvalue += seq_input[%d][%d][%d] * seq_kernel[%d][%d][%d][%d] ---> %lf += %lf * %lf\n",ch,ni,nj,kernel_id,ch,ki,kj,Pvalue, seq_input[ch][ni][nj], seq_kernel[kernel_id][ch][ki][kj]);
								//	printf(" Pvalue_d += in[%d][%d][%d] * kern[%d][%d][%d][%d] ---> %lf += %lf * %lf\n",ch,ni,nj,kernel_id,ch,ki,kj,Pvalue_d, input_array[ch*INPUT_SIZE_X*INPUT_SIZE_Y + ni*INPUT_SIZE_Y + nj], kernel[kernel_id*INPUT_CHANNEL_SIZE*KERNEL_SIZE_X*KERNEL_SIZE_Y + ch * KERNEL_SIZE_X*KERNEL_SIZE_Y + ki*KERNEL_SIZE_Y +kj]);
								}
							}
							
						}
					//	printf("\n");
					}
					//printf("\n");
				}
				seq_output[kernel_id][out_i][out_j] = Pvalue;
			}
		}
	}
	auto end_h = high_resolution_clock::now();
	//compare reference and cuda results
	for(int ch = 0 ; ch < OUTPUT_CHANNEL_SIZE; ch ++)
	{
		for(int i = 0; i<OUTPUT_SIZE_Y ; i++)
		{
			for(int j = 0; j<OUTPUT_SIZE_X ; j++)
			{
				
				if(seq_output[ch][i][j] != output_array[ch*(OUTPUT_SIZE_Y * OUTPUT_SIZE_X) + i*OUTPUT_SIZE_Y + j])
				{
					printf("ERROR i =%d    j=%d !!!\n",i,j);
					printf("ref[%d][%d][%d] != d_out[%d]  ---> %lf != %lf\n",ch,i,j,ch*(OUTPUT_SIZE_Y * OUTPUT_SIZE_X) + i*OUTPUT_SIZE_Y+j,seq_output[ch][i][j],output_array[ch*(OUTPUT_SIZE_Y * OUTPUT_SIZE_X) + i*OUTPUT_SIZE_Y+j]);
					return -1;
				}
				
			}
		}
	}
// --------------------------------------------------------------------------------------
// ------------------------------- FREE MEMORY SPACE ------------------------------------
// --------------------------------------------------------------------------------------
	cudaFree(d_input_array);
	//cudaFree(d_kernel);
	cudaFree(d_output_array);

        for (int i = 0; i < INPUT_CHANNEL_SIZE; i++) {
                for (int j = 0; j < INPUT_SIZE_Y; j++) {
			free(seq_input[i][j]);
                }
		free(seq_input[i]);
        }
	free(seq_input);

	for(int ch_id = 0 ; ch_id < OUTPUT_CHANNEL_SIZE; ch_id ++)
	{
		for (int i = 0; i < OUTPUT_SIZE_Y; i++) 
		{
			seq_output[ch_id][i] = new float[OUTPUT_SIZE_X];
			free(seq_output[ch_id][i]);
		}
		free(seq_output[ch_id]);
	}
	free(seq_output);
	

	for(int k_id = 0; k_id < NUM_OF_KERNELS; k_id++)
	{
		for (int ch_id = 0; ch_id < INPUT_CHANNEL_SIZE; ch_id++) 
		{
			for (int i = 0; i < KERNEL_SIZE_Y; i++) 
			{
				free(seq_kernel[k_id][ch_id][i]);
			}
			free(seq_kernel[k_id][ch_id]);
		}
		free(seq_kernel[k_id]);
        }
	free(seq_kernel);

	free(input_array);
	free(output_array);
//	free(kernel);

// --------------------------------------------------------------------------------------
	auto elapsed_d = duration_cast<microseconds>(end_d - start_d);
	auto elapsed_h = duration_cast<microseconds>(end_h - start_h);
	float acc = (float)(elapsed_h.count())/(float)(elapsed_d.count());
        cout<<"Elapesd CUDA device time : "<<elapsed_d.count()<<" us "<<endl;
        cout<<"Elapesd HOST device time : "<<elapsed_h.count()<<" us "<<endl;
        cout<<"Acceleration : "<<acc<<"x"<<endl;
	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
}
