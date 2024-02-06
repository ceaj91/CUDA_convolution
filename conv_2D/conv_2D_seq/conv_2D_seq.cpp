#include <stdio.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace chrono;
#define INPUT_SIZE_X 1024
#define INPUT_SIZE_Y 1024
#define KERNEL_SIZE_X 7
#define KERNEL_SIZE_Y 7
#define OUTPUT_SIZE_X 1024
#define OUTPUT_SIZE_Y 1024


void conv_2D_seq(float **kernel, float **in_data, float **out_data, int kernel_width, int data_width)
{


	float Pvalue = 0;
	for(int out_pix_i = 0; out_pix_i<data_width; out_pix_i++ )
	{
	
		for(int out_pix_j = 0; out_pix_j<data_width; out_pix_j++ )
		{
			Pvalue=0;

			for(int kernel_pix_i = 0; kernel_pix_i<kernel_width; kernel_pix_i++ )
			{

				for(int kernel_pix_j = 0; kernel_pix_j<kernel_width; kernel_pix_j++ )
				{
					int i_index = out_pix_i - (kernel_width/2) + kernel_pix_i;
					int j_index = out_pix_j - (kernel_width/2) + kernel_pix_j;
					if(i_index >= 0 && i_index < data_width && j_index >= 0 && j_index < data_width)
						Pvalue += in_data[i_index][j_index] * kernel[kernel_pix_i][kernel_pix_j];
				}
			}
			out_data[out_pix_i][out_pix_j] = Pvalue;
		}
	}
}



int main()
{
	float **input_array = new float*[INPUT_SIZE_X];
	for (int i = 0; i < INPUT_SIZE_X; i++) {
		input_array[i] = new float[INPUT_SIZE_Y];
		for (int j = 0; j < INPUT_SIZE_Y; j++) {
			input_array[i][j] = static_cast<float>(rand() % 10);
		}
	}

	float **kernel = new float*[KERNEL_SIZE_X];
	for (int i = 0; i < KERNEL_SIZE_X; i++) {
		kernel[i] = new float[KERNEL_SIZE_Y];
		for (int j = 0; j < KERNEL_SIZE_Y; j++) {
			kernel[i][j] = static_cast<float>(rand() % 5);
		}
	}

	float **output_array = new float*[OUTPUT_SIZE_X];
	for (int i = 0; i < OUTPUT_SIZE_X; i++) {
		output_array[i] = new float[OUTPUT_SIZE_Y];
	}

	auto start_grid = high_resolution_clock::now();
	conv_2D_seq(kernel, input_array, output_array, KERNEL_SIZE_X, INPUT_SIZE_X);
	auto stop_grid = high_resolution_clock::now();
	auto duration_grid = duration_cast<microseconds>(stop_grid - start_grid);
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3;j++)
			printf( "in[%d][%d] = %lf\n", i ,j , input_array[i][j]);
	}
	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3;j++)
			printf( "kernel[%d][%d] = %lf\n", i ,j , kernel[i][j]);
	}

	for(int i=0; i<3; i++)
	{
		for(int j=0; j<3;j++)
			printf( "output[%d][%d] = %lf\n", i ,j , output_array[i][j]);
	}
	for(int i=1021; i<1024; i++)
	{
		for(int j=1021; j<1024;j++)
			printf( "output[%d][%d] = %lf\n", i ,j , output_array[i][j]);
	}


	cout << "Time taken by function: " << duration_grid.count() << " microseconds" << endl;

	for (int i = 0; i < INPUT_SIZE_X; i++) {
		delete[] input_array[i];
	}
	delete[] input_array;

	for (int i = 0; i < KERNEL_SIZE_X; i++) {
		delete[] kernel[i];
	}
	delete[] kernel;

	for (int i = 0; i < OUTPUT_SIZE_X; i++) {
		delete[] output_array[i];
	}
	delete[] output_array;




}
