//#include "cublas_beamformer.h"
//
//
//#include <stdio.h>
//#include <stdlib.h>
//#include <cstdlib>
//#include <curand.h>
//#include <assert.h>
//#include <unistd.h>
//#include <string.h>
//#include <stdlib.h>
//#include <time.h>
//#include <iostream>
//
//using namespace std;
//
//void printUsage();
//
//int main(int argc, char * argv[]) {
//	// Parse input
//	if (argc != 4) {
//		printUsage();
//		return -1;
//	}
//	char input_filename[128];
//	char weight_filename[128];
//	char output_filename[128];
//
//	strcpy(input_filename,  argv[1]);
//	strcpy(weight_filename, argv[2]);
//	strcpy(output_filename, argv[3]);
//
//	// File pointers
//	FILE * data;
//	FILE * weights;
//
//	// File data pointers
//	float * bf_data;
//	float * bf_weights;
//
//	// Complex data pointers
//	float complex * data_dc;
//	float complex * weights_dc;
//	float complex * weights_dc_n;
//
//	// Device data pointers
//	cuComplex * d_weights;
//	cuComplex * d_data;
//
//	struct timespec tstart = {0,0};
//	struct timespec tstop  = {0,0};
//	clock_gettime(CLOCK_MONOTONIC, &tstart);
//
//	// Allocate heap memory for file data
//	bf_data = (float *)malloc(2*N_SAMP*sizeof(float));
//	bf_weights = (float *)malloc(2*N_WEIGHTS*sizeof(float));
//	data_dc = (float complex *)malloc(N_SAMP*sizeof(float complex *));
//	weights_dc = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
//	weights_dc_n = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
//
//	// Open files
//	data = fopen(input_filename, "r");
//	weights = fopen(weight_filename, "r");
//
//	/*********************************************************
//     * Read in weights
//     *********************************************************/
//	int j;
//	if (weights != NULL) {
//		fread(bf_weights, sizeof(float), 2*N_WEIGHTS, weights);
//
//		// Convert to complex numbers (do a conjugate at the same time)
//		for(j = 0; j < N_WEIGHTS; j++){
//			weights_dc_n[j] = bf_weights[2*j] - bf_weights[(2*j)+1]*I;
//		}
//
//		// Transpose the weights
//		int m,n;
//		float complex transpose[N_BEAM][N_ELE*N_BIN];
//		for(m=0;m<N_BEAM;m++){
//			for(n=0;n<N_ELE*N_BIN;n++){
//				transpose[m][n] = weights_dc_n[m*N_ELE*N_BIN + n];
//			}
//		}
//		for(n=0;n<N_ELE*N_BIN;n++){
//			for(m=0;m<N_BEAM;m++){
//				weights_dc[n*N_BEAM+ m] = transpose[m][n];
//			}
//		}
//		fclose(weights);
//	}
//	free(bf_weights);
//
//	// Copy weights to device
//	cudaMalloc((void **)&d_weights, N_WEIGHTS*sizeof(cuComplex)); //*N_TIME
//	cudaMemcpy(d_weights, weights_dc, N_WEIGHTS*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_weights instead of weights_dc //*N_TIME
//
//	/*********************************************************
//	 * Read in Data
//	 *********************************************************/
//	if (data != NULL) {
//		fread(bf_data, sizeof(float), 2*N_SAMP, data);
//
//		// Make 'em complex!
//		for (j = 0; j < N_SAMP; j++) {
//			data_dc[j] = bf_data[2*j] + bf_data[(2*j)+1]*I;
//		}
//
//		// Specify grid and block dimensions
//		dim3 dimBlock_d(N_ELE, 1, 1);
//		dim3 dimGrid_d(N_TIME, N_BIN, 1);
//
//		cuComplex * d_data1;
//
//		cudaMalloc((void **)&d_data1, N_SAMP*sizeof(cuComplex));
//		cudaMalloc((void **)&d_data, N_SAMP*sizeof(cuComplex));
//
//
//		cudaMemcpy(d_data1,    data_dc,   N_SAMP*sizeof(cuComplex), cudaMemcpyHostToDevice);
//		data_restructure<<<dimGrid_d, dimBlock_d>>>(d_data1, d_data);
//
//		fclose(data);
//	}
//	free(bf_data);
//
//	clock_gettime(CLOCK_MONOTONIC, &tstop);
//	printf("Data and Weights restructure elapsed time: %.5f seconds\n",
//	((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
//	((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
//
//
//	// Allocate memory for the output
//	float * output_f;
//	output_f = (float *)calloc(N_POL*(N_OUTPUTS/2),sizeof(float));
//
//
//
//	// Specify grid and block dimensions
//	dim3 dimBlock(N_STI_BLOC, 1, 1);
//	dim3 dimGrid(N_BIN, N_BEAM1, N_STI);
//
//	cuComplex * d_beamformed;//////////
//	float * d_outputs;
//
//	cudaMalloc((void **)&d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2));
//	cudaError_t err_malloc = cudaMalloc((void **)&d_beamformed, N_TBF*sizeof(cuComplex));
//	if (err_malloc != cudaSuccess) {
//		printf("CUDA Error (cudaMalloc2): %s\n", cudaGetErrorString(err_malloc));
//	}
//
//	printf("Starting beamformer\n");
//	cublasHandle_t handle;
//	beamform(d_weights, d_data, handle, d_beamformed);//beamform<<<dimGrid, dimBlock>>>(d_data, d_weights, d_beamformed);
//
//	cudaError_t err_code = cudaGetLastError();
//	if (err_code != cudaSuccess) {
//		printf("CUDA Error (beamform): %s\n", cudaGetErrorString(err_code));
//	}
//
//	printf("Starting sti_reduction\n");
//	sti_reduction<<<dimGrid, dimBlock>>>(d_beamformed,d_outputs);
//	printf("Finishing sti_reduction\n");
//
//	err_code = cudaGetLastError();
//	if (err_code != cudaSuccess) {
//		printf("CUDA Error (sti_reduction): %s\n", cudaGetErrorString(err_code));
//	}
//
//	cudaMemcpy(output_f, d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2),cudaMemcpyDeviceToHost);
//
//	cudaFree(d_data);
//	cudaFree(d_weights);
//	cudaFree(d_outputs);
//
//	// Save output data to file
//	FILE * output;
//	output = fopen(output_filename, "w");
//	fwrite(output_f, sizeof(float), N_POL*(N_OUTPUTS/2), output);
//	fclose(output);
//
//	free(data_dc);
//	free(weights_dc);
//	free(output_f);
//	//	cublasDestroy(handle);
//
//	return 0;
//}
//
//void printUsage() {
//	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
//}
//
////For makefile at the very end "-fno-exceptions -fno-rtti"


#include "cublas_beamformer.h"


#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

void printUsage();

int main(int argc, char * argv[]) {
	// Parse input
	if (argc != 4) {
		printUsage();
		return -1;
	}
	char input_filename[128];
	char weight_filename[128];
	char output_filename[128];

	strcpy(input_filename,  argv[1]);
	strcpy(weight_filename, argv[2]);
	strcpy(output_filename, argv[3]);

	/*********************************************************
	 *Create a handle for CUBLAS
	 *********************************************************/
	cublasHandle_t handle;
	cublasCreate(&handle);

	/*********************************************************
	 * Initialize beamformer
	 *********************************************************/
	init_beamformer();

	/*********************************************************
	 * Update in weights
	 *********************************************************/
	update_weights(weight_filename);

	/*********************************************************
	 * Input data and restructure for cublasCgemmBatched()
	 *********************************************************/
	data_in(input_filename);

	// Allocate memory for the output
	float * output_f;
	output_f = (float *)calloc(N_POL*(N_OUTPUTS/2),sizeof(float));

	/*********************************************************
     * Run beamformer
     *********************************************************/
	run_beamformer(handle, output_f);

	// Save output data to file
	FILE * output;
	output = fopen(output_filename, "w");
	fwrite(output_f, sizeof(float), N_POL*(N_OUTPUTS/2), output);
	fclose(output);

	free(output_f);
	//	cublasDestroy(handle);

	return 0;
}

void printUsage() {
	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
}
//  // Start and stop time - Used time certain sections of code (Not very accurate, use profiler or cudaThreadSynchronize())
//	struct timespec tstart = {0,0};
//	struct timespec tstop  = {0,0};
//	clock_gettime(CLOCK_MONOTONIC, &tstart);

//	clock_gettime(CLOCK_MONOTONIC, &tstop);
//	printf("Data and Weights restructure elapsed time: %.5f seconds\n",
//	((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
//	((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

//For makefile at the very end "-fno-exceptions -fno-rtti"
