
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		//std::cin.get();
		//exit(EXIT_FAILURE);
	}
}

unsigned char* createImageBuffer(unsigned int bytes, unsigned char **devicePtr)
{
	unsigned char *ptr = NULL;
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(devicePtr, ptr, 0);
	return ptr;
}

__global__ void horizontal_sgmentation_kernel(unsigned char* input, unsigned char* output, int width, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	//const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int nx = blockDim.x * gridDim.x;
	//int ny = blockDim.y * gridDim.y;


	for (int row = x; row < height; row += nx){
		int white_cells = 0;
		for (int col = 0; col < width; col++) {
			const int gray_tid = col + row * width;
			if (input[gray_tid] != static_cast<unsigned char>(0.0f)){
				white_cells++;
			}
		}

		if (white_cells > 100){
			for (int col = 0; col < width; col++) {
				const int tid = col + row*width;
				const float white = 255.0f;
				output[tid] = static_cast<unsigned char>(white);

			}
		}
	}
	__syncthreads();


}

__global__ void vertical_sgmentation_kernel(unsigned char* input, unsigned char* output, int width, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	//const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int nx = blockDim.x * gridDim.x;
	//int ny = blockDim.y * gridDim.y;


	for (int col = x; col < width; col += nx){
		int white_cells = 0;
		for (int row = 0; row < height; row++) {
			const int tid = col + row * width;
			if (input[tid] != static_cast<unsigned char>(0.0f)){
				white_cells++;
			}
		}

		if (white_cells > 3){
			for (int row = 0; row < height; row++) {
				const int tid = col + row * width;
				const float white = 255.0f;
				output[tid] = static_cast<unsigned char>(white);

			}
		}
	}
	__syncthreads();


}


#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace gpu;
int main()
{
	Mat img = imread("braille.JPG", CV_LOAD_IMAGE_ANYCOLOR);
	if (img.empty())
	{
		cout << "Image cannot be loaded..!!" << endl;
		return -1;
	}
	GpuMat dst, src;
	src.upload(img);

	GpuMat greyMat;
	gpu::cvtColor(src, greyMat, CV_BGR2GRAY);
	GpuMat img_hist_equalized;
	gpu::equalizeHist(greyMat, img_hist_equalized);
	GpuMat preProcessMat;
	gpu::threshold(greyMat, preProcessMat, 130, 255, THRESH_BINARY);
	gpu::GaussianBlur(preProcessMat, preProcessMat, Size(31, 31), 0.4);
	gpu::threshold(preProcessMat, preProcessMat, 30, 255, THRESH_BINARY);
	gpu::bitwise_not(preProcessMat, preProcessMat);


	Mat horizontalDilate = getStructuringElement(MORPH_RECT, Size(200, 4));
	GpuMat horizontalDilateMat;
	gpu::dilate(preProcessMat, horizontalDilateMat, horizontalDilate);
	Mat preProcessedMat;
	preProcessMat.download(preProcessedMat);
	Mat horizontalDilatedMat;
	horizontalDilateMat.download(horizontalDilatedMat);

	Mat output(horizontalDilatedMat.rows, horizontalDilatedMat.cols, CV_8UC1);

	const int inputBytes = horizontalDilatedMat.step * horizontalDilatedMat.rows;
	const int outputBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, horizontalDilatedMat.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16, 16);

	//Calculate grid size to cover the whole image
	const dim3 grid((horizontalDilatedMat.cols + block.x - 1) / block.x, (horizontalDilatedMat.rows + block.y - 1) / block.y);

	//Launch the color conversion kernel
	horizontal_sgmentation_kernel << <grid, block >> >(d_input, d_output, horizontalDilatedMat.cols, horizontalDilatedMat.rows);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
	//imshow("Output", output);

	vector<Mat> horizontalSubmats;
	bool isPreviousWhite = false;
	bool isCurrentWhite = false;
	int start_row = 0;
	int end_row = start_row;
	int row = 0;

	while ( row < output.rows) {
		if (output.at<uchar>(Point(0, row)) == 255){
			isCurrentWhite = true;
			end_row = row;
			if (!isPreviousWhite){
				start_row = row;
			}
		}

		if (!isCurrentWhite && isPreviousWhite) {
			Mat submat;
			Mat tmp = preProcessedMat(Rect(0, start_row, preProcessedMat.row(row).cols - 1, end_row - start_row));
			tmp.copyTo(submat);
			horizontalSubmats.push_back(submat);
		//	imshow("Output"+row, submat);
		}
		isPreviousWhite = isCurrentWhite;
		isCurrentWhite = false;
		row++;
	}

	//Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");


	vector<Mat> verticalSubmats;
	for (int i = 0; i < horizontalSubmats.size(); i++) {

		Mat verticalDilate = getStructuringElement(MORPH_RECT, Size(1, 10));
		Mat verticalDilatedMat;
		dilate(horizontalSubmats[i], verticalDilatedMat, verticalDilate);

		Mat output(verticalDilatedMat.rows, verticalDilatedMat.cols, CV_8UC1);

		const int inputBytes = verticalDilatedMat.step * verticalDilatedMat.rows;
		const int outputBytes = output.step * output.rows;

		unsigned char *d_input, *d_output;

		//Allocate device memory
		SAFE_CALL(cudaMalloc<unsigned char>(&d_input, inputBytes), "CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_output, outputBytes), "CUDA Malloc Failed");

		//Copy data from OpenCV input image to device memory
		SAFE_CALL(cudaMemcpy(d_input, verticalDilatedMat.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

		//Specify a reasonable block size
		const dim3 block(16, 16);

		//Calculate grid size to cover the whole image
		const dim3 grid((verticalDilatedMat.cols + block.x - 1) / block.x, (verticalDilatedMat.rows + block.y - 1) / block.y);

		//Launch the color conversion kernel
		vertical_sgmentation_kernel << <grid, block >> >(d_input, d_output, verticalDilatedMat.cols, verticalDilatedMat.rows);

		//Synchronize to check for any kernel launch errors
		SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

		//Copy back data from destination device meory to OpenCV output image
		SAFE_CALL(cudaMemcpy(output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

		verticalSubmats.push_back(output);

		//Free the device memory
		SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
		SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");

	}

	int black_cells = 0;
	vector<Mat> cellSubmats;
	for (int i = 0; i < verticalSubmats.size(); i++){

		bool isBlack = true;
		int white_cell_start = 0;
		int white_cell_end = 0;

		for (int col = 0; col < verticalSubmats[i].row(0).cols;) {
			if (verticalSubmats[i].at<uchar>(Point(col, 0)) == 0) {

				black_cells++;
				isBlack = true;
			}
			else{

				int previous_black_cells = black_cells;
				int spaces = previous_black_cells / 50;

				for (int space = 0; space < spaces; space++){
				
					Mat spaceCell;
					Mat tmp = horizontalSubmats[i](Rect(white_cell_end, 0, 50, horizontalSubmats[i].col(0).rows - 1));
					tmp.copyTo(spaceCell);
					cellSubmats.push_back(spaceCell);
				}

				black_cells = 0;

				if (isBlack){
					isBlack = false;

					if (previous_black_cells < 22 && previous_black_cells>0){
						while (verticalSubmats[i].at<uchar>(Point(col, 0)) == 255){
							col++;
						}
					
						col = col - 1;
						Mat submat;
						Mat tmp = horizontalSubmats[i](Rect(white_cell_start, 0, col - white_cell_start + 1, horizontalSubmats[i].col(0).rows - 1));
						tmp.copyTo(submat);
						cellSubmats.push_back(submat);
						
					}
					else{
						white_cell_start = col;
						while (col < verticalSubmats[i].row(0).cols && verticalSubmats[i].at<uchar>(Point(col, 0)) == 255){
							col++;
						}
						;
						col++;
						int white_single_cell_end = col;
						while (col < verticalSubmats[i].row(0).cols && verticalSubmats[i].at<uchar>(Point(col, 0)) == 0){
							col++;
						}
					
						if (col - white_single_cell_end > 22 && col < horizontalSubmats[i].row(0).cols){
							
							Mat submat;
							Mat tmp = horizontalSubmats[i](Rect(white_cell_start, 0, col - white_single_cell_end, horizontalSubmats[i].col(0).rows - 1));
							tmp.copyTo(submat);
							cellSubmats.push_back(submat);

							white_cell_start = col++;
						
						}
						else{

							col = white_cell_start;
						}

					}
				}

			}
			col++;
		}
		
	}

	vector<Mat> characters;
	vector<String> paths = { "d", "e", "f", "h", "o", "p", "s", "space" };
	for (int j = 0; j < 8; j++){
	
		Mat character = imread(paths[j]+".JPG", CV_LOAD_IMAGE_ANYCOLOR);
		characters.push_back(character);
	}

	for (int i = 0; i < cellSubmats.size(); i++){

		for (int j = 0; j < characters.size(); j++){

			String characterName = paths[j];
			resize(cellSubmats[i], cellSubmats[i], characters[j].size());

			Mat cellMat(cellSubmats[i].size(), cellSubmats[i].type());
			threshold(cellSubmats[i], cellMat, 100, 255, cv::THRESH_BINARY);

			Mat charMat(characters[j].size(), characters[j].type());
			threshold(characters[j], charMat, 100, 255, cv::THRESH_BINARY);

			Mat result(characters[j].size(), characters[j].type());
			compare(charMat, cellMat, result, CMP_EQ);

			if (countNonZero(result) >= 2000) {

				cout << characterName << endl;
				break;
			}
		}
	}

	int num;
	cin >>num;

	return 0;
}

