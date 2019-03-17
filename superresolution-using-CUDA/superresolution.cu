#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <cufft.h>
#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace::cv;
using namespace::std;
using namespace::chrono;

#define M_PI 3.14159265358979323846

__global__ void paddingGPU(cufftComplex *hshift_signal_fft, cufftComplex *h_padding, int x_dim, int y_dim, int x_res_dim, int y_res_dim)
{
	int diff_x = x_res_dim - x_dim;
	int diff_y = y_res_dim - y_dim;

	int up_offset_x = (int)ceilf(diff_x / 2);
	int up_offset_y = (int)ceilf(diff_y / 2);

	int down_offset_x = x_res_dim - (x_dim + up_offset_x);
	int down_offset_y = y_res_dim - (y_dim + up_offset_y);

	int i = threadIdx.x + blockDim.x*blockIdx.x + up_offset_x;
	int j = threadIdx.y + blockDim.y*blockIdx.y + up_offset_y;

	if (i < x_res_dim - down_offset_x && j < y_res_dim - down_offset_y)
	{
		h_padding[j + i * x_res_dim].x = hshift_signal_fft[j - up_offset_y + (i - up_offset_x)*x_dim].x;
		h_padding[j + i * x_res_dim].y = hshift_signal_fft[j - up_offset_y + (i - up_offset_x)*x_dim].y;
	}
}

__global__ void circshiftGPU(cufftComplex *in, cufftComplex *out, int xdim, int ydim, int xshift, int yshift)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	if (i < xdim)
	{
		int x = (i + xshift) % xdim;
		if (j < ydim)
		{
			int y = (j + yshift) % ydim;
			out[y + x * ydim].x = in[j + i * ydim].x;
			out[y + x * ydim].y = in[j + i * ydim].y;
		}
	}
}

Mat calcDFT2CPU(Mat &img)
{
	int width = img.cols;
	int height = img.rows;
	Mat fourierIMG = Mat::zeros(width, height, CV_64F);
	double **realOut = new double*[height];
	double **imagOut = new double*[height];

	double **amplitude = new double*[height];
	for (int i = 0; i < height; i++)
	{
		realOut[i] = new double[width];
		imagOut[i] = new double[width];
		amplitude[i] = new double[width];
	}

	for (int yWave = 0; yWave < height; yWave++)
	{
		for (int xWave = 0; xWave < width; xWave++)
		{
			realOut[yWave][xWave] = 0.0;
			imagOut[yWave][xWave] = 0.0;
			for (int ySpace = 0; ySpace < height; ySpace++)
			{
				for (int xSpace = 0; xSpace < width; xSpace++)
				{
					realOut[yWave][xWave] += (img.at<double>(ySpace, xSpace) * cos(
						2 * M_PI * ((1.0 * xWave * xSpace / width) + (1.0
							* yWave * ySpace / height)))) / sqrt(
								width * height);
					imagOut[yWave][xWave] -= (img.at<double>(ySpace, xSpace) * sin(
						2 * M_PI * ((1.0 * xWave * xSpace / width) + (1.0
							* yWave * ySpace / height)))) / sqrt(
								width * height);
				}
			}
			amplitude[yWave][xWave] = sqrt(
				(realOut[yWave][xWave] * realOut[yWave][xWave])
				+ (imagOut[yWave][xWave]
					* imagOut[yWave][xWave]));
			fourierIMG.at<double>(yWave, xWave) = amplitude[yWave][xWave];
		}
	}

	for (int i = 0; i < height; i++)
	{
		delete[] realOut[i];
		delete[] imagOut[i];
		delete[] amplitude[i];
	}
	delete[] realOut;
	delete[] imagOut;
	delete[] amplitude;

	return fourierIMG;
}

void circshift(cufftComplex *in, cufftComplex *out, int xdim, int ydim, int xshift, int yshift)
{
	for (int i = 0; i < xdim; i++) {
		int ii = (i + xshift) % xdim;
		//if (ii < 0) ii = xdim + ii;
		for (int j = 0; j < ydim; j++) {
			int jj = (j + yshift) % ydim;
			//if (jj < 0) jj = ydim + jj;
			out[ii * ydim + jj].x = in[i * ydim + j].x;
			out[ii * ydim + jj].y = in[i * ydim + j].y;
		}
	}

}

void fftshift(cufftComplex *in, cufftComplex *out, int xdim, int ydim)
{
	circshift(in, out, xdim, ydim, (xdim / 2), (ydim / 2));
}

void ifftshift(cufftComplex *in, cufftComplex *out, int xdim, int ydim)
{
	circshift(in, out, xdim, ydim, ((xdim + 1) / 2), ((ydim + 1) / 2));
}

void padding(cufftComplex *hshift_signal_fft, cufftComplex *h_padding, int x_dim, int y_dim, int x_res_dim, int y_res_dim)
{
	int diff_x = x_res_dim - x_dim;
	int diff_y = y_res_dim - y_dim;

	int up_offset_x = (int)ceil(diff_x / 2);
	int up_offset_y = (int)ceil(diff_y / 2);

	int down_offset_x = x_res_dim - (x_dim + up_offset_x);
	int down_offset_y = y_res_dim - (y_dim + up_offset_y);

	int temp = 0;
	for (int i = up_offset_x; i < x_res_dim - down_offset_x; i++)
	{
		for (int j = up_offset_y; j < y_res_dim - down_offset_y; j++)
		{
			h_padding[j + i * y_res_dim].x = hshift_signal_fft[temp].x;
			h_padding[j + i * y_res_dim].y = hshift_signal_fft[temp++].y;
		}
	}
}

void printMatrix(cufftComplex * mat, int ROW, int COL)
{
	std::cout << "\n Printing Matrix : \n";
	for (int i = 0; i <= ROW - 1; i++) {
		for (int j = 0; j <= COL - 1; j++)
			std::cout << mat[j + i * ROW].x << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void calcFFTCPU(cufftComplex *h_signal, cufftComplex *h_padding, cufftComplex *h_reversed_signal, int NX, int NY, int NX_RES, int NY_RES, float &time)
{
	cufftComplex *d_signal, *d_signal_t;
	cufftComplex *h_signal_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
	cufftComplex *h_shift_signal_fft = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
	cufftComplex *h_signal_ifft = (cufftComplex *)malloc(sizeof(cufftComplex) * NX_RES * NY_RES);
	cufftComplex *h_shift_signal_ifft = (cufftComplex *)malloc(sizeof(cufftComplex) * NX_RES * NY_RES);

	cudaMalloc((void **)&d_signal, NX * NY * sizeof(cufftComplex));
	cudaMalloc((void**)&d_signal_t, NX_RES*NY_RES * sizeof(cufftComplex));
	cudaSetDevice(0);
	cudaMemcpy(d_signal, h_signal, NX * NY * sizeof(cufftComplex), cudaMemcpyHostToDevice);

	printf("Transforming signal cufftExecC2C CPU \n");
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	cufftHandle plan;
	cufftPlan2d(&plan, NX, NY, CUFFT_C2C);
	cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);
	cudaMemcpy(h_signal_fft, d_signal, NX * NY * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	fftshift((cufftComplex *)h_signal_fft, (cufftComplex *)h_shift_signal_fft, NX, NY);
	padding((cufftComplex *)h_shift_signal_fft, (cufftComplex *)h_padding, NX, NY, NX_RES, NY_RES);

	ifftshift((cufftComplex *)h_padding, (cufftComplex *)h_shift_signal_ifft, NX_RES, NY_RES);

	cudaMemcpy(d_signal_t, h_shift_signal_ifft, NX_RES * NY_RES * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftPlan2d(&plan, NX_RES, NY_RES, CUFFT_C2C);
	cufftExecC2C(plan, (cufftComplex *)d_signal_t, (cufftComplex *)d_signal_t, CUFFT_INVERSE);
	cudaMemcpy(h_reversed_signal, d_signal_t, NX_RES * NY_RES * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	time = (float)duration_cast<microseconds>(t2 - t1).count();
	free(h_signal_ifft);
	free(h_shift_signal_ifft);
	free(h_signal_fft);
	free(h_shift_signal_fft);
	cudaFree(d_signal);
	cudaFree(d_signal_t);
	cufftDestroy(plan);
}

void calcFFTGPU(cufftComplex *h_signal, cufftComplex *h_padding, cufftComplex *h_reversed_signal, int NX, int NY, int NX_RES, int NY_RES, float &time)
{
	cufftComplex *d_signal, *d_signal_shift, *d_signal_padding, *d_signal_shift_ifft;
	cudaMalloc((void **)&d_signal, NX * NY * sizeof(cufftComplex));
	cudaMalloc((void **)&d_signal_shift, NX * NY * sizeof(cufftComplex));
	cudaMalloc((void **)&d_signal_padding, NX_RES * NY_RES * sizeof(cufftComplex));
	cudaMalloc((void **)&d_signal_shift_ifft, NX_RES * NY_RES * sizeof(cufftComplex));
	cudaSetDevice(0);
	// Copy host memory to device
	cudaMemcpy(d_signal, h_signal, NX * NY * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_signal_padding, h_padding, NX_RES*NY_RES * sizeof(cufftComplex), cudaMemcpyHostToDevice);
	cufftHandle plan;
	cufftPlan2d(&plan, NX, NY, CUFFT_C2C);

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_FORWARD);

	dim3 blockDim(32, 32, 1);
	dim3 gridDim((NX + 31) / 32, (NY + 31) / 32, 1);
	dim3 gridDim2((NX_RES + 31) / 32, (NY_RES + 31) / 32, 1);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	circshiftGPU << <gridDim, blockDim >> > ((cufftComplex *)d_signal, (cufftComplex *)d_signal_shift, NX, NY, (NX / 2), (NY / 2));
	paddingGPU << < gridDim, blockDim >> > ((cufftComplex *)d_signal_shift, (cufftComplex *)d_signal_padding, NX, NY, NX_RES, NY_RES);
	circshiftGPU << <gridDim2, blockDim >> > ((cufftComplex *)d_signal_padding, (cufftComplex *)d_signal_shift_ifft, NX_RES, NY_RES, ((NX_RES + 1) / 2), ((NY_RES + 1) / 2));
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	time = (float)duration_cast<microseconds>(t2 - t1).count();
	printf("Transforming signal back cufftExecC2C\n");
	cufftPlan2d(&plan, NX_RES, NY_RES, CUFFT_C2C);
	cufftExecC2C(plan, (cufftComplex *)d_signal_shift_ifft, (cufftComplex *)d_signal_shift_ifft, CUFFT_INVERSE);

	// Copy device to host memory
	cudaMemcpy(h_reversed_signal, d_signal_shift_ifft, NX_RES * NY_RES * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
	cudaFree(d_signal);
	cudaFree(d_signal_padding);
	cudaFree(d_signal_shift);
	cudaFree(d_signal_shift_ifft);
	cufftDestroy(plan);
}

int main()
{
	int NX, NY, NX_RES, NY_RES;
	float factory, time;
	char choice;
	cufftComplex *h_signal, *h_reversed_signal, *h_padding;
	Mat img, imgResize;
	img = imread("lena2.jpg", IMREAD_COLOR);

	cvtColor(img, img, cv::COLOR_BGR2GRAY);
	img.convertTo(img, CV_64F, 1.0 / 255.0);
	NX = img.cols;
	NY = img.rows;
	while (true)
	{
		cout << "Factory: ";
		cin >> factory;

		if (factory <= 0.0)
		{
			cout << "Wrong input." << endl;
			system("pause");
			return 0;
		}

		cout << "Which architecture(g/c): ";
		cin >> choice;

		if (choice != 'g' && choice != 'c')
		{
			cout << "Wrong input." << endl;;
			system("pause");
			return 0;
		}

		namedWindow("Source window", WINDOW_AUTOSIZE);
		imshow("Source window", img);
		waitKey(1);

		NX_RES = (int)(NX * factory);
		NY_RES = (int)(NY * factory);
		resize(img, imgResize, Size(), factory, factory);

		h_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * NX * NY);
		h_reversed_signal = (cufftComplex *)malloc(sizeof(cufftComplex) * NX_RES * NY_RES);
		h_padding = (cufftComplex *)malloc(sizeof(cufftComplex) * NX_RES * NY_RES);

		for (unsigned int i = 0; i < NX*NY; i++)
			h_signal[i].x = img.at<double>((int)(i / NX), i%NX);

		for (unsigned int i = 0; i < NX_RES*NY_RES; i++)
		{
			h_padding[i].x = 0;
			h_padding[i].y = 0;
		}

		if (choice == 'g')
		{
			calcFFTGPU(h_signal, h_padding, h_reversed_signal, NX, NY, NX_RES, NY_RES, time);
		}
		else if (choice == 'c')
		{
			calcFFTCPU(h_signal, h_padding, h_reversed_signal, NX, NY, NX_RES, NY_RES, time);
		}

		cout << "Time needed to calculate FFT: " << time << " us." << endl;


		// check result
		for (unsigned int i = 0; i < NX_RES * NY_RES; i++)
		{
			h_reversed_signal[i].x = h_reversed_signal[i].x / (double)(NX_RES*NY_RES);
			h_reversed_signal[i].y = h_reversed_signal[i].y / (double)(NX_RES*NY_RES);
		}

		Mat result = Mat::zeros(NY_RES, NX_RES, CV_64F);
		// Initalize the memory for the signal
		for (unsigned int i = 0; i < NX_RES*NY_RES; i++)
			result.at<double>((int)(i / NX_RES), i%NX_RES) = h_reversed_signal[i].x;

		normalize(result, result, 1, 0, NORM_INF);

		result.convertTo(result, CV_32F);
		//medianBlur(result, result, 3);

		namedWindow("Result window", WINDOW_AUTOSIZE);
		imshow("Result window", result);
		result.convertTo(result, CV_8UC3, 255);
		imwrite("lenaFFTMedian.jpg", result);
		waitKey(1);

		namedWindow("Resized OpenCV", WINDOW_AUTOSIZE);
		imshow("Resized OpenCV", imgResize);
		waitKey(0);

		destroyWindow("Resized OpenCV");
		destroyWindow("Source window");
		destroyWindow("Result window");
		result.release();
		imgResize.release();

		// cleanup memory
		free(h_signal);
		free(h_padding);
		free(h_reversed_signal);
		cudaDeviceReset();
	}

	img.release();
	system("pause");
	return 0;
}
