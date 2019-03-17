# superresolution-using-CUDA

Superresolution project using CUDA and Fast Fourier Transform techniques.
Fast Fourier Transform is used to change the matrix of pixels in frequency domain and operate on this signal.
There are used techniques like padding and shifting on singal using CUDA.
You can specify the factor in which resoulution of your image will be changed. There are some operatin modes. 
This program manages to create an image with much higher resolution than the input image in a very short time.
Results are wonderful.

More specific information about the project will be added soon.

To run the project you need to have CUDA installed with cufft library(also add cufft libary in a linker), OpenCV 4.0 or higher(if you want
to use earlier version of OpenCV - there are some changes to make in code with constants). To run the project in Visual Studio you need
to have Visual Studio 2015 compiler(v140) or earlier.
