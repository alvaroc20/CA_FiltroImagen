/*****************************************************

Proyecto: Implementación de Filtro Sobel en CUDA
Nombre: Álvaro Cerdá Pulla
Asignatura: Computadores Avanzados
GitHub: https://github.com/alvaroc20/CA_Cuda

Uso: 
    make compile
    make run

******************************************************/




#include <stdio.h>
#include <assert.h>
#include </usr/include/opencv4/opencv2/opencv.hpp>
#include </usr/include/opencv4/opencv2/core.hpp>
#include </usr/include/opencv4/opencv2/opencv_modules.hpp>
#include </usr/include/opencv4/opencv2/core/mat.hpp>
#include <thread>
#include <chrono>
#include <iostream>
#include <string.h>


#define N atoi(argv[1])


__global__ void sobelFilterGPU(unsigned char* srcImg, unsigned char* dstImg, const unsigned int width, const unsigned int height){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        float dx = (-1* srcImg[(y-1)*width + (x-1)]) + (-2*srcImg[y*width+(x-1)]) + (-1*srcImg[(y+1)*width+(x-1)]) +
             (    srcImg[(y-1)*width + (x+1)]) + ( 2*srcImg[y*width+(x+1)]) + (   srcImg[(y+1)*width+(x+1)]);
             
        float dy = (    srcImg[(y-1)*width + (x-1)]) + ( 2*srcImg[(y-1)*width+x]) + (   srcImg[(y-1)*width+(x+1)]) +
             (-1* srcImg[(y+1)*width + (x-1)]) + (-2*srcImg[(y+1)*width+x]) + (-1*srcImg[(y+1)*width+(x+1)]);
        dstImg[y*width + x] = sqrt( (dx*dx) + (dy*dy) ) > 255 ? 255 : sqrt( (dx*dx) + (dy*dy) );
    }
}


void checkCUDAError(const char* msg) 
{
	cudaError_t err = cudaGetLastError();
  	if (cudaSuccess != err) 
  	{
    		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    		exit(EXIT_FAILURE);
  	}
}



cv::Mat loadImage(char image_name[]){
    using namespace cv;

    Mat image = imread(image_name, 255);
    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);

    if(image.empty()) {
        std::cout << "Error: La imagen no se ha cargado correctamente" << std::endl;
    }
    return image;
}



int main(int argc, char *argv[]){
    using namespace cv;

    // Definir las propiedades de CPU y GPU
    cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
    int cores = devProp.multiProcessorCount;


    /*Imprimir las propiedades de CPU y GPU. 
    Esta bloque de codigo esta obtenido desde varios repositorios en los que vi lo mismo*/
    printf("CPU: %d hardware threads\n", std::thread::hardware_concurrency());
    printf("GPU Description: %s, CUDA %d.%d, %zd Mbytes global memory, %d CUDA cores\n",
    devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores);


    // Cargar la imagen
    Mat image = loadImage(argv[2]);
    

    // Definiciones e inicializaciones de los recursos necesarios
    unsigned char *gpu_src, *gpu_sobel;
    auto start_time = std::chrono::system_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // Asignar memoria para las imágenes en memoria GPU.
    cudaMalloc( (void**)&gpu_src, (image.cols * image.rows));
    cudaMalloc( (void**)&gpu_sobel, (image.cols * image.rows));

    // Transfiera del host al device y configura la matriz resultante a 0s
    cudaMemcpy(gpu_src, image.data, (image.cols*image.rows), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (image.cols*image.rows));

    // configura los dim3 para que el gpu los use como argumentos, hilos por bloque y número de bloques
    dim3 threadsPerBlock(N, N, 1);
    dim3 numBlocks(ceil(image.cols/N), ceil(image.rows/N), 1);



    // Ejecutar el filtro sobel utilizando la GPU.
    cudaEventRecord(start);
    start_time = std::chrono::system_clock::now();
    sobelFilterGPU<<< numBlocks, threadsPerBlock, 0, stream >>>(gpu_src, gpu_sobel, image.cols, image.rows);
    cudaError_t cudaerror = cudaDeviceSynchronize();
    if ( cudaerror != cudaSuccess ) 
        std::cout <<  "Cuda failed to synchronize: " << cudaGetErrorName( cudaerror ) <<std::endl;
    std::chrono::duration<double> crono_gpu = std::chrono::system_clock::now() - start_time;
    

    // Copia los datos al CPU desde la GPU, del device al host
    cudaMemcpy(image.data, gpu_sobel, (image.cols*image.rows), cudaMemcpyDeviceToHost);



    // Libera recursos
    cudaEventRecord(stop);
    float time_milliseconds = 0;
    cudaEventElapsedTime(&time_milliseconds, start, stop);
    cudaStreamDestroy(stream); 
    cudaFree(gpu_src); 
    cudaFree(gpu_sobel);



    // Imprimir el resultado de tiempo del cronometro
    std::cout << "Tiempo de ejecución en CUDA:   = " << 1000*crono_gpu.count() <<" ms"<<std::endl;


    // Crear la nueva imagen
    char new_image[50];
    strcpy(new_image, argv[2]);
    strcat(new_image, "_sobel.png");
    cv::imwrite(new_image,image);


    return 0;
}

