/*****************************************************

Proyecto: Implementación de Filtro Sobel en CUDA
Nombre: Álvaro Cerdá Pulla
Asignatura: Computadores Avanzados
GitHub: https://github.com/alvaroc20/CA_Cuda

Uso: 
    make compile
    make run

Para cambiar la imagen, hacerlo desde el Makefile
******************************************************/

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_RESET   "\x1b[0m"

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
#include <stdexcept>

#define N atoi(argv[1])

/* 
Autor: LevidRodriguez
Plataforma: GitHub
Enlace: https://github.com/LevidRodriguez/Sobel_with_OpenCV-CUDA 
Ult.Publicacion: Jun,28,2019
*/

__global__ void filtroSobelGPU(unsigned char* srcImg, unsigned char* dstImg, const unsigned int width, const unsigned int height){
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
    		fprintf(stderr, ANSI_COLOR_RED "Cuda error: %s: %s.\n" ANSI_COLOR_RESET, msg, cudaGetErrorString(err));
    		exit(EXIT_FAILURE);
  	}
}



cv::Mat loadImage(char image_name[]){
    using namespace cv;
    
    Mat image = imread(image_name, 255);
    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);


    if(image.empty()) {
        std::cout << ANSI_COLOR_RED "Error: La imagen no se ha cargado correctamente" ANSI_COLOR_RESET<< std::endl;
    }

    return image;
}



int main(int argc, char *argv[]){
    using namespace cv;

    // Control de la compilacion
    if(argc != 3){
        std::cout << ANSI_COLOR_RED "Error en la entrada de argumentos, asegurese de que está haciendo buen uso de la sintaxis.\n" << std::endl;
        std::cout << "Uso: ./filtro <n_hilos> <nombreImagen>\n" ANSI_COLOR_RESET << std::endl;
        return -1;
    }

    // Definir las propiedades de CPU y GPU
    cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
    int cores = devProp.multiProcessorCount;



    
    /* 
    Autor: Lucas Carpenter
    Plataforma: GitHub
    Enlace: https://github.com/lukas783/CUDA-Sobel-Filter
    Ult.Publicacion: Oct,18,2017
    */  

    //Imprimir las propiedades de CPU y GPU.
    std::cout << "\n**********************************************************************************" << std::endl;
    printf(ANSI_COLOR_BLUE "CPU: %d hardware threads\n" ANSI_COLOR_RESET, std::thread::hardware_concurrency());
    printf(ANSI_COLOR_GREEN "GPU Description: %s, CUDA %d.%d, %zd Mbytes global memory, %d CUDA cores\n" ANSI_COLOR_RESET,
    devProp.name, devProp.major, devProp.minor, devProp.totalGlobalMem / 1048576, cores);
    std::cout << "**********************************************************************************\n" << std::endl;





    // Cargar la imagen mediante la libreria cv2.
    Mat image;
    try{
    image = loadImage(argv[2]);
    }catch(Exception e){
        std::cout << ANSI_COLOR_RED "Error: El nombre de imagen que esta seleccionando no existe" ANSI_COLOR_RESET<< std::endl;
        return -1;
    }
    

    // Definiciones e inicializaciones de los recursos necesarios
    unsigned char *gpu_src, *gpu_sobel;
    auto start_time = std::chrono::system_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    // reservamos espacio en la memoria global del device
    cudaMalloc( (void**)&gpu_src, (image.cols * image.rows));
    cudaMalloc( (void**)&gpu_sobel, (image.cols * image.rows));

    // copia del host al device y rellena la matriz resultante de 0.
    cudaMemcpy(gpu_src, image.data, (image.cols*image.rows), cudaMemcpyHostToDevice);
    cudaMemset(gpu_sobel, 0, (image.cols*image.rows));

    // configura los dim3 para que el gpu los use como argumentos, hilos por bloque y número de bloques
    dim3 hilosBloque(N, N, 1);
    dim3 n_bloques(ceil(image.cols/N), ceil(image.rows/N), 1);



    // Ejecutar el filtro sobel utilizando la GPU.
    cudaEventRecord(start);
    start_time = std::chrono::system_clock::now();
    filtroSobelGPU<<< n_bloques, hilosBloque, 0, stream >>>(gpu_src, gpu_sobel, image.cols, image.rows);
    cudaError_t cudaerror = cudaDeviceSynchronize();
    if ( cudaerror != cudaSuccess ) 
        std::cout <<  ANSI_COLOR_RED "Cuda falló al sincronizar: " ANSI_COLOR_RESET<< cudaGetErrorName( cudaerror ) <<std::endl;
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



    // Imprimir el resultado de tiempo del cronometro y el nombre de la imagen original
    std::cout << ANSI_COLOR_YELLOW "Imagen Original:   = " ANSI_COLOR_RESET << argv[2]<<std::endl;
    std::cout << ANSI_COLOR_MAGENTA "Tiempo de ejecución en CUDA:   = " ANSI_COLOR_RESET << 1000*crono_gpu.count() <<" ms"<<std::endl;


    // Crear la nueva imagen
    char new_image[50];
    strcpy(new_image, argv[2]);
    strcat(new_image, "_sobel.png");
    cv::imwrite(new_image,image);
    std::cout << ANSI_COLOR_MAGENTA "Imagen Sobel:   = " ANSI_COLOR_RESET << new_image<<std::endl;

    return 0;
}

