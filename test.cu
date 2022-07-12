#include <../local/cuda-11.0/targets/x86_64-linux/include/cuda.h>
#include <../local/cuda-11.0/targets/x86_64-linux/include/cuda_runtime.h>
#include <../local/cuda-11.0/targets/x86_64-linux/include/cuda_runtime_api.h>
#include <../include/cudnn.h>
#include <../local/cuda-11.0/targets/x86_64-linux/include/device_launch_parameters.h>

#include <iostream>
#include <stdio.h>

#include "/yuhwan/yuhwan/opencv2/world.hpp"
#include "/yuhwan/yuhwan/opencv2/core.hpp"
#include "/yuhwan/yuhwan/opencv2/highgui.hpp"
#include "/yuhwan/yuhwan/opencv2/imgproc.hpp"
#include "/yuhwan/yuhwan/opencv2/core/mat.hpp"

// /usr/bin/g++ -fdiagnostics-color=always -g /yuhwan/yuhwan/Projects/CUDA/parctice/test.cpp -o /yuhwan/yuhwan/Projects/CUDA/parctice/test -lstdc++ -I/usr/local/cuda-11.0/include -L/usr/local/cuda-11.0/lib64 -Icudart -Icuda -std=c++11

// using namespace std;

int main(int argc, char** argv)
{
    
    cv::Mat();
    
    cudaDeviceProp prob;
    
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    cudaSetDevice(0);

    for(int i = 0; i < numGPUs; i++)
    {
        cudaGetDeviceProperties(&prob, i);
        std::cout << "Device name: " << prob.name << std::endl;
        std::cout << "Global ram: " << prob.totalGlobalMem << std::endl;
        std::cout << "Constent ram: " << prob.totalConstMem << std::endl;
    }
    
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);

    // Define descroptor
    int b = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = b*h*w*c;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, b, h, w, c);
    
    // Create tensor
    float *x;
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float));
    for(int i=0; i<NUM_ELEMENTS; i++)
    {
        x[i] = i * 1.00f;
    }
    std::cout << "Original array: ";
    for(int i=0; i<NUM_ELEMENTS; i++) std::cout << x[i] << ", ";

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x
    );
    

    cudnnDestroy(handle_);
    std::cout << std::endl << "Destroyed cuDNN handle." << std::endl;
    std::cout << "New array: ";
    for(int i=0;i<NUM_ELEMENTS;i++) std::cout << x[i] << " ";
    std::cout << std::endl;
    cudaFree(x);
    return 0;

}
