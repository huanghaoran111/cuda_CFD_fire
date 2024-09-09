#pragma once

#include "_/helper_cuda.h"
#include "_/helper_math.h"

#ifndef __CUDA_ARCH__
    #include <chrono>
#endif // !__CUDA_ARCH__

#ifdef __CUDA_ARCH__ 
    #define TIMERSTART(label)                                                   \
        cudaEvent_t start##label, stop##label;                                  \
        float time##label;                                                      \
        cudaEventCreate(&start##label);                                         \
        cudaEventCreate(&stop##label);                                          \
        cudaEventRecord(start##label, 0);
    #define TIMERSTOP(label)                                                    \
        cudaEventRecord(stop##label, 0);                                        \
        cudaEventSynchronize(stop##label);                                      \
        cudaEventElapsedTime(&time##label, start##label, stop##label);          \
        std::cout << "TIMING: " << time##label << " ms (" << #label << ")"      \
                    << std::endl;
#else
    #define TIMERSTART(label)                                                   \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label;  \
        a##label = std::chrono::system_clock::now();
    #define TIMERSTOP(label)                                                    \
        b##label = std::chrono::system_clock::now();                            \
        std::chrono::duration<double> delta##label = b##label-a##label;         \
        std::cout << "# elapsed time ("<< #label <<"): "                        \
                  << delta##label.count()  << "s" << std::endl;
#endif

// 线程块分块向上取整，保证每个数据都能被执行(safe division)
#define SDIV(x,y)(((x)+(y)-1)/(y))

// checkCudaErrors(val)宏，用于检查CUDA API调用是否成功
// checkCudaErrors(val)

#define getAndPrintLastCudaError(msg) _getAndPrintLastCudaError(msg, __FILE__, __LINE__)

#define _getAndPrintLastCudaError(msg, __FILE__, __LINE__)                                              \
    do{                                                                         \
        __getLastCudaError(msg, __FILE__, __LINE__);                            \
        __printLastCudaError(msg, __FILE__, __LINE__);                          \
    } while(0);
