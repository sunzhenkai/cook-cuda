#pragma once

#include "cuda_runtime.h"
#include "iostream"
#include "vector"

std::vector<std::string> GetDevices() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::vector<std::string> result;
    result.reserve(deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp dp{};
        cudaGetDeviceProperties(&dp, i);
        result.emplace_back(dp.name);
    }
    return std::move(result);
}

void PrintDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "GPU device count: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        // sm: 流式多处理器, Streaming Multiprocessor
        cudaDeviceProp dp{};
        cudaGetDeviceProperties(&dp, i);
        std::cout << "device.0  " << std::endl;
        std::cout << "  sm count: \t\t\t\t" << dp.multiProcessorCount << std::endl;
        std::cout << "  shared memory per block: \t\t" << dp.sharedMemPerBlock / 1024 << "KB" << std::endl;
        std::cout << "  max threads per block:\t\t" << dp.maxThreadsPerBlock << std::endl;
        std::cout << "  max threads per multi processor:\t" << dp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  max threads per sm:\t\t\t" << dp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "  max blocks per multi processor:\t" << dp.maxBlocksPerMultiProcessor << std::endl;
    }
}

template<typename T>
void PrintVector(const std::vector<T> &v) {
    for (auto t: v) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
}