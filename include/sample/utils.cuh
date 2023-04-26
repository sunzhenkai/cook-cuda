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
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, i);
        result.emplace_back(deviceProp.name);
    }
    return std::move(result);
}

template<typename T>
void PrintVector(const std::vector<T> &v) {
    for (auto t: v) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
}