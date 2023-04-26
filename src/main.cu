#include <iostream>
#include "sample/utils.cuh"

int main() {
    auto devices = GetDevices();
    PrintVector(devices);
    return 0;
}
