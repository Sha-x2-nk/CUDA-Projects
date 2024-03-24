#include <iostream>

#include <cuda_runtime.h>

int _ConvertSMVer2Cores(int major, int minor) {
    // Refer to the CUDA Compute Capability documentation for the number of cores per multiprocessor
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    switch ((major << 4) + minor) {
        case 0x10: return 8;  // Tesla
        case 0x11: return 8;  // Tesla
        case 0x12: return 8;  // Tesla
        case 0x13: return 8;  // Tesla
        case 0x20: return 32; // Fermi
        case 0x21: return 48; // Fermi
        case 0x30: return 192; // Kepler
        case 0x32: return 192; // Kepler
        case 0x35: return 192; // Kepler
        case 0x37: return 192; // Kepler
        case 0x50: return 128; // Maxwell
        case 0x52: return 128; // Maxwell
        case 0x53: return 128; // Maxwell
        case 0x60: return 64;  // Pascal
        case 0x61: return 128; // Pascal
        case 0x62: return 128; // Pascal
        case 0x70: return 64;  // Volta
        case 0x72: return 64;  // Volta
        case 0x75: return 64;  // Turing
        case 0x80: return 64;  // Ampere
        case 0x86: return 128;  // Ampere
        default: return -1;    // Unknown
    }
}


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable devices found." << std::endl;
        return 1;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        std::cout << "Device #" << deviceId << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  CUDA Cores per Multiprocessor: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << std::endl;
        std::cout << "  Total CUDA Cores: " << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Clock Rate: " << deviceProp.clockRate << " kHz" << std::endl;
        std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
        std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize << " bytes" << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << " threads" << std::endl;
        std::cout << "  Maximum Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Maximum Block Dimensions: " << deviceProp.maxThreadsDim[0] << " x " << deviceProp.maxThreadsDim[1] << " x " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "  Maximum Grid Dimensions: " << deviceProp.maxGridSize[0] << " x " << deviceProp.maxGridSize[1] << " x " << deviceProp.maxGridSize[2] << std::endl;
        std::cout << "  Total Constant Memory: " << deviceProp.totalConstMem << " bytes" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        std::cout << "  Max Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Concurrent Kernels: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
        std::cout << "  ECC Enabled: " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
        std::cout << "  Async Engine Count: " << deviceProp.asyncEngineCount << std::endl;
        std::cout << "  Device Overlap: " << (deviceProp.deviceOverlap ? "Yes" : "No") << std::endl;
        std::cout << std::endl;
    }

    return 0;
}