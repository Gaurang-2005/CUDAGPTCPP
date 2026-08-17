#include "memoryManager/memoryManager.hpp"

void memoryManager::evict() {
    bool free = true;
    while (memManagerActive) {
        size_t free_vram = 0;
        size_t total_vram = 0;

        cudaError_t error = cudaMemGetInfo(&free_vram, &total_vram);

        if (error == cudaSuccess) {
            if (free_vram < threshold) {
                if (qu.top()) qu.top() -> toCPU();
                free = false;
            }
            else free = true;
        } 
        else {
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
            std::abort();
        }
        if (free) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}