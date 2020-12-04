/* 
 * Instrument functions for record.cu
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* mutex */
#include "concurrency.cpp"

/* channel for getting message from device */
#include "utils/channel.hpp"


extern "C" __device__ __noinline__ void acquire_lock(int32_t pred,
                                                     uint64_t m_ptr,
                                                     uint64_t pchannel_dev) {
    if (!pred) {
        return;
    }

    mutex* m = (mutex *) m_ptr;

    m->lock();
    printf("pred: %d\n", pred);

    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("%d\n", thread_id);
}

extern "C" __device__ __noinline__ void release_lock(int32_t pred,
                                                       uint64_t m_ptr
                                                       ) {
    printf("pred: %d", pred);
    if (!pred) {
        return;
    }

    mutex* m = (mutex *) m_ptr;

    printf("unlock!\n");
    m->unlock();
}