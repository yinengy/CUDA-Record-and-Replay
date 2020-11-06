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

extern "C" __device__ __noinline__ void acquire_lock(int32_t pred,
                                                       uint64_t m_ptr
                                                       ) {
    if (!pred) {
        return;
    }

    mutex* m = (mutex *) m_ptr;

    m->lock();

    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("#?#%d,%d\n", block_id, l_thread_id);
}

extern "C" __device__ __noinline__ void release_lock(int32_t pred,
                                                       uint64_t m_ptr
                                                       ) {
    if (!pred) {
        return;
    }

    mutex* m = (mutex *) m_ptr;

    m->unlock();
}