/* 
 * Instrument functions for record.cu
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <stdint.h>
#include <stdio.h>

extern "C" __device__ __noinline__ void acquire_lock(int32_t pred,
                                                     volatile int *current_threadid) {
    if (!pred) {
        return;
    }
    
    // it is in fact an adopted version of ticket lock
    // since (block_id, thread_id) are unique to all threads, 
    // it is ok to just use it as ticket
    // so there is no need to get new ticket when acquire the lock
    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int thread_id = block_id * (blockDim.x * blockDim.y * blockDim.z)
                    + (threadIdx.z * (blockDim.x * blockDim.y))
                    + (threadIdx.y * blockDim.x) + threadIdx.x;

    // waiting for being called
    // ticket is volatile so will be read once its val change
    while (true) { 
        if (thread_id == *current_threadid || *current_threadid == -1) {
            // lock acquired
            // now all other threads are spinning
            break;
        } 
    }
}

extern "C" __device__ __noinline__ void release_lock(int32_t pred,
                                                     long *schedule_counter,
                                                     int *schedule,
                                                     int *current_threadid) {
    if (!pred) {
        return;
    }
    
    long idx = *schedule_counter + 1;
    *schedule_counter = idx;

    // update ticket so the lock is released
    *current_threadid = schedule[idx];
    // next thread is read to go
    __threadfence();
}