/* 
 * Instrument functions for record.cu
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"


extern "C" __device__ __noinline__ void acquire_lock(int32_t pred,
                                                     volatile int *has_new_ticket,
                                                     volatile int *current_blockid,
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

    int thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;


    // waiting for being called
    // ticket is volatile so will be read once its val change
    while (true) {  
        if (*has_new_ticket == 1) {
            if (block_id == *current_blockid && thread_id == *current_threadid || *current_blockid == -1) {
                // lock acquired
                // now all other threads are spinning
                printf("#?#%d,%d\n", block_id, thread_id);
                *has_new_ticket = 0;
                break;
            } 
        }
    }
}

extern "C" __device__ __noinline__ void release_lock(int32_t pred,
                                                     int *has_new_ticket,
                                                     long *schedule_counter,
                                                     int *schedule_blockid,
                                                     int *schedule_threadid,
                                                     int *current_blockid,
                                                     int *current_threadid) {
    if (!pred) {
        return;
    }
    
    long idx = *schedule_counter + 1;
    *schedule_counter = idx;

    // update ticket so the lock is released
    *current_blockid = schedule_blockid[idx];
    *current_threadid = schedule_threadid[idx];
    *has_new_ticket = 1; // to ensure thread reads blockid and threadid together
    // next thread is read to go
}