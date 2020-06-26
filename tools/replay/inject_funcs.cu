/* 
 * Instrument functions for replay.cu
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void replace_mem_val(int32_t pred,
                                                        uint64_t mem_val_ptr,
                                                        uint64_t mem_counter_ptr,
                                                        uint32_t num_access,
                                                        uint32_t num_thread,
                                                        uint32_t reg_num) {
    if (!pred) {
        return;
    }
    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

    uint32_t *mem_val = (uint32_t *) mem_val_ptr;
    uint32_t *counter = (uint32_t *) mem_counter_ptr;

    uint32_t current_counter = counter[block_id * num_thread + l_thread_id];
    uint32_t reg_val = mem_val[block_id * num_access * num_thread + 
                               l_thread_id * num_access + current_counter];

    /* increment the counter */
    counter[block_id * num_thread + num_thread]++;

    nvbit_write_reg(reg_num, reg_val);

    printf("%d ", reg_val);
}