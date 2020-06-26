/* 
 * Instrument functions for record.cu
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

#include "nvbit_reg_rw.h"

extern "C" __device__ __noinline__ void record_mem_val(int32_t pred,
                                                       uint64_t pchannel_dev,
                                                       uint32_t val) {
    if (!pred) {
        return;
    }

    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

    mem_access_t ma;

    ma.block_id = block_id;
    ma.l_thread_id = l_thread_id;
    ma.val = nvbit_read_reg(0);
    
    ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
    channel_dev->push(&ma, sizeof(mem_access_t));
}