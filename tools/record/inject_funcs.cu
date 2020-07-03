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
                                                       uint32_t reg_num) {
    if (!pred) {
        return;
    }

    mem_access_t ma;

    ma.block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    ma.l_thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                      (threadIdx.y * blockDim.x) + threadIdx.x;

    /* Opportunistic Warp-level Programming */
    ma.active_mask = ballot(1);
    const int laneid = ma.l_thread_id & 31;  /* use get_laneid() is less efficient */
    const int leader = __ffs(ma.active_mask) - 1;

    /* read reg[num]'s value */
    uint32_t val = nvbit_read_reg(reg_num);

    /* collect reg value from other threads */
    for (int i = 0; i < 32; i++) {
        ma.vals[i] = __shfl_sync(ma.active_mask, val, i);
    }
    
    /* first active lane pushes information on the channel */
    if (leader == laneid) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}