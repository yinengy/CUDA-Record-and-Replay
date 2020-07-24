/* 
 * An NVBit tool, which will detect conflict memory access in the kernel.
 * The raw output will be processed by a Pytyhon script
 *
 * The tool is based on thetool (mem_trace) in nvbit_release
 * the original code is modified and extended to support the use of detecting data races
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

extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       int func_id,
                                                       int inst_id,
                                                       uint64_t addr,
                                                       uint64_t pchannel_dev,                                                  
                                                       int32_t is_shared_memory,
                                                       int32_t is_load,
                                                       uint64_t psyn_ops_counter) {
    //NOTE: the way to use pred doesn't work on atomic operations
    if (!pred) {
        return;
    }

    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int active_mask = ballot(1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = shfl(addr, i);
    }

    ma.block_id = block_id;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;
    ma.func_id = func_id;
    ma.inst_id = inst_id;
    ma.is_shared_memory = is_shared_memory;
    ma.is_load = is_load;
    ma.SFR_id = ((int *)psyn_ops_counter)[block_id];

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}

// use this when it is generic memory
// need to convert address at runtime
extern "C" __device__ __noinline__ void instrument_gen_mem(int pred, int opcode_id,
                                                       int func_id,
                                                       int inst_id,
                                                       uint64_t addr,
                                                       uint64_t pchannel_dev,                                                  
                                                       uint64_t shared_mem_base,
                                                       uint64_t local_mem_base,
                                                       int32_t is_load,
                                                       uint64_t psyn_ops_counter) {
    //NOTE: the way to use pred doesn't work on atomic operations
    if (!pred) {
        return;
    }

    int block_id = blockIdx.x + blockIdx.y * gridDim.x +
                   gridDim.x * gridDim.y * blockIdx.z;

    int active_mask = ballot(1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = shfl(addr, i);
    }

    ma.block_id = block_id;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;
    ma.func_id = func_id;
    ma.inst_id = inst_id;
    ma.is_load = is_load;
    ma.SFR_id = ((int *)psyn_ops_counter)[block_id];

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        // shmem range is [shared_mem_base, shared_mem_base+16MB)
        if (shared_mem_base <= ma.addrs[first_laneid] && ma.addrs[first_laneid] < (shared_mem_base + (1 << 24))) {
            for (int i = 0; i < 32; i++) {
                ma.addrs[i] -= ma.addrs[i] ? shared_mem_base : 0;
            }
            ma.is_shared_memory = 1;
        // local mem range is [local_mem_base, local_mem_base+16MB)
        } else if (local_mem_base <= ma.addrs[first_laneid] && ma.addrs[first_laneid] < (local_mem_base + (1 << 24))) {
            for (int i = 0; i < 32; i++) {
                // 0 memory will be ignored by the detector
                // so set local memory address to 0
                ma.addrs[i] = 0;
            }
            ma.is_shared_memory = 0;
        } else {
            ma.is_shared_memory = 0;
        }
        
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}


/* will increment syn_ops_counter by one 
 * the counter will be used as an ID for code regions between synchronization operations
 * which are also refered as synchronization-free regions (SFRs)
 */
extern "C" __device__ __noinline__ void instrument_syn(uint64_t psyn_ops_counter) {
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
       ((int *)psyn_ops_counter)[block_id] += 1;
    }

    // wait for other threads so can gurantee that the counter is updated after this function call
    __syncthreads();
    
}
