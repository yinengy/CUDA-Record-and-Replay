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

extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint64_t addr,
                                                       uint64_t pchannel_dev) {
    if (!pred) {
        return;
    }

    int active_mask = ballot(1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = shfl(addr, i);
    }

    int4 cta = get_ctaid();
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev *channel_dev = (ChannelDev *)pchannel_dev;
        channel_dev->push(&ma, sizeof(mem_access_t));
    }
}