/* 
 * definition of mem_access_t
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
    int block_id;
    int warp_id;
    int func_id;
    int inst_id;
    int is_shared_memory;
    int is_load;
    int SFR_id;  // id of synchronization-free regions
    uint64_t addrs[32];
} mem_access_t;

#endif /* COMMON_H */
