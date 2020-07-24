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

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
    int block_id;
    int warp_id;
    int opcode_id;
    int func_id;
    int inst_id;
    int is_shared_memory;
    int is_load;
    int SFR_id;  // id of synchronization-free regions
    uint64_t addrs[32];
} mem_access_t;
