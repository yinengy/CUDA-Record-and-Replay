/* 
 * Message type used in channel
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <stdint.h>

typedef struct {
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int warp_id;
    int opcode_id;
    uint64_t addrs[32];
} mem_access_t;