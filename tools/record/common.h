#include <stdint.h>

typedef struct {
    int block_id;
    int l_thread_id;
    int active_mask;
    uint32_t vals[32];
} mem_access_t;