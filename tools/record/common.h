#include <stdint.h>

typedef struct {
    int block_id;
    int l_thread_id;
    uint32_t val;
} mem_access_t;