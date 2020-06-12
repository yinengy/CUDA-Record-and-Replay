/* 
 *
 * A NVBit tool, which will dump content of cudaMemcpy from Device to Host
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */
#include <stdio.h>

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

int cudaMemcpy_count = 0;

void dump_mem(void *dstHost, size_t ByteCount) {
    cudaMemcpy_count++;

    char filename[20]; // large enough for a counter
    sprintf(filename, "mem%d.bin", cudaMemcpy_count);
    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        printf("failed to open file.\n");
    }

    if (fwrite(dstHost, ByteCount, 1, fp) != 1) {
        printf("failed to write file.\n");
    }
}

/* This is triggered every time a cudaMemcpy is called */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* if cudaMemcpy from Device to Host */
    /* it is treated as output of the kernel */
    if ((cbid == API_CUDA_cuMemcpyDtoH_v2) && is_exit)  {
        /* get parameters */
        cuMemcpyDtoH_params *p = (cuMemcpyDtoH_params *)params;

        dump_mem(p->dstHost, p->ByteCount);
    }
}