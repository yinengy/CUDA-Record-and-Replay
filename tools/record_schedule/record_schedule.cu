/* 
 *
 * A NVBit tool, which will log the input and output of kernel function
 * input is defined as memory copied from host to device and arguments of kernel
 * output is defined as memory copied from device to host
 *
 * And it will also record memory access of each thread for selected instructions.
 * code that related to the use of channel credit to nvbit sample tool "mem_trace"
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cerrno>
#include <algorithm>
#include <sys/stat.h>
#include <assert.h>
#include <unordered_set>
#include <stdint.h>
#include <map>

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* mutex */
#include "concurrency.cpp"

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

mutex* mtx;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }
        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            std::string opcode = instr->getOpcodeShort();

            if (opcode == "LDG" || opcode == "STG") {
                nvbit_insert_call(instr, "acquire_lock", IPOINT_BEFORE);
                /* predicate value */
                nvbit_add_call_arg_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) mtx);

                nvbit_insert_call(instr, "release_lock", IPOINT_AFTER);
                /* predicate value */
                nvbit_add_call_arg_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) mtx);
            }
        }
    }
}

/* This is triggered every time a cudaMemcpy is called */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    
    if ((cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel)) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        if (!is_exit) {
            /* instrument this kernel */
            instrument_function_if_needed(ctx, p->f);

            nvbit_enable_instrumented(ctx, p->f, true);
        } 
    }
}


void nvbit_at_ctx_init(CUcontext ctx) {
    mtx = make_<mutex>();
}