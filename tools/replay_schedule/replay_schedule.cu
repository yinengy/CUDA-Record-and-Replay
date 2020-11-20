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
#include <sstream>
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

/* deserialize vector */
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

int *d_schedule;
int *d_current_threadid;

uint64_t *d_schedule_counter;

void load_schedule() {
    std::ifstream file;
    file.open("schedule.bin");

    if (file.fail()) {
        std::cerr << "cannot open schedule.bin" << std::endl;
        exit(1);
    }

    cereal::BinaryInputArchive iarchive(file);
    std::vector<int> schedule;
    iarchive(schedule);
    schedule.push_back(-1); // indicates the end of the array

    cudaMalloc(&d_schedule, schedule.size() * sizeof(int));
    cudaMemcpy(d_schedule, schedule.data(), schedule.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_schedule_counter, sizeof(uint64_t));
    cudaMemset(d_schedule_counter, 0, sizeof(uint64_t));

    // first thread to schedule
    cudaMalloc(&d_current_threadid, sizeof(int));
    cudaMemcpy(d_current_threadid, schedule.data(), sizeof(int), cudaMemcpyHostToDevice);
}

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
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_current_threadid);

                nvbit_insert_call(instr, "release_lock", IPOINT_AFTER);
                /* predicate value */
                nvbit_add_call_arg_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_schedule_counter);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_schedule);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_current_threadid);
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
        } else {
            cudaFree(d_schedule);
            cudaFree(d_current_threadid);
            cudaFree(d_schedule_counter);
        }
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    load_schedule();
}