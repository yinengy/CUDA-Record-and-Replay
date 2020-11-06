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

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

int *d_schedule_blockid;
int *d_schedule_threadid;
int *d_current_blockid;
int *d_current_threadid;
int *d_has_new_ticket;

uint64_t *d_schedule_counter;

void load_schedule() {
    std::ifstream file;
    file.open("schedule.txt");

    if (file.fail()) {
        std::cerr << "cannot open schedule.txt." << std::endl;
        exit(1);
    }

    std::vector<int> schedule_blockid;
    std::vector<int> schedule_threadid;

    int block_id, thread_id;
    char comma;
    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '#' || line[1] != '?' || line[2] != '#') {
            continue;
        }

        /* format: block id, thread id */
        std::istringstream iss(line.substr(3));
        iss >> block_id;
        iss >> comma;
        iss >> thread_id;
        

        schedule_blockid.push_back(block_id);
        schedule_threadid.push_back(thread_id);
    }

    // insert canary at the end to avoid index out of bound
    schedule_blockid.push_back(-1);
    schedule_threadid.push_back(-1);

    cudaMalloc(&d_schedule_blockid, schedule_blockid.size() * sizeof(int));
    cudaMemcpy(d_schedule_blockid, schedule_blockid.data(), schedule_blockid.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_schedule_threadid, schedule_threadid.size() * sizeof(int));
    cudaMemcpy(d_schedule_threadid, schedule_threadid.data(), schedule_threadid.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_schedule_counter, sizeof(uint64_t));
    cudaMemset(d_schedule_counter, 0, sizeof(uint64_t));

    // first thread to schedule
    cudaMalloc(&d_current_blockid, sizeof(int));
    cudaMemcpy(d_current_blockid, schedule_blockid.data(), sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_current_threadid, sizeof(int));
    cudaMemcpy(d_current_threadid, schedule_threadid.data(), sizeof(int), cudaMemcpyHostToDevice);

    int val = 1;
    cudaMalloc(&d_has_new_ticket, sizeof(int));
    cudaMemcpy(d_has_new_ticket, &val, sizeof(int), cudaMemcpyHostToDevice);
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
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_has_new_ticket);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_current_blockid);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_current_threadid);

                nvbit_insert_call(instr, "release_lock", IPOINT_AFTER);
                /* predicate value */
                nvbit_add_call_arg_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_has_new_ticket);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_schedule_counter);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_schedule_blockid);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_schedule_threadid);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) d_current_blockid);
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
            cudaFree(d_schedule_blockid);
            cudaFree(d_schedule_threadid);
            cudaFree(d_current_blockid);
            cudaFree(d_current_threadid);
            cudaFree(d_has_new_ticket);
            cudaFree(d_schedule_counter);
        }
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    load_schedule();
}