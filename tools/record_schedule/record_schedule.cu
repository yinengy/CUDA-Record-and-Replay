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
#include <vector>

/* mutex */
#include "concurrency.cpp"

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* serialize vector */
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

/* nvbit utility functions */
#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* logged schedule */
std::vector<int> schedule;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

mutex* mtx;

__global__ void flush_channel() {
    /* push negative block id to communicate the kernel is
     * completed */
    int thread_id = -1;
    channel_dev.push(&thread_id, sizeof(int));

    /* flush channel */
    channel_dev.flush();
}

void *recv_thread_fun(void *) {
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;
        if (recv_thread_receiving &&
            (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
                0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                int *id =
                    (int *)&recv_buffer[num_processed_bytes];

                /* when we get negative value it means the kernel has completed
                 */
                if (*id == -1) {
                    recv_thread_receiving = false;
                    break;
                }

                /* save memory value to vector */
                schedule.push_back(*id);

                num_processed_bytes += sizeof(int);
            }
        }
    }
    free(recv_buffer);
    return NULL;
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

            if (opcode == "LDG" || opcode == "STG" || opcode == "RED" || opcode == "ATOM" || opcode == "ATOMS") {
                nvbit_insert_call(instr, "acquire_lock", IPOINT_BEFORE);
                /* predicate value */
                nvbit_add_call_arg_pred_val(instr);
                nvbit_add_call_arg_const_val64(instr, (uint64_t) mtx);
                /* add pointer to channel_dev*/
                nvbit_add_call_arg_const_val64(instr, (uint64_t) &channel_dev);

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
    if (skip_flag) return;

    if ((cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel)) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        if (!is_exit) {
            /* instrument this kernel */
            instrument_function_if_needed(ctx, p->f);

            nvbit_enable_instrumented(ctx, p->f, true);

            recv_thread_receiving = true;
        } else {
            /* make sure current kernel is completed */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            /* make sure we prevent re-entry on the nvbit_callback when issuing
             * the flush_channel kernel */
            skip_flag = true;

            /* issue flush of channel so we are sure all the memory accesses
             * have been pushed */
            flush_channel<<<1, 1>>>();
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            /* unset the skip flag */
            skip_flag = false;

            /* wait here until the receiving thread has not finished with the
             * current kernel */
            while (recv_thread_receiving) {
                pthread_yield();
            }

            /* serilize vector */
            char filename[40];
            sprintf(filename, "schedule.bin");
            std::ofstream file;
            file.open(filename);

            if (file.fail()) {
                std::cerr << strerror(errno) << "failed to open file.\n";
                exit(1);
            }

            {
                cereal::BinaryOutputArchive oarchive(file);

                oarchive(schedule);
            }
        }
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    mtx = make_<mutex>();

    /* initialize channel */
    recv_thread_started = true;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    if (recv_thread_started) {
        recv_thread_started = false;
        pthread_join(recv_thread, NULL);
    }
}
