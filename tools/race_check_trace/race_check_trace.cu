/* 
 * An NVBit tool, which will detect conflict memory access in the kernel.
 * The raw output will be processed by a Pytyhon script
 *
 * The tool is based on thetool (mem_trace) in nvbit_release
 * the original code is modified and extended to support the use of detecting data races
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* synchronization operation counter, updated by the GPU threads */
int *syn_ops_counter = 0;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* vector of func_name, index by func_id */
std::vector<std::string> id_to_func_name;

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

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
        const char*func_name = nvbit_get_func_name(ctx, f);
        if (verbose) {
            printf("Inspecting function %s at address 0x%lx\n",
                   func_name, nvbit_get_func_addr(f));
        }

        /* insert function name into the vector*/
        /* its index is func_id */
        int func_id = id_to_func_name.size();
        id_to_func_name.push_back(std::string(func_name));

        uint32_t inst_id = 0;

        /* tell python script the content of a function begins here*/
        printf("\n#func_begin#%s\n", func_name);

        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            /* print SASS, which is used by python script*/
            printf("\n#SASS#%s\n", instr->getSass());

            // check syn op first
            const char *shortOpcode = instr->getOpcodeShort();

            // to see if it is a synchronization operation
            bool is_syn_op = (strcmp(shortOpcode, "RED") == 0)   || // Atomic Memory Reduction Operation
                             (strcmp(shortOpcode, "ATOM") == 0)  || // Atomic Operation on generic Memory
                             (strcmp(shortOpcode, "ATOMS") == 0) || // Atomic Operation on Shared Memory
                             (strcmp(shortOpcode, "BAR") == 0) ||   // Barrier (e.g. __syncthreads())
                             (strcmp(shortOpcode, "MEMBAR") == 0);  // Memory Barrier
            if (is_syn_op) {
                // instrument after the syn op so all thread are ready
                // then we can just update the counter by one
                nvbit_insert_call(instr, "instrument_syn", IPOINT_AFTER);
                nvbit_add_call_arg_const_val64(instr, (uint64_t)syn_ops_counter);
                inst_id++;
                continue; //skip the rest
            } 

            if ((instr->getMemOpType()!=Instr::memOpType::GLOBAL
                    && instr->getMemOpType()!=Instr::memOpType::SHARED && instr->getMemOpType()!=Instr::memOpType::GENERIC)) {
                inst_id++;
                continue;
            }

            if (opcode_to_id_map.find(instr->getOpcode()) ==
                opcode_to_id_map.end()) {
                int opcode_id = opcode_to_id_map.size();
                opcode_to_id_map[instr->getOpcode()] = opcode_id;
                id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
            }

            int opcode_id = opcode_to_id_map[instr->getOpcode()];

            bool is_GENERIC = instr->getMemOpType()==Instr::memOpType::GENERIC;

            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++) {
                /* get the operand "i" */
                const Instr::operand_t *op = instr->getOperand(i);
            
                if (op->type == Instr::operandType::MREF) {
                    /* insert call to the instrumentation function with its
                     * arguments */
                    if (is_GENERIC) {
                        nvbit_insert_call(instr, "instrument_gen_mem", IPOINT_BEFORE);
                    } else {
                        nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                    }
                    /* predicate value */
                    nvbit_add_call_arg_pred_val(instr);
                    /* opcode id */
                    nvbit_add_call_arg_const_val32(instr, opcode_id);
                    /* func id */
                    nvbit_add_call_arg_const_val32(instr, func_id);    
                    /* inst id */
                    nvbit_add_call_arg_const_val32(instr, inst_id); 
                    /* memory reference 64 bit address */
                    nvbit_add_call_arg_mref_addr64(instr);
                    /* add pointer to channel_dev*/
                    nvbit_add_call_arg_const_val64(instr,
                                                   (uint64_t)&channel_dev);
                    /* insert base address to predict Generic pointer at run time */
                    if (is_GENERIC) {
                        nvbit_add_call_arg_const_val64(instr, nvbit_get_shmem_base_addr(ctx));
                        nvbit_add_call_arg_const_val64(instr, nvbit_get_local_mem_base_addr(ctx));
                    } else {
                        nvbit_add_call_arg_const_val32(instr, instr->getMemOpType()==Instr::memOpType::SHARED);
                    }                               
                    nvbit_add_call_arg_const_val32(instr, instr->isLoad());
                    nvbit_add_call_arg_const_val64(instr, (uint64_t)syn_ops_counter);

                }
            }
            inst_id++;
        }

        /* tell python script the content of a function ends here*/
        printf("\n#func_end#\n");
    }
}

__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    mem_access_t ma;
    ma.block_id = -1;
    channel_dev.push(&ma, sizeof(mem_access_t));

    /* flush channel */
    channel_dev.flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_flag) return;

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        if (!is_exit) {
            /* allocate syn_ops_counter for each block */
            int num_block = p->gridDimX * p->gridDimY * p->gridDimZ;
            CUDA_SAFECALL(cudaMalloc(&syn_ops_counter, num_block * sizeof(int)));
            CUDA_SAFECALL(cudaMemset(syn_ops_counter, 0, num_block * sizeof(int)));

            instrument_function_if_needed(ctx, p->f);

            nvbit_enable_instrumented(ctx, p->f, true);

            recv_thread_receiving = true;

        } else {
            /* make sure current kernel is completed */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            /* free allocated cuda memory */
            CUDA_SAFECALL(cudaFree(syn_ops_counter));

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

            printf("\n#kernelends#\n");
        }
    }
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
                mem_access_t *ma =
                    (mem_access_t *)&recv_buffer[num_processed_bytes];

                /* when we get this block_id it means the kernel has completed
                 */
                if (ma->block_id == -1) {
                    recv_thread_receiving = false;
                    break;
                }
                
                /* "#{ld/st}#" is used to grep */
                if (ma->is_load) {
                    for (int i = 0; i < 32; i++) {
                        if (ma->addrs[i] == 0) continue;
                        printf("\n#ld#%d,%d,%d,%d,%d,%d,%d,0x%016lx\n",
                            ma->is_shared_memory,  ma->block_id, ma->warp_id, i, ma->func_id, ma->inst_id,
                            ma->SFR_id, ma->addrs[i]);
                    }
                } else {
                    for (int i = 0; i < 32; i++) {
                        if (ma->addrs[i] == 0) continue;
                        printf("\n#st#%d,%d,%d,%d,%d,%d,%d,0x%016lx\n",
                            ma->is_shared_memory,  ma->block_id, ma->warp_id, i, ma->func_id, ma->inst_id,
                            ma->SFR_id, ma->addrs[i]);
                    }
                }
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }
    free(recv_buffer);
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
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
