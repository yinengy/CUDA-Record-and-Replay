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

/* serialize vector */
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* definition of the mem_access_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* counter for cudaMemcpy and kernel launch */
int cudaMemcpy_input_count = 0;
int cudaMemcpy_output_count = 0;
int funcParams_count = 0;

/* 3D vector of [block_id, thread_id, num_access] to value of memory accesses */
std::vector<std::vector<std::vector<uint32_t>>> mem_val;

/* map of func_id to set of inst_id */
std::map<int, std::unordered_set<int>> data_race_log;

/* counter for func_id */
int func_counter = 0;


/* will save memory to files 
 * if is_input is 0, it means the mem is copied from device to host
 * otherwise, the mem is copied from host to device
 */
void dump_mem(const void *src, size_t ByteCount, int is_input) {
    
    char filename[25]; // large enough for a counter

    if (is_input) {
        cudaMemcpy_input_count++;
        sprintf(filename, "kernel_log/imem%d.bin", cudaMemcpy_input_count);
    } else {
        cudaMemcpy_output_count++;
        sprintf(filename, "kernel_log/omem%d.bin", cudaMemcpy_output_count);
    }
    
    std::ofstream file(filename, std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    file.write((char *) src, ByteCount);

    file.close();
}

void get_data_race_log() {
    std::ifstream file;
    file.open("datarace.txt");

    if (file.fail()) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    int func_id, inst_id;
    char comma;
    std::string line;
    while (std::getline(file, line)) {
        /* read type */
        std::istringstream iss(line);
        iss >> func_id;
        iss >> comma;
        iss >> inst_id;

        if (data_race_log.find(func_id) == data_race_log.end()) {
            data_race_log[func_id] = std::unordered_set<int>();
        } 
        
        data_race_log[func_id].insert(inst_id);
    }
}

/* save arguments to files
 * if it is not a pointer, its value with be saved
 * TODO: user defined type is not support
 */
void save_nonpointer_arguments(void **kernelParams,std::string func_sig) {
    /* get range of parameters */
    size_t begin = func_sig.find_first_of("(") + 1; // left parenthesis
    size_t end = func_sig.find_last_of(")"); // right parenthesis

    /* get string of parameters from string of signature */
    std::string func_params_str = func_sig.substr(begin, end - begin);

    /* remove whitespace */
    func_params_str.erase(remove_if(func_params_str.begin(), func_params_str.end(), isspace), func_params_str.end());

    /* split by comma */
    std::vector<std::string> func_params;
    std::string delim = ",";
    begin = 0;
    end = func_params_str.find(delim);
    while (end != std::string::npos) {
        func_params.push_back(func_params_str.substr(begin, end - begin));
        begin = end + delim.length();
        end = func_params_str.find(delim, begin);
    }
    func_params.push_back(func_params_str.substr(begin, end));

    /* check types of parameters
     * if it is not a pointer (pointer will be processed separately)
     * save its index, type and value to file
     * if it is a pointer, its value will be marked as "POINTER"
     */ 
    funcParams_count++;
    char filename[25];
    sprintf(filename, "kernel_log/param%d.txt", funcParams_count);
    std::ofstream file;
    file.open(filename);

    if (file.fail()) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    std::string type;
    for (size_t i = 0; i < func_params.size(); i++) {
        type = func_params[i];
        if (type.empty()) { 
            // the function has no parameter
            break;
        } 

        file << i << "," << type << ",";

        /* cast kernelParams based on parameter type
         * it will gives the argument of the kernel function
         * refer to https://en.wikipedia.org/wiki/C_data_types
         */
        if (type.find('*') != std::string::npos) {
            // the parameter is a pointer
            file << "POINTER"; 
        } else if (type == "char") {
            file << ((char **) kernelParams)[i][0]; 
        } else if (type == "signedchar") {
            file << ((signed char **) kernelParams)[i][0]; 
        } else if (type == "unsignedchar") {
            file << ((unsigned char **) kernelParams)[i][0]; 
        } else if (type == "short" ||
                   type == "shortint" ||
                   type == "signedshort" ||
                   type == "signedshortint") {
            // signed short
            file << ((short **) kernelParams)[i][0]; 
        } else if (type == "unsignedshort" ||
                   type == "unsigned short int") {
            // unsigned short
            file << ((unsigned short **) kernelParams)[i][0]; 
        } else if (type == "int" ||
                   type == "signed" ||
                   type == "signedint") {
            // signed int
            file << ((int **) kernelParams)[i][0]; 
        } else if (type == "unsigned" ||
                   type == "unsignedint") {
            // unsigned int
            file << ((unsigned **) kernelParams)[i][0]; 
        } else if (type == "long" ||
                   type == "longint" ||
                   type == "signedlong" ||
                   type == "signedlongint") {
            // signed long
            file << ((long **) kernelParams)[i][0]; 
        } else if (type == "unsignedlong" ||
                   type == "unsignedlongint") {
            // unsigned long
            file << ((unsigned long **) kernelParams)[i][0]; 
        } else if (type == "longlong" ||
                   type == "longlongint" ||
                   type == "signedlonglong" ||
                   type == "signedlonglongint") {
            // signed long long
            file << ((long long **) kernelParams)[i][0]; 
        } else if (type == "unsignedlonglong" ||
                   type == "unsignedlonglongint") {
            // unsigned long long
            file << ((unsigned long long **) kernelParams)[i][0]; 
        } else if (type == "float") {
            // float
            file << ((float **) kernelParams)[i][0]; 
        } else if (type == "double") {
            // double
            file << ((double **) kernelParams)[i][0]; 
        } else if (type == "longdouble") {
            // long double
            file << ((long double **) kernelParams)[i][0]; 
        } else {
            // TODO: implement more types
            file << "UNKNOWN";
        }

        file << "\n";
    }

    file.close();
}

__global__ void flush_channel() {
    /* push memory access with negative block id to communicate the kernel is
     * completed */
    mem_access_t ma;
    ma.block_id = -1;
    channel_dev.push(&ma, sizeof(mem_access_t));

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
                mem_access_t *ma =
                    (mem_access_t *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ma->block_id == -1) {
                    recv_thread_receiving = false;
                    break;
                }

                /* save memory value to vector */
                for (int i = 0; i < 32; i++) {   
                    if ((ma->active_mask >> i) & 1) { // if the thread is active
                        mem_val[ma->block_id][ma->l_thread_id + i].push_back(ma->vals[i]);
                    }
                }

                num_processed_bytes += sizeof(mem_access_t);
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

        /* insert function name into the vector*/
        /* its index is func_id */
        int func_id = func_counter;
        func_counter++;

        int inst_id = 0;

        /* check if there is data race in this function */
        if (data_race_log.find(func_id) == data_race_log.end()) {
            continue; // no data race, go to next function
        }

        std::unordered_set<int> data_race_inst = data_race_log[func_id]; 

        /* iterate on all the static instructions in the function */
        for (auto instr : instrs) {
            /* check if there is data race in this instruction */
            if (data_race_inst.find(inst_id) == data_race_inst.end()) {
                inst_id++;
                continue; // no data race, go to next inst
            }

            std::string opcode = instr->getOpcodeShort();

            int reg_idx = 0;

            if (opcode == "LDG") {  // load
                /* it should has two operands */
                assert(instr->getNumOperands() == 2);

                /* first reg is the destination */
                const Instr::operand_t *dst_reg = instr->getOperand(0);
                assert(dst_reg->type == Instr::operandType::REG);
                reg_idx = dst_reg->u.reg.num;

                /* after this instruction so dst reg has the value */
                nvbit_insert_call(instr, "record_mem_val", IPOINT_AFTER);
            } else if (opcode == "STG") {  // store
                /* it should has two operands */
                assert(instr->getNumOperands() == 2);

                /* second reg is the source */
                const Instr::operand_t *src_reg = instr->getOperand(1);
                assert(src_reg->type == Instr::operandType::REG);
                reg_idx = src_reg->u.reg.num;

                /* before this instruction so src reg is not changed */
                nvbit_insert_call(instr, "record_mem_val", IPOINT_BEFORE);
            } else {
                // TODO: support more load and store instructions
                std::cerr << opcode << ": unhandled instruction.\n";
                instr->printDecoded();
                exit(1);
            }

            /* predicate value */
            nvbit_add_call_arg_pred_val(instr);
            /* add pointer to channel_dev*/
            nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
            /* add reg index */
            nvbit_add_call_arg_const_val32(instr, reg_idx, false);
            

            inst_id++;
        }
    }
}

/* This is triggered every time a cudaMemcpy is called */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_flag) return;

    /* if cudaMemcpy from Device to Host
     * it is treated as output of the kernel
     * should be trigger at the end of cuMemcpy
     * so that dstHost has the memory ready
     */
    if ((cbid == API_CUDA_cuMemcpyDtoH_v2) && is_exit)  {
        /* get parameters */
        cuMemcpyDtoH_v2_params *p = (cuMemcpyDtoH_v2_params *)params;

        dump_mem(p->dstHost, p->ByteCount, 0);
    } else if ((cbid == API_CUDA_cuMemcpyHtoD_v2) && is_exit)  {
        cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;

        dump_mem(p->srcHost, p->ByteCount, 1);
    } else if ((cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel)) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        if (!is_exit) {
            /* get kernel function signature */    
            std::string func_sig(nvbit_get_func_name(ctx, p->f));

            /* record kernel arguments */
            save_nonpointer_arguments(p->kernelParams, func_sig);

            /* instrument this kernel */
            instrument_function_if_needed(ctx, p->f);

            nvbit_enable_instrumented(ctx, p->f, true);

            /* reserve space */
            int num_block = p->gridDimX * p->gridDimY * p->gridDimZ;
            int num_thread = p->blockDimX * p->blockDimY * p->blockDimZ;
            mem_val.empty();
            mem_val.resize(num_block, std::vector<std::vector<uint32_t>>(num_thread));

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
            char filename[25];
            sprintf(filename, "kernel_log/vmem%d.bin", funcParams_count);
            std::ofstream file;
            file.open(filename);

            if (file.fail()) {
                std::cerr << strerror(errno) << "failed to open file.\n";
                exit(1);
            }

            {
                cereal::BinaryOutputArchive oarchive(file);

                oarchive(mem_val);
            }
        }
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    /* all log files will put into this directory */
    mkdir("kernel_log", 0777);

    /* initialize channel */
    recv_thread_started = true;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);

    /* read data race log */
    get_data_race_log();
}

void nvbit_at_ctx_term(CUcontext ctx) {
    if (recv_thread_started) {
        recv_thread_started = false;
        pthread_join(recv_thread, NULL);
    }
}