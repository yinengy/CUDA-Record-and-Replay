/* 
 *
 * A NVBit tool, which will replay the program by examine the log file generated in record phase
 * input is defined as memory copied from host to device and arguments of kernel
 * output is defined as memory copied from device to host
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

/* counter for cudaMemcpy and kernel launch */
int cudaMemcpy_input_count = 0;
int cudaMemcpy_output_count = 0;
int funcParams_count = 0;

void *recorded_mem;
const void *origin_srcHost;

/* pointer to values of memory accessed */
uint32_t *d_mem_val;

/* pointer to counter of mem_val */
uint32_t *d_mem_counter;

/* used to index into flat array */
uint32_t num_access;
uint32_t num_thread;

/* skip flag used to avoid re-entry on the nvbit_callback */
bool skip_flag = false;

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

/* map of func_id to set of inst_id */
std::map<int, std::unordered_set<int>> data_race_log;

/* counter for func_id */
int func_counter = 0;

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

/* read memory content recorded in record phase from files 
 * if is_input is 0, it means the mem is copied from device to host
 * otherwise, the mem is copied from host to device
 * a memory of BtyeCount will be allocated and returned
 */
void *get_recorded_mem(size_t ByteCount, int is_input) {
    
    char filename[40]; // large enough for a counter

    if (is_input) {
        cudaMemcpy_input_count++;
        sprintf(filename, "kernel_log/imem%d.bin", cudaMemcpy_input_count);
    } else {
        cudaMemcpy_output_count++;
        sprintf(filename, "kernel_log/omem%d.bin", cudaMemcpy_output_count);
    }
    
    void *buffer = malloc(ByteCount);

    std::ifstream file(filename, std::ios::in | std::ios::binary);
        
    if (!file.is_open()) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    file.read((char *) buffer, ByteCount);

    if (!file) {
        std::cerr << "only " << file.gcount() << " could be read from " << filename << std::endl;
        exit(1);
    }

    file.close();

    return buffer;
}

/* compare ptr with output in record phase */
void compare_mem(const void *ptr, size_t ByteCount) {
    void *to_compare = get_recorded_mem(ByteCount, 0);

    int is_equal = memcmp(ptr, to_compare, ByteCount);

    if (is_equal != 0) {
        std::cerr << cudaMemcpy_output_count << "th output doesn't match!\n";
    }

    free(to_compare);
}


/* load arguments from log files
 * if it is not a pointer, its value with be saved to kernelParams
 * TODO: user defined type is not support
 */
void replace_nonpointer_arguments(void **kernelParams) {
    /* open log file */
    funcParams_count++;
    char filename[40];
    sprintf(filename, "kernel_log/param%d.txt", funcParams_count);
    std::ifstream file;
    file.open(filename);

    if (file.fail()) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    int i;
    std::string trash, type, line;
    while (std::getline(file, line)) {
        /* read type */
        std::istringstream iss(line);
        iss >> i;
        std::getline(iss, trash, ',');
        std::getline(iss, type, ',');

        /* cast kernelParams based on parameter type and assign value to it
         * it will gives the argument of the kernel function
         * refer to https://en.wikipedia.org/wiki/C_data_types
         */
        if (type.find('*') != std::string::npos) {
            // the parameter is a pointer
            continue;
        } else if (type == "char") {
            char value;
            iss >> value;
            *(((char **) kernelParams)[i]) = value;
        } else if (type == "signedchar") {
            signed char value;
            iss >> value;
            *(((signed char **) kernelParams)[i]) = value;
        } else if (type == "unsignedchar") {
            unsigned char value;
            iss >> value;
            *(((unsigned char **) kernelParams)[i]) = value;
        } else if (type == "short" ||
                   type == "shortint" ||
                   type == "signedshort" ||
                   type == "signedshortint") {
            // signed short
            short value;
            iss >> value;
            *(((short **) kernelParams)[i]) = value;
        } else if (type == "unsignedshort" ||
                   type == "unsigned short int") {
            // unsigned short
            unsigned short value;
            iss >> value;
            *(((unsigned short **) kernelParams)[i]) = value;
        } else if (type == "int" ||
                   type == "signed" ||
                   type == "signedint") {
            // signed int
            int value;
            iss >> value;
            *(((int **) kernelParams)[i]) = value;
        } else if (type == "unsigned" ||
                   type == "unsignedint") {
            // unsigned int
            unsigned value;
            iss >> value;
            *(((unsigned **) kernelParams)[i]) = value;
        } else if (type == "long" ||
                   type == "longint" ||
                   type == "signedlong" ||
                   type == "signedlongint") {
            // signed long
            long value;
            iss >> value;
            *(((long **) kernelParams)[i]) = value;
        } else if (type == "unsignedlong" ||
                   type == "unsignedlongint") {
            // unsigned long
            unsigned long value;
            file >> value;
            *(((unsigned long **) kernelParams)[i]) = value;
        } else if (type == "longlong" ||
                   type == "longlongint" ||
                   type == "signedlonglong" ||
                   type == "signedlonglongint") {
            // signed long long
            signed long long value;
            iss >> value;
            *(((signed long long **) kernelParams)[i]) = value;
        } else if (type == "unsignedlonglong" ||
                   type == "unsignedlonglongint") {
            // unsigned long long
            unsigned long long value;
            iss >> value;
            *(((unsigned long long **) kernelParams)[i]) = value;
        } else if (type == "float") {
            // float
            float value;
            iss >> value;
            *(((float **) kernelParams)[i]) = value;
        } else if (type == "double") {
            // double
            double value;
            iss >> value;
            *(((double **) kernelParams)[i]) = value;
        } else if (type == "longdouble") {
            // long double
            long double value;
            iss >> value;
            *(((long double **) kernelParams)[i]) = value;
        } else {
            // TODO: implement more types
            continue;
        }
    }

    file.close();
}

/* get memory access record from files 
 * will return a *large* array
 */
void get_mem_val() {
    /* deserialize vector */
    char filename[40];
    sprintf(filename, "kernel_log/vmem%d.bin", funcParams_count);
    std::ifstream file;
    file.open(filename);

    if (file.fail()) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    uint32_t *h_mem_val;
    int size = 0;

    {
        cereal::BinaryInputArchive iarchive(file);
        std::vector<std::vector<std::vector<uint32_t>>> mem_val_vec;
        iarchive(mem_val_vec);

        /* convert 3d vector to a flat array */
        num_access = 0;
        for (size_t x = 0; x < mem_val_vec.size(); x++) {
            for (size_t y = 0; y < mem_val_vec[0].size(); y++) {
                if (mem_val_vec[x][y].size() > num_access) {
                    num_access = mem_val_vec[x][y].size();
                }
            }
        }

        if (num_access == 0) {
            return; // no record
        }

        size_t num_block = mem_val_vec.size();
        num_thread = mem_val_vec[0].size();

        /* reserve space, waste some memory at dim z */
        int size_counter = sizeof(uint32_t) * num_block * num_thread;
        size = size_counter * num_access;
        h_mem_val = (uint32_t *) malloc(size);

        /* malloc a zero array on GPU */
        cudaMalloc(&d_mem_counter, size_counter);
        cudaMemset(d_mem_counter, 0, size_counter);

        /* copy data from vector to array */
        for (size_t x = 0; x < num_block; x++) {
            for (size_t y = 0; y < num_thread; y++) {
                memcpy(h_mem_val + x * num_thread * num_access + y * num_access, 
                        (uint32_t *) mem_val_vec[x][y].data(), sizeof(uint32_t) * mem_val_vec[x][y].size());
            }
        }
    }  // mem_val_vec out of scope here

    /* copy array from host to device */
    cudaMalloc(&d_mem_val, size);
    cudaMemcpy(d_mem_val, h_mem_val, size, cudaMemcpyHostToDevice);
    free(h_mem_val);
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
                nvbit_insert_call(instr, "replace_mem_val", IPOINT_AFTER);
            } else if (opcode == "STG") {  // store
                /* it should has two operands */
                assert(instr->getNumOperands() == 2);

                /* second reg is the source */
                const Instr::operand_t *src_reg = instr->getOperand(1);
                assert(src_reg->type == Instr::operandType::REG);
                reg_idx = src_reg->u.reg.num;

                /* before this instruction so src reg is not changed */
                nvbit_insert_call(instr, "replace_mem_val", IPOINT_BEFORE);
            } else {
                // TODO: support more load and store instructions
                std::cerr << opcode << ": unhandled instruction.\n";
                instr->printDecoded();
                exit(1);
            }

            /* predicate value */
            nvbit_add_call_arg_pred_val(instr);
            /* add mem_val pointer */
            nvbit_add_call_arg_const_val64(instr, (uint64_t) d_mem_val, false);
            /* add mem_counter pointer */
            nvbit_add_call_arg_const_val64(instr, (uint64_t) d_mem_counter, false);
            /* add num block */
            nvbit_add_call_arg_const_val32(instr, num_access, false);
            /* add num thread */
            nvbit_add_call_arg_const_val32(instr, num_thread, false);
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

    if ((cbid == API_CUDA_cuMemcpyDtoH_v2) && is_exit)  {
        /* dump output, so can compare it with the output in record phase */
        cuMemcpyDtoH_v2_params *p = (cuMemcpyDtoH_v2_params *)params;

        compare_mem(p->dstHost, p->ByteCount);
    } else if (cbid == API_CUDA_cuMemcpyHtoD_v2)  {
        cuMemcpyHtoD_v2_params *p = (cuMemcpyHtoD_v2_params *)params;

        if (!is_exit) {
            /* it is should be trigger at the begin of cuMemcpy
             * so that the memory content can be replace 
             * by the content in record phase */
            recorded_mem = get_recorded_mem(p->ByteCount, 1);
            origin_srcHost = p->srcHost;
            p->srcHost = recorded_mem;
        } else {
            /* after cuMemcpy, free the memory allocated by get_recorded_mem */
            p->srcHost = origin_srcHost;
            free(recorded_mem);
            recorded_mem = nullptr;
            origin_srcHost = nullptr;
        }
    } else if ((cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel)) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        if (!is_exit) {
            replace_nonpointer_arguments(p->kernelParams);

            /* prevent re-entry on the nvbit_callback when call cudaMalloc */
            skip_flag = true;
            get_mem_val();
            skip_flag = false;

            /* instrument this kernel */
            instrument_function_if_needed(ctx, p->f);

            nvbit_enable_instrumented(ctx, p->f, true);
        } else {
            skip_flag = true;
            cudaFree(d_mem_val);
            cudaFree(d_mem_counter);
            skip_flag = false;
        }
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    /* read data race log */
    get_data_race_log();
}