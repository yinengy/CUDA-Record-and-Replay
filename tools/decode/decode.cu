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

            std::cout  << func_id << ',' << inst_id << ',' << nvbit_get_func_name(ctx, func) << ',';
            instr->print();

            char *file_name = (char *) malloc(200);
            char *dir_name = (char *) malloc(200);
            uint32_t line = 0;
            nvbit_get_line_info(ctx, func, instr->getOffset(), &file_name, &dir_name, &line);
            
            std::cout << file_name << ',' << "@ line " << line << std::endl;

            inst_id++;
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
    /* read data race log */
    get_data_race_log();
}