/* 
 *
 * A NVBit tool, which will log the input and output of kernel function
 * input is defined as memory copied from host to device and arguments of kernel
 * output is defined as memory copied from device to host
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

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

int cudaMemcpy_input_count = 0;
int cudaMemcpy_output_count = 0;
int funcParams_count = 0;

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
    
    FILE *fp = fopen(filename, "wb");

    if (!fp) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    if (fwrite(src, ByteCount, 1, fp) != 1) {
        std::cerr << strerror(errno) << "failed to write file.\n";
        exit(1);
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

/* This is triggered every time a cudaMemcpy is called */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* if cudaMemcpy from Device to Host
     * it is treated as output of the kernel
     * should be trigger at the end of cuMemcpy
     * so that dstHost has the memory ready
     */
    if ((cbid == API_CUDA_cuMemcpyDtoH_v2) && is_exit)  {
        /* get parameters */
        cuMemcpyDtoH_params *p = (cuMemcpyDtoH_params *)params;

        dump_mem(p->dstHost, p->ByteCount, 0);
    } else if ((cbid == API_CUDA_cuMemcpyHtoD_v2) && is_exit)  {
        cuMemcpyHtoD_params *p = (cuMemcpyHtoD_params *)params;

        dump_mem(p->srcHost, p->ByteCount, 1);
    } else if ((cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) && !is_exit) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        /* get kernel function signature */    
        std::string func_sig(nvbit_get_func_name(ctx, p->f));

        save_nonpointer_arguments(p->kernelParams, func_sig);
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    mkdir("kernel_log", 0777);
}