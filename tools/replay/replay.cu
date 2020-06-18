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

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

int cudaMemcpy_input_count = 0;
int cudaMemcpy_output_count = 0;
int funcParams_count = 0;

void *recorded_mem;
const void *origin_srcHost;

/* read memory content recorded in record phase from files 
 * if is_input is 0, it means the mem is copied from device to host
 * otherwise, the mem is copied from host to device
 * a memory of BtyeCount will be allocated and returned
 */
void *get_recorded_mem(size_t ByteCount, int is_input) {
    
    char filename[25]; // large enough for a counter

    if (is_input) {
        cudaMemcpy_input_count++;
        sprintf(filename, "kernel_log/imem%d.bin", cudaMemcpy_input_count);
    } else {
        cudaMemcpy_output_count++;
        sprintf(filename, "kernel_log/omem%d.bin", cudaMemcpy_output_count);
    }
    
    FILE *fp = fopen(filename, "rb");

    if (!fp) {
        std::cerr << strerror(errno) << "failed to open file.\n";
        exit(1);
    }

    void *buffer = malloc(ByteCount);

    if (fread(buffer, ByteCount, 1, fp) != 1) {
        std::cerr << strerror(errno) << "failed to read file.\n";
        exit(1);
    }

    return buffer;
}

/* compare ptr with output in record phase */
void compare_mem(const void *ptr, size_t ByteCount) {
    void *to_compare = get_recorded_mem(ByteCount, 0);

    int is_equal = memcmp(ptr, to_compare, ByteCount);

    if (is_equal != 0) {
        std::cout << cudaMemcpy_output_count << "th output doesn't match\n!";
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
    char filename[25];
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

/* This is triggered every time a cudaMemcpy is called */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
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
        cbid == API_CUDA_cuLaunchKernel) && !is_exit) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        replace_nonpointer_arguments(p->kernelParams);
    }
}