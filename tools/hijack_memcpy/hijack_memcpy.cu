/* 
 *
 * A NVBit tool, which will will hijack cudaMemcpy
 *
 * Yineng Yan (yinengy@umich.edu), 2020
 */
#include <stdio.h>
#include <vector>

/* header for every nvbit tool */
#include "nvbit_tool.h"

/* interface of nvbit */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* for signal handler */
#include <signal.h>
#include <sys/mman.h>

#define PAGE_SIZE 4096

typedef unsigned long long uint64;

std::vector<void *> input_pages;
std::vector<void *> modified_pages;

void *aline_addr(const void *addr) {
    return (void *) ((uint64) addr - ((uint64) addr % PAGE_SIZE));
}

/* get pages range from src to (stc + ByteCount) */
void load_input_pages(const void * src, size_t ByteCount) {
    // aline to PAGE SIZE
    void *base_addr = aline_addr(src);

    for (uint64 trav_addr = (uint64) base_addr; trav_addr < (uint64) src + ByteCount; trav_addr += PAGE_SIZE) {
        input_pages.push_back((void *) trav_addr);
    }
}

/* initialize environment for recording */
void init_record() {
    /* set input pages to read only */
    for (size_t i = 1; i < input_pages.size(); i++) {
        void *page = input_pages[i];
        if (mprotect(page, PAGE_SIZE, PROT_READ) != 0) {
        }
    }
}

/* this handler will set page to read and write first
 * and the page triggered sigsegv will be recorded
 * it should be installed during cudaMemcpy
 */
void sigsegv_handler_save_modified_page(int sig, siginfo_t *si, void* v) {
    void *base_addr = aline_addr(si->si_addr);

    // add the page to modified list
    modified_pages.push_back(base_addr);

    // set the page to read and write instead of read only
    mprotect(base_addr, PAGE_SIZE, PROT_READ|PROT_WRITE);
}

/* this handler will set page to read and write
 * it should be used after cudaMemcpy
 */
void sigsegv_handler_resotre_perm(int sig, siginfo_t *si, void* v) {
    void *base_addr = aline_addr(si->si_addr);

    // set the page to read and write instead of read only
    mprotect(base_addr, PAGE_SIZE, PROT_READ|PROT_WRITE);
}

void install_sigsegv_handler() {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO; // indicate the handler will take three arguments
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = sigsegv_handler_save_modified_page;

    if (sigaction(SIGSEGV, &sa, NULL) < 0)
        printf("cannot set sigaction\n");
    return;
}

void uninstall_sigsegv_handler() {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = sigsegv_handler_resotre_perm;

    if (sigaction(SIGSEGV, &sa, NULL) < 0)
        printf("cannot set sigaction\n");
    return;
}

/* This is triggered every time a cudaMemcpy is called */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* if cudaMemcpy from Host to Device */
    /* it is treated as input of the kernel */
    if (cbid == API_CUDA_cuMemcpyHtoD_v2) {
        /* get parameters */
        cuMemcpyHtoD_params *p = (cuMemcpyHtoD_params *)params;

        if (!is_exit)
            load_input_pages(p->srcHost, p->ByteCount);
    }
    /* if kernel lunches */
    else if ((cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel)) {

        if (!is_exit) {
            init_record();
        }
    }
    else if (cbid == API_CUDA_cuMemcpyDtoH_v2) {
        /* get parameters */
        if (is_exit) {
            uninstall_sigsegv_handler();
        } else {
            install_sigsegv_handler();
        }
    }
}