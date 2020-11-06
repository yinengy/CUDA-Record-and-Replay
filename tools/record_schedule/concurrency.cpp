/*

Copyright (c) 2019, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#ifndef CONCURRENCY_CPP
#define CONCURRENCY_CPP

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <iostream>
#include <cuda/std/atomic>

#ifdef __CUDACC__
#  include <cuda_awbarrier.h>
#endif

#ifdef __CUDACC__
# define _ABI __host__ __device__
# define check(ans) { assert_((ans), __FILE__, __LINE__); }
inline void assert_(cudaError_t code, const char *file, int line) {
  if (code == cudaSuccess)
    return;
  std::cerr << "check failed: " << cudaGetErrorString(code) << ": " << file << ':' << line << std::endl;
  abort();
}
#else
# define _ABI
#endif

template <class T>
struct managed_allocator {
  typedef cuda::std::size_t size_type;
  typedef cuda::std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;// (deprecated in C++17)(removed in C++20)    T*
  typedef const T* const_pointer;// (deprecated in C++17)(removed in C++20)    const T*
  typedef T& reference;// (deprecated in C++17)(removed in C++20)    T&
  typedef const T& const_reference;// (deprecated in C++17)(removed in C++20)    const T&

  template< class U > struct rebind { typedef managed_allocator<U> other; };
  managed_allocator() = default;
  template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* out = nullptr;
#ifdef __CUDACC__
# ifdef __aarch64__
    check(cudaMallocHost(&out, n*sizeof(T), cudaHostAllocMapped));
    void* out2;
    check(cudaHostGetDevicePointer(&out2, out, 0));
    assert(out2==out); //< we can't handle non-uniform addressing
# else
    check(cudaMallocManaged(&out, n*sizeof(T)));
# endif
#else
    out = malloc(n*sizeof(T));
#endif
    return static_cast<T*>(out);
  }
  void deallocate(T* p, std::size_t) noexcept {
#ifdef __CUDACC__
# ifdef __aarch64__
    check(cudaFreeHost(p));
# else
    check(cudaFree(p));
# endif
#else
    free(p);
#endif
  }
};
template<class T, class... Args>
T* make_(Args &&... args) {
    managed_allocator<T> ma;
    auto n_ = new (ma.allocate(1)) T(std::forward<Args>(args)...);
#if defined(__CUDACC__) && !defined(__aarch64__)
    check(cudaMemAdvise(n_, sizeof(T), cudaMemAdviseSetPreferredLocation, 0));
    check(cudaMemPrefetchAsync(n_, sizeof(T), 0));
#endif
    return n_;
}
template<class T>
void unmake_(T* ptr) {
    managed_allocator<T> ma;
    ptr->~T();
    ma.deallocate(ptr, sizeof(T));
}

struct null_mutex {
    _ABI void lock() noexcept { }
    _ABI void unlock() noexcept { }
};

struct mutex {
    _ABI void lock() noexcept {
        while (1 == l.exchange(1, cuda::std::memory_order_acquire))
#ifndef __NO_WAIT
            l.wait(1, cuda::std::memory_order_relaxed)
#endif
            ;
    }
    _ABI void unlock() noexcept {
        l.store(0, cuda::std::memory_order_release);
#ifndef __NO_WAIT
        l.notify_one();
#endif
    }
    alignas(64) cuda::atomic<int, cuda::thread_scope_device> l = ATOMIC_VAR_INIT(0);
};
#endif