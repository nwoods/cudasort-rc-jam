#pragma once

#include<iostream>

#include<assert.h>

#include "utils.hpp"


// A and B contiguous
template<typename T>
__device__ void merge_uneven_sorted_subarrays(T* in, T* out, unsigned len1, unsigned len2)
{
    T* a = in;
    T* a_end = a + len1;
    T* b = a_end;
    T* b_end = a_end + len2;
    T* o = out;
    while(a < a_end && b < b_end)
    {
        if((*a) < (*b))
        {
            *(o++) = *(a++);
        }
        else
        {
            *(o++) = *(b++);
        }
    }

    while(a < a_end)
    {
        *(o++) = *(a++);
    }
    while(b < b_end)
    {
        *(o++) = *(b++);
    }
}

// A and B not contiguous
template<typename T>
__device__ void merge_uneven_sorted_subarrays(T* in1, T* in2, T* out, unsigned len1, unsigned len2)
{
    T* a = in1;
    T* a_end = a + len1;
    T* b = in2;
    T* b_end = b + len2;
    T* o = out;
    while(a < a_end && b < b_end)
    {
        if((*a) < (*b))
        {
            *(o++) = *(a++);
        }
        else
        {
            *(o++) = *(b++);
        }
    }

    while(a < a_end)
    {
        *(o++) = *(a++);
    }
    while(b < b_end)
    {
        *(o++) = *(b++);
    }
}

// First levels iterations of a bottom-up merge sort
// For subproblems that fit fully in shared memory including a workspace buffer

// First levels iterations for a single thread
// Caller allocates arr and workspace (which must not be the same)
// Caller is responsible for knowing that data end in workspace if levels is odd
// Caller is responsible for making sure threads do not overlap
template<typename T>
__device__ void merge_sort_buffered(T* arr, T* workspace, unsigned len, unsigned levels)
{
    T* a = arr;
    T* b = workspace;

    for(unsigned level = 0; level < levels; ++level)
    {
        const unsigned width = 1u << level;
        for(unsigned i = 0; i < len; i += 2 * width)
        {
            const unsigned len1 = utils::min(width, len - i);
            const unsigned len2 = utils::min(width, len - (i + len1));
            merge_uneven_sorted_subarrays(a + i, b + i, len1, len2);
        }
        utils::swap(a, b);
    }
}

// First levels iterations of a bottom-up merge sort
// For subproblems that fit fully in shared memory including a workspace buffer
// in and out may be the same
template<typename T>
__global__ void merge_sort_buffered_shared(T* in, T* out, unsigned len, unsigned levels)
{
    const unsigned block_elems = utils::min(1u << levels, len);

    extern __shared__ T shared[];
    T* a = shared;
    T* b = shared + block_elems;

    const unsigned block_first_i = block_elems * blockIdx.x;

    utils::vectorized_block_copy(a, in + block_first_i, block_elems);

    __syncthreads();

    const unsigned thread_idx = utils::thread_index_shuffle(threadIdx.x, blockDim.x);

    for(unsigned level = 0; level < levels; ++level)
    {
        const unsigned width = 1u << level;
        for(unsigned i = 2 * width * thread_idx; i < block_elems; i += 2 * width * blockDim.x)
        {
            const unsigned len1 = utils::min(width, len - (block_first_i + i));
            const unsigned len2 = utils::min(width, len - (block_first_i + i + len1));
            merge_uneven_sorted_subarrays(a + i, b + i, len1, len2);
        }

        __syncthreads();
        utils::swap(a, b);
    }

    utils::vectorized_block_copy(out + block_first_i, a, block_elems);
}

template<typename T>
__device__ void _verify(T* arr, unsigned start, unsigned end)
{
    printf("verifying from %u to %u\n", start, end);
    for(unsigned i = start; i < (end - 1); ++i)
    {
        if(arr[i] > arr[i + 1])
        {
            printf("Sort breaks after item %u\n", i);
            assert(0);
        }
    }
}

template<typename T>
__global__ void __verify(T* arr, unsigned start, unsigned end)
{
    printf("Double underscore verify\n");
    _verify(arr, start, end);
    printf("Double underscore done\n");
}

// One iteration (level) of bottom-up merge sort in global memory
template<typename T>
__global__ void merge_sort_buffered_global(T* in, T* out, unsigned len, unsigned level)
{
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned width = 1u << level;

    const unsigned i = 2 * width * idx;
    if(i >= len) return;
    const unsigned len1 = utils::min(width, len - i);
    const unsigned len2 = utils::min(width, len - (i + len1));
    merge_uneven_sorted_subarrays(in + i, out + i, len1, len2);
}

template<typename T>
void verify_(T* arr, size_t len)
{
    for(size_t i = 0; i < (len - 1); ++i)
    {
        if(arr[i+1] < arr[i])
        {
            std::cout << "Sort fails after item " << i << std::endl;
            assert(false);
        }
    }
}

// Simple CS101 version. Slow. Buffered.
template<typename T>
__host__ void merge_sort_gpu_buffered(T* h_arr, std::size_t len)
{
    cudaError_t err;

    T *d_arr, *d_buf;
    const std::size_t arrsize = len * sizeof(T);
    cudaMalloc(&d_arr, arrsize);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in malloc 1: " << cudaGetErrorString(err);
    }

    cudaMemcpy(d_arr, h_arr, arrsize, cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in memcopy to device: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop, 0); // assume only 1 GPU

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in device properties call: " << cudaGetErrorString(err) << std::endl;
    }

    const std::size_t shared_mem_space = devprop.sharedMemPerBlock - devprop.reservedSharedMemPerBlock;

    cudaFuncSetAttribute(merge_sort_buffered_shared<T>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_space);

    const unsigned levels = utils::log_base_2(2 * len - 1);
    const unsigned max_shared_items = utils::prev_power_of_2((shared_mem_space / sizeof(T)) >> 1);
    const unsigned max_shared_level = utils::log_base_2(max_shared_items);
    const unsigned n_blocks_shared = (arrsize / sizeof(T) + max_shared_items - 1) / max_shared_items;
    const unsigned threads_per_block = 1024;
    std::cout << "Running with " << shared_mem_space << "B (" << shared_mem_space / sizeof(T) << " items) in shared memory per block; max level in shared memory: " << max_shared_level << std::endl;

    unsigned level = std::min(max_shared_level, levels);

    std::cout << "Running shared memory steps with " << n_blocks_shared << " blocks of " << threads_per_block << " threads each" << std::endl;

    merge_sort_buffered_shared<<<n_blocks_shared, threads_per_block, shared_mem_space>>>(d_arr, d_arr, len, level);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in kernel: " << cudaGetErrorString(err) << std::endl;
    }

    T* d_ret_from = d_arr;
    T* d_a = d_arr;

    if(max_shared_level < levels)
    {
        cudaMalloc(&d_buf, arrsize);
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error in malloc 2: " << cudaGetErrorString(err) << std::endl;
        }
        d_a = d_buf;
    }


    for(; level < levels; ++level)
    {
        const unsigned items_per_merge = (1u << (level + 1));
        const unsigned threads_needed = (len + items_per_merge - 1) / items_per_merge;
        const unsigned blocks_needed = (threads_needed + threads_per_block - 1) / threads_per_block;
        merge_sort_buffered_global<<<blocks_needed, threads_per_block>>>(d_ret_from, d_a, len, level);

        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error in global kernel (level " << level << "): " << cudaGetErrorString(err) << std::endl;
        }

        std::swap(d_a, d_ret_from);
    }

    cudaMemcpy(h_arr, d_ret_from, arrsize, cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in copy back: " << cudaGetErrorString(err) << std::endl;


    }

    cudaFree(d_arr);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in free: " << cudaGetErrorString(err) << std::endl;
    }

    if(max_shared_level < levels)
    {
        cudaFree(d_buf);

        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error in workspace free: " << cudaGetErrorString(err) << std::endl;
        }
    }
}



// Version based on merge paths (https://arxiv.org/pdf/1406.2628) for better parallelization.
// Buffered.
template<typename T>
__host__ void merge_sort_gpu_buffered_path(T* h_arr, std::size_t len)
{
    cudaError_t err;

    // Items each thread is responsible for in each kernel invocation. Does not change across algorithm.
    constexpr unsigned items_per_thread_exp = 5;
    constexpr unsigned items_per_thread = 1u << items_per_thread_exp;
    const unsigned levels = utils::log_base_2(2 * len - 1);

    T *d_arr, *d_buf;
    const std::size_t arrsize = len * sizeof(T);
    cudaMalloc(&d_arr, arrsize);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in malloc 1: " << cudaGetErrorString(err);
    }

    cudaMemcpy(d_arr, h_arr, arrsize, cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in memcopy to device: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop, 0); // assume only 1 GPU

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in device properties call: " << cudaGetErrorString(err) << std::endl;
    }

    const std::size_t shared_mem_max = devprop.sharedMemPerBlock - devprop.reservedSharedMemPerBlock; // can change to opt-in size if needed
    const unsigned threads_per_block_max = 1024;
    const unsigned threads_per_block = std::min(threads_per_block_max, static_cast<unsigned>(utils::prev_power_of_2(shared_mem_max / (2 * items_per_thread * sizeof(T)))));
    const unsigned items_per_block = threads_per_block * items_per_thread;
    const unsigned n_blocks = (len + items_per_block - 1) / items_per_block;
    const std::size_t shared_mem_space = items_per_block * sizeof(T) * 2; // CUDA may allocate more
    cudaFuncSetAttribute(merge_sort_buffered_shared<T>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_space);

    // When sub-arrays to merge are smaller than one thread's responsibility, do the "obvious" thing

    const unsigned initial_levels = std::min(levels, items_per_thread_exp);
    merge_sort_buffered_shared<<<n_blocks, threads_per_block, shared_mem_space>>>(d_arr, d_arr, len, initial_levels);

    const unsigned max_shared_items = utils::prev_power_of_2((shared_mem_space / sizeof(T)) >> 1);
    const unsigned max_shared_level = utils::log_base_2(max_shared_items);
    const unsigned n_blocks_shared = (arrsize / sizeof(T) + max_shared_items - 1) / max_shared_items;
    std::cout << "Running " << initial_levels << "levels of naive merge sort with " << shared_mem_space << "B (" << shared_mem_space / sizeof(T) << " items) in shared memory per block" << std::endl;

    unsigned level = std::min(max_shared_level, levels);

    std::cout << "Running shared memory steps with " << n_blocks_shared << " blocks of " << threads_per_block << " threads each" << std::endl;



    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in kernel: " << cudaGetErrorString(err) << std::endl;
    }

    T* d_ret_from = d_arr;
    T* d_a = d_arr;

    if(max_shared_level < levels)
    {
        cudaMalloc(&d_buf, arrsize);
        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error in malloc 2: " << cudaGetErrorString(err) << std::endl;
        }
        d_a = d_buf;
    }


    for(; level < levels; ++level)
    {
        const unsigned items_per_merge = (1u << (level + 1));
        const unsigned threads_needed = (len + items_per_merge - 1) / items_per_merge;
        const unsigned blocks_needed = (threads_needed + threads_per_block - 1) / threads_per_block;
        merge_sort_buffered_global<<<blocks_needed, threads_per_block>>>(d_ret_from, d_a, len, level);

        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error in global kernel (level " << level << "): " << cudaGetErrorString(err) << std::endl;
        }

        std::swap(d_a, d_ret_from);
    }

    cudaMemcpy(h_arr, d_ret_from, arrsize, cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in copy back: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_arr);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in free: " << cudaGetErrorString(err) << std::endl;
    }

    if(max_shared_level < levels)
    {
        cudaFree(d_buf);

        err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            std::cout << "Error in workspace free: " << cudaGetErrorString(err) << std::endl;
        }
    }
}
