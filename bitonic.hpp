#ifndef CUDASORT_RC_JAM_BITONIC__
#define CUDASORT_RC_JAM_BITONIC__

#include<cstring>

#include<iostream>

#include "vec_typedefs.hpp"
#include "utils.hpp"


template<typename T>
__global__ void bitonic_kernel_a(T* arr, std::size_t pass, std::size_t len)
{
    const std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // i = idx // 2^k * 2^(k+1) + idx % k where k=pass
    // note idx//2^k * 2^(k+1) != 2*idx because integer division is floored
    const std::size_t i = ((idx >> pass) << (pass + 1)) + utils::fast_mod_pow_2(idx, pass);
    const std::size_t j = ((idx >> pass) << (pass + 1)) + (1 << (pass + 1)) - utils::fast_mod_pow_2(idx, pass) - 1;

    if(j < len && arr[j] < arr[i])
    {
        utils::swap(arr[i], arr[j]);
    }
}

template<typename T>
__global__ void bitonic_kernel_b(T* arr, std::size_t pass, std::size_t subpass, std::size_t len)
{
    const std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // i = idx // 2^k * 2^(k+1) + idx % k where k=(pass-subpass)
    const std::size_t i = ((idx >> (pass - subpass - 1)) << (pass - subpass)) + utils::fast_mod_pow_2(idx, pass - subpass - 1);
    const std::size_t j = i + (1 << (pass - subpass - 1));

    if(j < len && arr[j] < arr[i])
    {
        utils::swap(arr[i], arr[j]);
    }
}

template<typename T>
__host__ void bitonic_sort_gpu(T* h_arr, std::size_t len)
{
    const std::size_t fakelen = utils::next_power_of_2(len);

    T* d_arr;
    const std::size_t arrsize = len * sizeof(T);
    cudaMalloc(&d_arr, arrsize);

    cudaMemcpy(d_arr, h_arr, arrsize, cudaMemcpyHostToDevice);

    const std::size_t threadsPerBlock = 256; // does it matter?
    // in principle we only need len/2 threads, not fakelen/2, but it would require some logic changes to skip some values of idx.
    // Maybe a feature to add later.
    const std::size_t n_blocks = (((fakelen + 1) / 2) + threadsPerBlock - 1) / threadsPerBlock;

    for(std::size_t pass = 0; (1 << pass) < fakelen; ++pass)
    {
        bitonic_kernel_a<<<n_blocks, threadsPerBlock>>>(d_arr, pass, len);

        for(std::size_t subpass = 0; subpass < pass; ++subpass)
        {
            bitonic_kernel_b<<<n_blocks, threadsPerBlock>>>(d_arr, pass, subpass, len);
        }
    }

    cudaMemcpy(h_arr, d_arr, arrsize, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

template<typename T>
__device__ void bitonic_step_b_shared(T* shared, unsigned block_elems, unsigned pass, unsigned first_subpass, std::size_t len)
{
    // const unsigned block_elems = 1 << (pass - first_subpass);
    const unsigned thread_idx = utils::thread_index_shuffle(threadIdx.x, blockDim.x);
    const unsigned block_first_i = block_elems * blockIdx.x;
    const unsigned thread_elems = (block_elems + blockDim.x - 1) / blockDim.x;
    const unsigned thread_cmps = (thread_elems + 1) >> 1;

    for(unsigned subpass = first_subpass; subpass < pass; ++subpass)
    {
        for(unsigned k = thread_cmps * thread_idx; k < thread_cmps * (thread_idx + 1); ++k)
        {
            const unsigned i = ((k >> (pass - subpass - 1)) << (pass - subpass)) + utils::fast_mod_pow_2(k, pass - subpass - 1);
            const unsigned j = i + (1 << (pass - subpass - 1));

            if((block_first_i + j) < len && shared[j] < shared[i])
            {
                utils::swap(shared[i], shared[j]);
            }
        }

        // sync if a different thread will check some of these elements next
        if(thread_elems < (1 << (pass - subpass)))
        {
            __syncthreads();
        }
    }
}

template<typename T>
__global__ void finish_bitonic_step_b_shared(T* arr, unsigned pass, unsigned first_subpass, unsigned len)
{
    extern __shared__ T shared[];

    const unsigned block_elems = 1 << (pass - first_subpass);
    const unsigned block_first_i = block_elems * blockIdx.x;
    const unsigned thread_elems = block_elems / blockDim.x + (threadIdx.x < (block_elems % blockDim.x) ? 1u : 0u);

    constexpr std::size_t vecsize = 4;

    typedef typename Vec_t<T, vecsize>::type VecType;

    unsigned i = 0;
    for(; i < (thread_elems / vecsize); ++i)
    {
        reinterpret_cast<VecType*>(&shared[(i * blockDim.x + threadIdx.x) * vecsize])[0] = reinterpret_cast<VecType*>(&arr[block_first_i + (i * blockDim.x + threadIdx.x) * vecsize])[0];
    }
    i *= vecsize;
    for(; i < thread_elems; ++i) // in case it's not a multiple of 4
    {
        shared[i * blockDim.x + threadIdx.x] = arr[block_first_i + i * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    bitonic_step_b_shared(shared, block_elems, pass, first_subpass, len);

    __syncthreads();

    // stop shuffling (keep global memory accesses coalesced)
    i = 0;
    for(; i < (thread_elems / vecsize); ++i)
    {
        reinterpret_cast<VecType*>(&arr[block_first_i + (i * blockDim.x + threadIdx.x) * vecsize])[0] = reinterpret_cast<VecType*>(&shared[(i * blockDim.x + threadIdx.x) * vecsize])[0];
    }
    i *= vecsize;
    for(; i < thread_elems; ++i) // in case it's not a multiple of 4
    {
        arr[block_first_i + i * blockDim.x + threadIdx.x] = shared[i * blockDim.x + threadIdx.x];
    }
}

template<typename T>
__global__ void bitonic_kernel_shared(T* arr, unsigned passes_to_do, unsigned len)
{
    extern __shared__ T shared[];

    const unsigned block_elems = 1 << passes_to_do;
    const unsigned block_first_i = block_elems * blockIdx.x;
    const unsigned thread_elems_cpy = block_elems / blockDim.x + (threadIdx.x < (block_elems % blockDim.x) ? 1u : 0u);

    constexpr std::size_t vecsize = 4;

    typedef typename Vec_t<T, vecsize>::type VecType;

    unsigned i = 0;
    for(; i < (thread_elems_cpy / vecsize); ++i)
    {
        reinterpret_cast<VecType*>(&shared[(i * blockDim.x + threadIdx.x) * vecsize])[0] = reinterpret_cast<VecType*>(&arr[block_first_i + (i * blockDim.x + threadIdx.x) * vecsize])[0];
    }
    i *= vecsize;
    for(; i < thread_elems_cpy; ++i) // in case it's not a multiple of 4
    {
        shared[i * blockDim.x + threadIdx.x] = arr[block_first_i + i * blockDim.x + threadIdx.x];
    }

    __syncthreads();

    // shuffle shared memory access to avoid bank conflicts
    const unsigned thread_idx = utils::thread_index_shuffle(threadIdx.x, blockDim.x);

    const unsigned thread_elems = (block_elems + blockDim.x - 1) / blockDim.x;
    const unsigned thread_cmps = (thread_elems + 1) >> 1;

    for(unsigned pass = 0; pass < passes_to_do; ++pass)
    {
        // sync if a different thread processed some of these elements last
        if(thread_elems < (1 << (pass + 1)))
        {
            __syncthreads();
        }

        // step a
        for(unsigned k = thread_cmps * thread_idx; k < thread_cmps * (thread_idx + 1); ++k)
        {
            const unsigned i = ((k >> pass) << (pass + 1)) + utils::fast_mod_pow_2(k, pass);
            const unsigned j = (((k >> pass) + 1) << (pass + 1)) - 1 - utils::fast_mod_pow_2(k, pass);
            if((block_first_i + j) < len && shared[j] < shared[i])
            {
                utils::swap(shared[i], shared[j]);
            }
        }

        // sync if a different thread will process some of these elements next
        if(thread_elems < (1 << (pass + 1)))
        {
            __syncthreads();
        }

        bitonic_step_b_shared(shared, block_elems, pass, 0, len);
    }

    __syncthreads();

    // stop shuffling (keep global memory accesses coalesced)
    i = 0;
    for(; i < (thread_elems_cpy / vecsize); ++i)
    {
        reinterpret_cast<VecType*>(&arr[block_first_i + (i * blockDim.x + threadIdx.x) * vecsize])[0] = reinterpret_cast<VecType*>(&shared[(i * blockDim.x + threadIdx.x) * vecsize])[0];
    }
    i *= vecsize;
    for(; i < thread_elems_cpy; ++i) // in case it's not a multiple of 4
    {
        arr[block_first_i + i * blockDim.x + threadIdx.x] = shared[i * blockDim.x + threadIdx.x];
    }
}

template<typename T>
__host__ void bitonic_sort_shared(T* h_arr, std::size_t len)
{
    cudaError_t err;
    const std::size_t fakelen = utils::next_power_of_2(len);

    T* d_arr;
    const std::size_t arrsize = len * sizeof(T);
    cudaMalloc(&d_arr, arrsize);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in malloc: " << cudaGetErrorString(err) << std::endl;
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
    const std::size_t shared_mem_space = (devprop.sharedMemPerBlockOptin - devprop.reservedSharedMemPerBlock) / 2;
    const std::size_t elems_per_block = utils::prev_power_of_2(shared_mem_space / sizeof(T));
    const unsigned passes_shared = utils::log_base_2(std::min(elems_per_block, fakelen));
    const std::size_t shared_blocks = (fakelen + elems_per_block - 1) / elems_per_block;
    cudaFuncSetAttribute(bitonic_kernel_shared<T>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_space);
    cudaFuncSetAttribute(finish_bitonic_step_b_shared<T>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_space);
    const std::size_t shared_threads_per_block = 1024; // how do we tune this?

    std::cout << "len: " << len << ", shared_mem_space: " << shared_mem_space << ", elems_per_block: " << elems_per_block << ", passes_shared: " << passes_shared << ", shared_blocks: " << shared_blocks << std::endl;

    bitonic_kernel_shared<<<shared_blocks, shared_threads_per_block, shared_mem_space>>>(d_arr, passes_shared, len);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in kernel: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "Done with shared memory part, moving to final passes" << std::endl;

    const std::size_t threadsPerBlock = 1024; // does this matter?
    // in principle we only need len/2 threads, not fakelen/2, but it would require some logic changes to skip some values of idx.
    // Maybe a feature to add later.
    const std::size_t n_blocks = (((fakelen + 1) / 2) + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Running passes " << passes_shared << " to " << utils::log_base_2(fakelen)-1 << std::endl;

    for(std::size_t pass = passes_shared; pass < utils::log_base_2(fakelen); ++pass)
    {
        bitonic_kernel_a<<<n_blocks, threadsPerBlock>>>(d_arr, pass, len);

        for(std::size_t subpass = 0; subpass < pass; ++subpass)
        {
            if((1 << (pass - subpass)) > elems_per_block)
            {
                bitonic_kernel_b<<<n_blocks, threadsPerBlock>>>(d_arr, pass, subpass, len);
            }
            else
            {
                finish_bitonic_step_b_shared<<<shared_blocks, shared_threads_per_block, shared_mem_space>>>(d_arr, pass, subpass, len);

                err = cudaGetLastError();
                if(err != cudaSuccess)
                {
                    std::cout << "Error while finishing step b with shared memory: " << cudaGetErrorString(err) << std::endl;
                }
                break;
            }
        }
    }

    cudaMemcpy(h_arr, d_arr, arrsize, cudaMemcpyDeviceToHost);

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
}

#endif // header guard
