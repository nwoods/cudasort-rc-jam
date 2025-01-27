#ifndef CUDASORT_RC_JAM_BITONIC__
#define CUDASORT_RC_JAM_BITONIC__

#include<concepts>
#include<cstring>

#include<iostream>


namespace
{
    template<std::unsigned_integral T>
    T next_power_of_2(T x)
    {
        x--;
        std::size_t shift = 1;
        while(shift < (sizeof(T) * 8))
        {
            x |= (x >> shift);
            shift <<= 1;
        }
        return ++x;
    }

    template<std::unsigned_integral T>
    unsigned log_base_2(T x)
    {
        unsigned shift = 0;
        while(x > 1)
        {
            x >>= 1;
            ++shift;
        }
        return shift;
    }

    template<std::unsigned_integral T>
    T prev_power_of_2(T x)
    {
        return 1 << log_base_2(x);
    }

    // fast computation of x % (2^n)
    __device__ inline size_t fast_mod_pow_2(size_t x, size_t n)
    {
        return x & ((1 << n) - 1);
    }

    template<typename T>
    __device__ T my_min(T a, T b)
    {
        return a < b ? a : b;
    }
} // anonymous namespace

template<typename T>
__global__ void bitonic_kernel_a(T* arr, std::size_t pass, std::size_t len)
{
    const std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // i = idx // 2^k * 2^(k+1) + idx % k where k=pass
    // note idx//2^k * 2^(k+1) != 2*idx because integer division is floored
    const std::size_t i = ((idx >> pass) << (pass + 1)) + ::fast_mod_pow_2(idx, pass);
    const std::size_t j = ((idx >> pass) << (pass + 1)) + (1 << (pass + 1)) - ::fast_mod_pow_2(idx, pass) - 1;

    if(/*j > i && */j < len && arr[j] < arr[i])
    {
        T tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

template<typename T>
__global__ void bitonic_kernel_b(T* arr, std::size_t pass, std::size_t subpass, std::size_t len)
{
    const std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // i = idx // 2^k * 2^(k+1) + idx % k where k=(pass-subpass)
    const std::size_t i = ((idx >> (pass - subpass - 1)) << (pass - subpass)) + ::fast_mod_pow_2(idx, pass - subpass - 1);
    const std::size_t j = i + (1 << (pass - subpass - 1));

    if(j < len && arr[j] < arr[i])
    {
        T tmp = arr[j];
        arr[j] = arr[i];
        arr[i] = tmp;
    }
}

template<typename T>
__host__ void bitonic_sort_gpu(T* h_arr, std::size_t len)
{
    const std::size_t fakelen = ::next_power_of_2(len);

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
__global__ void bitonic_kernel_shared(T* arr, unsigned passes_to_do, unsigned len)
{
    extern __shared__ T shared[];

    const unsigned block_elems = 1 << passes_to_do;
    const unsigned block_first_i = block_elems * blockIdx.x;
    const unsigned thread_elems = (block_elems + blockDim.x - 1) / blockDim.x;



    if((block_first_i + thread_elems * threadIdx.x) < len)
    {
        // printf("Block %d thread %d copying global elements %d to %d into shared memory elements %d to %d, which is %d bytes (thread_elems=%d, len=%d, block_first_i=%d)\n",
        //     blockIdx.x,
        //     threadIdx.x,
        //     block_first_i + thread_elems * threadIdx.x,
        //     (block_first_i + thread_elems * threadIdx.x) + ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)),
        //     thread_elems * threadIdx.x,
        //     (thread_elems * threadIdx.x) + ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)),
        //     ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)) * unsigned(sizeof(T)),
        //     thread_elems,
        //     len,
        //     block_first_i);
        std::memcpy(&shared[thread_elems * threadIdx.x], &arr[block_first_i + thread_elems * threadIdx.x], ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)) * sizeof(T));
    }
    else
    {
        // printf("Block %d thread %d NOT copying global elements %d to %d into shared memory elements %d to %d, which is %d bytes (thread_elems=%d, len=%d, block_first_i=%d)\n",
        //     blockIdx.x,
        //     threadIdx.x,
        //     block_first_i + thread_elems * threadIdx.x,
        //     (block_first_i + thread_elems * threadIdx.x) + ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)),
        //     thread_elems * threadIdx.x,
        //     (thread_elems * threadIdx.x) + ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)),
        //     ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)) * unsigned(sizeof(T)),
        //     thread_elems,
        //     len,
        //     block_first_i);
    }

    __syncthreads();

    for(unsigned pass = 0; pass < passes_to_do; ++pass)
    {
        // step a
        if(thread_elems >= (1 << (pass + 1))) // thread can do entire sub-problem
        {
            for(unsigned k = 0; k < (thread_elems + 1) / 2; ++k)
            {
                // here, tier is within the elements this thread is responsible for
                const unsigned tier = k / (1 << pass);
                const unsigned column = k % (1 << pass);
                const unsigned i = thread_elems * threadIdx.x + tier * (1 << (pass + 1)) + column;
                const unsigned j = thread_elems * threadIdx.x + (tier + 1) * (1 << (pass + 1)) - 1 - column;

                if((block_first_i + j) < len && shared[j] < shared[i])
                {
                    // printf("Block %d thread %d pass %d out of %d (non-sync version), k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, k, tier, column, i, j, shared[i], shared[j]);
                    T tmp = shared[j];
                    shared[j] = shared[i];
                    shared[i] = tmp;
                }
                else
                {
                    // printf("Block %d thread %d pass %d out of %d (non-sync version), k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: NOT SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, k, tier, column, i, j, shared[i], shared[j]);
                }
            }
        }
        else
        {
            // previous step let threads run independently, sync here now that they are dependent for the first time
            __syncthreads();

            for(unsigned k = 0; k < (thread_elems + 1) / 2; ++k)
            {
                const unsigned thread_first_idx = thread_elems / 2 * threadIdx.x;
                // Here, tier is within the elements the block is responsible for
                const unsigned tier = (thread_first_idx + k) / (1 << (pass));
                const unsigned column = (thread_first_idx + k) % (1 << pass);
                const unsigned i = tier * (1 << (pass + 1)) + column;
                const unsigned j = (tier + 1) * (1 << (pass + 1)) - column - 1;

                if((block_first_i + j) < len && shared[j] < shared[i])
                {
                    // printf("Block %d thread %d pass %d out of %d (sync version), k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, k, tier, column, i, j, shared[i], shared[j]);
                    T tmp = shared[j];
                    shared[j] = shared[i];
                    shared[i] = tmp;
                }
                else
                {
                    // printf("Block %d thread %d pass %d out of %d (sync version), k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: NOT SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, k, tier, column, i, j, shared[i], shared[j]);
                }
            }

            __syncthreads();
        }

        // step b
        for(unsigned subpass = 0; subpass < pass; ++subpass)
        {
            if(thread_elems >= (1 << (pass - subpass)))
            {
                for(unsigned k = 0; k < (thread_elems + 1) / 2; ++k)
                {
                    // here, tier is within the elements this thread is responsible for
                    const unsigned tier = k / (1 << (pass - subpass - 1));
                    const unsigned column = k % (1 << (pass - subpass - 1));
                    const unsigned i = thread_elems * threadIdx.x + tier * (1 << (pass - subpass)) + column;
                    const unsigned j = i + (1 << (pass - subpass - 1));

                    if((block_first_i + j) < len && shared[j] < shared[i])
                    {
                        // printf("Block %d thread %d pass %d out of %d (non-sync version), subpass %d, k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, subpass, k, tier, column, i, j, shared[i], shared[j]);
                        T tmp = shared[j];
                        shared[j] = shared[i];
                        shared[i] = tmp;
                    }
                    else
                    {
                        // printf("Block %d thread %d pass %d out of %d (non-sync version), subpass %d, k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: NOT SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, subpass, k, tier, column, i, j, shared[i], shared[j]);
                    }
                }
            }
            else
            {
                for(unsigned k = 0; k < (thread_elems + 1) / 2; ++k)
                {
                    const unsigned thread_first_idx = thread_elems / 2 * threadIdx.x;
                    // Here, tier is within the elements the block is responsible for
                    const unsigned tier = (thread_first_idx + k) / (1 << (pass - subpass - 1));
                    const unsigned column = (thread_first_idx + k) % (1 << (pass - subpass - 1));
                    const unsigned i = tier * (1 << (pass - subpass)) + column;
                    const unsigned j = i + (1 << (pass - subpass - 1));

                    if((block_first_i + j) < len && shared[j] < shared[i])
                    {
                        // printf("Block %d thread %d pass %d out of %d (sync version), subpass %d, k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, subpass, k, tier, column, i, j, shared[i], shared[j]);
                        T tmp = shared[j];
                        shared[j] = shared[i];
                        shared[i] = tmp;
                    }
                    else
                    {
                        // printf("Block %d thread %d pass %d out of %d (non-sync version), subpass %d, k=%d, tier=%d, column=%d, i=%d, j=%d, arr[i]=%d, arr[j]=%d :: NOT SWAPPING\n", blockIdx.x, threadIdx.x, pass, passes_to_do, subpass, k, tier, column, i, j, shared[i], shared[j]);
                    }
                }

                __syncthreads();
            }
        }
    }

    __syncthreads();

    if((block_first_i + thread_elems * threadIdx.x) < len)
    {
        std::memcpy(&arr[block_first_i + thread_elems * threadIdx.x], &shared[thread_elems * threadIdx.x], ::my_min(thread_elems, len - (block_first_i + thread_elems * threadIdx.x)) * sizeof(T));
    }
}

template<typename T>
__host__ void bitonic_sort_shared(T* h_arr, std::size_t len)
{
    cudaError_t err;
    const std::size_t fakelen = ::next_power_of_2(len);

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
    const std::size_t shared_mem_space = devprop.sharedMemPerBlockOptin - devprop.reservedSharedMemPerBlock;
    const std::size_t elems_per_block = ::prev_power_of_2(shared_mem_space / sizeof(T));
    const unsigned passes_shared = ::log_base_2(std::min(elems_per_block, fakelen));
    const std::size_t shared_blocks = (fakelen + elems_per_block - 1) / elems_per_block;
    cudaFuncSetAttribute(bitonic_kernel_shared<T>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_space);
    const std::size_t shared_threads_per_block = 128; // how do we tune this?

    std::cout << "len: " << len << ", shared_mem_space: " << shared_mem_space << ", elems_per_block: " << elems_per_block << ", passes_shared: " << passes_shared << ", shared_blocks: " << shared_blocks << std::endl;

    bitonic_kernel_shared<<<shared_blocks, shared_threads_per_block, shared_mem_space>>>(d_arr, passes_shared, len);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in kernel: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_arr, d_arr, arrsize, cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cout << "Error in early copy back: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "Done with shared memory part, moving to final passes" << std::endl;

    const std::size_t threadsPerBlock = 256; // does this matter?
    // in principle we only need len/2 threads, not fakelen/2, but it would require some logic changes to skip some values of idx.
    // Maybe a feature to add later.
    const std::size_t n_blocks = (((fakelen + 1) / 2) + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Running passes " << passes_shared << " to " << ::log_base_2(fakelen)-1 << std::endl;

    for(std::size_t pass = passes_shared; pass < ::log_base_2(fakelen); ++pass)
    {
        // std::cout << "Starting pass " << pass << std::endl;
        bitonic_kernel_a<<<n_blocks, threadsPerBlock>>>(d_arr, pass, len);

        for(std::size_t subpass = 0; subpass < pass; ++subpass)
        {
            // std::cout << "Starting subpass " << subpass << std::endl;
            bitonic_kernel_b<<<n_blocks, threadsPerBlock>>>(d_arr, pass, subpass, len);
            // std::cout << "Done with subpass " << subpass << std::endl;
        }
        // std::cout << "Done with pass " << pass << std::endl;
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
