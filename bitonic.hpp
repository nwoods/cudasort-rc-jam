#ifndef CUDASORT_RC_JAM_BITONIC__
#define CUDASORT_RC_JAM_BITONIC__

#include<concepts>

#include<iostream>


namespace
{
    template<std::integral T>
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

    // fast computation of x % (2^n)
    __device__ inline size_t fast_mod_pow_2(size_t x, size_t n)
    {
        return x & ((1 << n) - 1);
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

#endif // header guard
