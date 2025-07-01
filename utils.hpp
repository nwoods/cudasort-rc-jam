#pragma once

#include<concepts>

#include "vec_typedefs.hpp"

namespace utils
{

template<std::unsigned_integral T>
__host__ __device__ T next_power_of_2(T x)
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
__host__ __device__ unsigned log_base_2(T x)
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
__host__ __device__ T prev_power_of_2(T x)
{
    return 1 << log_base_2(x);
}

// fast computation of x % (2^n)
template<std::unsigned_integral T>
__device__ inline T fast_mod_pow_2(T x, unsigned n)
{
    return x & ((1 << n) - 1);
}

// As above but you give it a number assumed to be a power of 2 instead of the exponent
template<std::unsigned_integral T>
__device__ inline T fast_mod_known_pow_2(T x, T m)
{
    return x & (m - 1);
}

// x / d assuming d is a power of 2
template<std::unsigned_integral T>
__device__ inline T fast_div_known_pow_2(T x, T d)
{
    while(d && x)
    {
        d >>= 1;
        x >>= 1;
    }

    return x;
}

template<std::integral T>
__device__ inline bool is_pow_2(T x)
{
    return bool(x) && !bool(x & (x - 1));
}

template<typename T>
__device__ T min(T a, T b)
{
    return a < b ? a : b;
}

// shuffle thread indices 32 ways within a block so threads in the same warp will almost never try to access nearby memory
// Only works if blocksize is a power of 2 (otherwise leaves index unchangede)
__device__ int thread_index_shuffle(unsigned threadidx, unsigned blocksize)
{
    const unsigned warpsize = 32;

    if(blocksize < warpsize || !is_pow_2(blocksize))
    {
        return threadidx;
    }

    return fast_mod_known_pow_2(threadidx + fast_mod_known_pow_2(threadidx, warpsize) * (fast_div_known_pow_2(blocksize, warpsize)), blocksize);
}

template<typename T>
__device__ void swap(T& a, T& b)
{
    // I still don't completely understand when the compiler does or does not optimize out this temporary
    // TODO: under certain circumstances, bitwise in-place swap may be better (https://graphics.stanford.edu/~seander/bithacks.html#SwappingValuesXOR)
    T tmp = a;
    a = b;
    b = tmp;
}

// Use whole block to copy
// E.g. for global->shared or vice versa
template<typename T>
__device__ void _vectorized_block_copy(T* dest, T* src, unsigned len)
{
    constexpr unsigned vecsize = 4;
    typedef typename Vec_t<T, static_cast<std::size_t>(vecsize)>::type VecType;

    const unsigned thread_idx = threadIdx.x;
    const unsigned n_threads = blockDim.x;
    const unsigned thread_elems = (len / n_threads) + (thread_idx < (len % n_threads) ? 1u : 0u);

    unsigned i = 0;
    for(; i < (thread_elems / vecsize); ++i)
    {
        reinterpret_cast<VecType*>(&dest[(i * n_threads + thread_idx) * vecsize])[0] = reinterpret_cast<VecType*>(&src[(i * n_threads + thread_idx) * vecsize])[0];
    }
    i *= vecsize;
    for(; i < thread_elems; ++i) // in case it's not a multiple of 4
    {
        dest[i * n_threads + thread_idx] = src[i * n_threads + thread_idx];
    }
}

// Use whole block to copy
// E.g. for global->shared or vice versa
template<typename T>
__device__ void vectorized_block_copy(T* dest, T* src, unsigned len)
{
    constexpr unsigned vecsize = 4;
    typedef typename Vec_t<T, static_cast<std::size_t>(vecsize)>::type VecType;

    const unsigned thread_idx = threadIdx.x;
    const unsigned n_threads = blockDim.x;

    unsigned i = thread_idx * vecsize;
    for(; i + vecsize <= len; i += n_threads * vecsize)
    {
        if(reinterpret_cast<std::size_t>(dest + i) % (vecsize * sizeof(T)))
        {
            printf("Thread %u trying bad write of size %u with offset %u", threadIdx.x, vecsize * static_cast<unsigned>(sizeof(T)), i);
        }
        reinterpret_cast<VecType*>(&dest[i])[0] = reinterpret_cast<VecType*>(&src[i])[0];
    }
    for(; i < len; ++i) // in case it's not a multiple of 4
    {
        if(reinterpret_cast<std::size_t>(dest + i) % (sizeof(T)))
        {
            printf("Thread %u trying bad write of size %u with offset %u", threadIdx.x, static_cast<unsigned>(sizeof(T)), i);
        }
        dest[i] = src[i];
    }
}

}
