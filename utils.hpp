#pragma once

#include<concepts>

namespace utils
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
__device__ T my_min(T a, T b)
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
    T tmp = a;
    a = b;
    b = tmp;
}

}
