#ifndef CUDASORT_RC_JAM_SERIAL_SORTS__
#define CUDASORT_RC_JAM_SERIAL_SORTS__

#include<algorithm>

#include<iostream>
#include<cassert>
#include<concepts>


template<std::integral T>
T next_power_of_2(T x)
{
    x--;
    size_t shift = 1;
    while(shift < (sizeof(T) * 8))
    {
        x |= (x >> shift);
        shift <<= 1;
    }
    return ++x;
}

template<typename T>
void bitonic_sort_flip_serial(T* in, T* out, size_t len)
{
    assert(bool(len & (!(len & (len - 1))))); // len must be a power of 2

    std::copy(in, in + len, out);

    for(size_t k = 2; k <= len; k <<= 1)
    {
        for(size_t j = k / 2; j > 0; j >>= 1)
        {
            for(size_t i = 0; i < len; ++i)
            {
                const size_t l = (i ^ j);

                if(l > i && l < len && (((i & k) == 0) == (out[i] > out[l])))
                {
                    std::swap(out[i], out[l]);
                }
            }
        }
    }
}

template<typename T>
void bitonic_sort_serial(T* in, T* out, size_t len)
{
    std::copy(in, in + len, out);

    const size_t fakelen = next_power_of_2(len);

    for(size_t pass = 0; (1 << pass) < fakelen; ++pass)
    {
        for(size_t tier = 0; tier < fakelen / (1 << (pass + 1)); ++tier)
        {
            for(size_t i = 0; i < (1 << pass); ++i)
            {
                const size_t lo = (1 << (pass + 1)) * tier + i;
                const size_t hi = (1 << (pass + 1)) * tier + (1 << (pass + 1)) - i - 1;

                if(hi < len && out[hi] < out[lo])
                {
                    std::swap(out[hi], out[lo]);
                }
            }
        }

        for(size_t subpass = 0; subpass < pass; ++subpass)
        {
            for(size_t subtier = 0; subtier < fakelen / (1 << (pass - subpass)); ++subtier)
            {
                for(size_t i = 0; i < (1 << (pass - subpass - 1)); ++i)
                {
                    const size_t lo = (1 << (pass - subpass)) * subtier + i;
                    const size_t hi = lo + (1 << (pass - subpass - 1));

                    if(hi < len && out[hi] < out[lo])
                    {
                        std::swap(out[hi], out[lo]);
                    }
                }
            }
        }
    }
}

#endif // header guard
