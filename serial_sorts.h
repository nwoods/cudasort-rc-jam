#ifndef CUDASORT_RC_JAM_SERIAL_SORTS__
#define CUDASORT_RC_JAM_SERIAL_SORTS__

#include<algorithm>

#include<iostream>

template<typename T>
void bitonic_sort_flip_serial(T* in, T* out, size_t len)
{
    std::copy(in, in + len, out);

    for(size_t k = 2; k <= len; k <<= 1)
    {
        for(size_t j = k / 2; j > 0; j >>= 1)
        {
            for(size_t i = 0; i < len; ++i)
            {
                const size_t l = (i ^ j);

                if(l > i) std::cout << "k=" << k << ", j=" << j << ", i=" << i << ", l=" << l << ", arr[i]=" << out[i] << ", arr[l]=" << out[l] << " :: ";

                if(l > i && (((i & k) == 0) == (out[i] > out[l])))
                {
                    std::swap(out[i], out[l]);
                    std::cout << "Swapping!";
                }
                else if(l > i)
                {
                    std::cout << "Not swapping";
                }

                if(l>i) std::cout << std::endl;
            }
        }
    }
}

#endif // header guard
