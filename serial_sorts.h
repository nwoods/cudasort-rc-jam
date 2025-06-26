#ifndef CUDASORT_RC_JAM_SERIAL_SORTS__
#define CUDASORT_RC_JAM_SERIAL_SORTS__

#include<algorithm>

#include<iostream>
#include<cassert>
#include<concepts>
#include<cstring>


namespace
{
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
} // anonymous namespace

template<typename T>
void bitonic_sort_flip_serial(T* arr, size_t len)
{
    assert(bool(len & (!(len & (len - 1))))); // len must be a power of 2

    for(size_t k = 2; k <= len; k <<= 1)
    {
        for(size_t j = k / 2; j > 0; j >>= 1)
        {
            for(size_t i = 0; i < len; ++i)
            {
                const size_t l = (i ^ j);

                if(l > i && l < len && (((i & k) == 0) == (arr[i] > arr[l])))
                {
                    std::swap(arr[i], arr[l]);
                }
            }
        }
    }
}

template<typename T>
void bitonic_sort_flip_serial(T* in, T* out, size_t len)
{
    std::copy(in, in + len, out);

    bitonic_sort_flip_serial<T>(out, len);
}

template<typename T>
void bitonic_sort_serial(T* arr, size_t len)
{
    const size_t fakelen = ::next_power_of_2(len);

    for(size_t pass = 0; (1 << pass) < fakelen; ++pass)
    {
        for(size_t tier = 0; tier < fakelen / (1 << (pass + 1)); ++tier)
        {
            for(size_t i = 0; i < (1 << pass); ++i)
            {
                const size_t lo = (1 << (pass + 1)) * tier + i;
                const size_t hi = (1 << (pass + 1)) * tier + (1 << (pass + 1)) - i - 1;

                if(hi < len && arr[hi] < arr[lo])
                {
                    std::swap(arr[hi], arr[lo]);
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

                    if(hi < len && arr[hi] < arr[lo])
                    {
                        std::swap(arr[hi], arr[lo]);
                    }
                }
            }
        }
    }
}

template<typename T>
void bitonic_sort_serial(T* in, T* out, size_t len)
{
    std::copy(in, in + len, out);

    bitonic_sort_serial<T>(out, len);
}

template<typename T>
void merge_uneven_sorted_subarrays(T* in, T* out, size_t len1, size_t len2)
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

    if(a < a_end)
    {
        std::copy(a, a_end, o);
    }
    else if(b < b_end)
    {
        std::copy(b, b_end, o);
    }
}

// given an array in where subarrays 0-len/2 and len/2-len are sorted, place the fully sorted array in out
template<typename T>
void merge_sorted_subarrays(T* in, T* out, size_t len)
{
    merge_uneven_sorted_subarrays(in, out, len / 2, (len + 1) / 2);
}

// assumes arrays are already allocated
template<typename T>
void merge_sort_serial_recursive_impl(T* in, T* out, size_t len)
{
    if(len < 2) return;

    merge_sort_serial_recursive_impl(out, in, len / 2);
    merge_sort_serial_recursive_impl(out + (len / 2), in + (len / 2), (len + 1) / 2);
    merge_sorted_subarrays(in, out, len);
}

// assumes arrays are already allocated
template<typename T>
void merge_sort_serial_recursive(T* arr, T* workspace, size_t len)
{
    std::copy(arr, arr + len, workspace);
    merge_sort_serial_recursive_impl(workspace, arr, len);
}

template<typename T>
void merge_sort_serial_iterative(T* arr, T* workspace, size_t len)
{
    T* a = arr;
    T* b = workspace;

    for(size_t width = 1; width < len; width *= 2)
    {
        for(size_t i = 0; i < len; i += 2 * width)
        {
            const size_t len1 = std::min(width, len - i);
            const size_t len2 = std::min(width, len - (i + len1));
            merge_uneven_sorted_subarrays(a + i, b + i, len1, len2);
        }

        std::swap(a, b);
    }

    if(a != arr)
    {
        std::copy(a, a + len, arr);
    }
}

template<typename T>
void merge_with_interior_buffer(T* arr, size_t len1, size_t len_buf, size_t len2)
{
    T* a = arr;
    T* a_end = arr + len1;
    T* b = arr + len1 + len_buf;
    T* b_end = b + len2;
    T* buf = arr + len_buf; // counter-intuitive, but fixes an off-by-one error

    while(a < a_end && b < b_end)
    {
        if((*a) < (*b))
        {
            std::swap(*(a++), *(buf++));
        }
        else
        {
            std::swap(*(b++), *(buf++));
        }
    }

    if(a < a_end) std::swap_ranges(a, a_end, buf);
}

template<typename T>
void merge_sort_serial_inplace(T* arr, size_t len)
{
    if(len <= 1) return;
    if(len == 2)
    {
        if(arr[1] < arr[0]) std::swap(arr[0], arr[1]);
        return;
    }
    if(len == 3) // insertion sort if trivial
    {
        if(arr[2] < arr[1]) std::swap(arr[1], arr[2]);
        if(arr[1] < arr[0])
        {
            std::swap(arr[0], arr[1]);
            if(arr[2] < arr[1]) std::swap(arr[1], arr[2]);
        }
        return;
    }

    size_t len1 = len / 4;
    size_t len_buf = (len + 2) / 4;
    size_t len2 = (len + 1) / 2;

    merge_sort_serial_inplace(arr + len1 + len_buf, len2);

    while(len_buf > 1)
    {
        merge_sort_serial_inplace(arr, len1);
        merge_with_interior_buffer(arr, len1, len_buf, len2);

        len2 += len1;
        len1 = len_buf / 2;
        len_buf = (len_buf + 1) / 2;
    }

    // insertion sort the last item or 2
    size_t i = 2;
    while(i < len && arr[i] < arr[1]) ++i;
    if(i > 2)
    {
        T tmp = arr[1];
        std::copy(&arr[2], &arr[i], &arr[1]);
        arr[i - 1] = tmp;
    }
    i = 1;
    while(i < len && arr[i] < arr[0]) ++i;
    if(i > 1)
    {
        T tmp = arr[0];
        std::copy(&arr[1], &arr[i], &arr[0]);
        arr[i - 1] = tmp;
    }
}

template<typename T>
void print_array(const T* arr, size_t len)
{
    std::cout << "[" << arr[0];
    for(size_t i = 1; i < len; ++i)
    {
        std::cout << ", " << arr[i];
    }
    std::cout << "]" << std::endl;
}

template<typename T>
void path_merge_sorted_subarrays(T* in, T* out, size_t len1, size_t len2, size_t n_per_thread)
{
    if(len2 == 0)
    {
        std::copy(in, in + len1, out);
        return;
    }

    if(len1 + len2 <= n_per_thread)
    {
        merge_uneven_sorted_subarrays(in, out, len1, len2);
        return;
    }

    T* a = in;
    T* a_end = a + len1;
    T* b = a_end;
    T* b_end = a_end + len2;

    for(size_t i_thread = 0; i_thread < (len1 + len2 + n_per_thread - 1) / n_per_thread; ++i_thread)
    {
        T* o_start = out + i_thread * n_per_thread;
        T *ai, *bi;
        if(i_thread == 0)
        {
            ai = a;
            bi = b;
        }
        else
        {
            T *a_lo, *a_hi, *b_hi;
            const size_t diag = (i_thread) * n_per_thread;
            if(a + diag < a_end)
            {
                a_hi = a + diag;
                b_hi = b;
                a_lo = a;
            }
            else
            {
                a_hi = a_end;
                b_hi = b + diag - len1;
                a_lo = a + diag - len1;
            }

            size_t n_iter = 0;
            while(true)
            {
                ++n_iter;
                const long offset = (a_hi - a_lo) / 2;
                ai = a_hi - offset;
                bi = b_hi + offset;
                if(bi == b || ai >= a_end || *ai > *(bi - 1))
                {
                    if(ai == a || bi >= b_end || *(ai - 1) <= *bi)
                    {
                        break;
                    }
                    a_hi = ai - 1;
                    b_hi = bi + 1;
                }
                else
                {
                    a_lo = ai + 1;
                }
            }
        }

        T* o = o_start;
        while(o < (o_start + n_per_thread) && o < out + len1 + len2)
        {
            if(ai >= a_end || (bi < b_end && (*ai) > (*bi)))
            {
                *(o++) = *(bi++);
            }
            else
            {
                *(o++) = *(ai++);
            }
        }
    }
}

template<typename T>
void path_merge_sort_serial_iterative(T* arr, T* workspace, size_t len, size_t n_per_thread)
{
    T* a = arr;
    T* b = workspace;

    for(size_t width = 1; width < len; width *= 2)
    {
        for(size_t i = 0; i < len; i += 2 * width)
        {
            const size_t len1 = std::min(width, len - i);
            const size_t len2 = std::min(width, len - (i + len1));
            path_merge_sorted_subarrays(a + i, b + i, len1, len2, n_per_thread);
        }

        std::swap(a, b);
    }

    if(a != arr)
    {
        std::copy(a, a + len, arr);
    }
}

#endif // header guard
