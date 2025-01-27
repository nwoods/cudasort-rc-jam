// #include "serial_sorts.h"
#include "bitonic.hpp"
#include "gen_random_array.h"

#include<iostream>
#include<array>
#include<cassert>
#include<algorithm>

template<typename T, size_t len>
void print_array(const std::array<T, len>& arr)
{
    std::cout << "[" << arr[0];
    for(size_t i = 1; i < arr.size(); ++i)
    {
        std::cout << ", " << arr[i];
    }
    std::cout << "]" << std::endl;
}

template<typename T>
void print_array(const std::vector<T>& arr)
{
    std::cout << "[" << arr[0];
    for(size_t i = 1; i < arr.size(); ++i)
    {
        std::cout << ", " << arr[i];
    }
    std::cout << "]" << std::endl;
}

void verify(const auto& arr)
{
    for(size_t i = 0; i < (arr.size() - 1); ++i)
    {
        assert(arr[i] <= arr[i+1]); // Sorted means monotonically increasing
    }
}

constexpr size_t len = (1 << 29);

int main()
{
    std::vector<uint32_t> arr = random_array_v<uint32_t, len>(0, 500000);

    if(len <= 1024)
    {
        std::cout << "Unsorted input array: " << std::endl;
        print_array(arr);
    }

    // bitonic_sort_gpu(arr.data(), len);
    bitonic_sort_shared(arr.data(), len);

    // std::sort(arr.begin(), arr.end());

    if(len <= 1024)
    {
        std::cout << "Sorted output array: " << std::endl;
        print_array(arr);
    }

    verify(arr);

    std::cout << "Sort successful!" << std::endl;
}
