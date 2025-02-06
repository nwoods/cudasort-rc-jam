// #include "serial_sorts.h"
#include "bitonic.hpp"
#include "gen_random_array.h"

#include<iostream>
#include<array>
#include<cassert>
#include<algorithm>
#include<chrono>

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

constexpr size_t len = (1 << 28);

int main()
{
    std::vector<uint32_t> arr = random_array_v<uint32_t, len>(0, 5000000);

    if(len <= 1024)
    {
        std::cout << "Unsorted input array: " << std::endl;
        print_array(arr);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // bitonic_sort_gpu(arr.data(), len);
    bitonic_sort_shared(arr.data(), len);

    // std::sort(arr.begin(), arr.end());

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Sort took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    if(len <= 1024)
    {
        std::cout << "Sorted output array: " << std::endl;
        print_array(arr);
    }

    verify(arr);

    std::cout << "Sort successful!" << std::endl;
}
