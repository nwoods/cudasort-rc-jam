// #include "serial_sorts.h"
#include "mergesort.hpp"
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
        if(arr[i+1] < arr[i])
        {
            std::cout << "Sort fails after item " << i << std::endl;
            assert(false);
        }
    }
}

constexpr size_t len = (1u << 27); // (1 << 28);
using DataType = unsigned;

int main()
{
    std::vector<DataType> arr = random_array_v<DataType, len>(0, 999999);

    if(len <= 1024)
    {
        std::cout << "Unsorted input array: " << std::endl;
        print_array(arr);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // merge_sort_gpu_buffered(arr.data(), len);
    path_merge_sort_gpu_buffered(static_cast<DataType*>(arr.data()), len);

    // std::sort(arr.begin(), arr.end());

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Sort took " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    if(len <= 2048)//1024)
    {
        std::cout << "Sorted output array: " << std::endl;
        print_array(arr);
    }

    verify(arr);

    std::cout << "Sort successful!" << std::endl;
}
