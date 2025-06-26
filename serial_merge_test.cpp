#include "serial_sorts.h"
#include "gen_random_array.h"

#include<iostream>
#include<array>
#include<cassert>

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

void verify(const auto& arr)
{
    for(size_t i = 0; i < (arr.size() - 1); ++i)
    {
        assert(arr[i] <= arr[i+1]); // Sorted means monotonically increasing
    }
}

constexpr size_t len = 500;

int main()
{
    std::array<uint32_t, len> arr = random_array<uint32_t, len>(0, 500);

    std::cout << "Unsorted input array: " << std::endl;
    print_array(arr);


    std::array<uint32_t, len> workspace;
    // merge_sort_serial_recursive(arr.data(), workspace.data(), arr.size());
    // merge_sort_serial_iterative(arr.data(), workspace.data(), arr.size());

    // merge_sort_serial_inplace(arr.data(), arr.size());

    path_merge_sort_serial_iterative(arr.data(), workspace.data(), len, 32);

    std::cout << "Sorted output array: " << std::endl;
    print_array(arr);

    verify(arr);

    std::cout << "Sort successful!" << std::endl;
}
