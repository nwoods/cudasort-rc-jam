#include "serial_radix.h"
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

constexpr size_t len = 200;

int main()
{
    std::array<uint32_t, len> in = random_array<uint32_t, len>(0, 500);

    std::cout << "Unsorted input array: " << std::endl;
    print_array(in);

    std::array<uint32_t, len> out;

    radix_sort_serial(in.data(), out.data(), in.size(), 4);

    std::cout << "Sorted output array: " << std::endl;
    print_array(out);

    verify(out);

    std::cout << "Sort successful!" << std::endl;
}
