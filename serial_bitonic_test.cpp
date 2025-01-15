#include "serial_sorts.h"
#include "gen_random_array.h"

#include<iostream>
#include<array>

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

int main()
{
    std::array<uint32_t, 16> in = random_array<uint32_t, 16>(0, 128);

    std::cout << "Unsorted input array: " << std::endl;
    print_array(in);

    std::array<uint32_t, 16> out;

    bitonic_sort_flip_serial(in.data(), out.data(), in.size());

    std::cout << "Sorted output array: " << std::endl;
    print_array(out);
}
