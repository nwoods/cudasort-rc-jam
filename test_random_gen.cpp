#include<gen_random_array.h>

#include<iostream>
#include<array>

int main()
{
    std::cout << "Generating random int array: " << std::endl << "[";
    auto intarr = random_array<int, 10>();
    std::cout << intarr[0];
    for(size_t i = 1; i < intarr.size(); ++i)
    {
        std::cout << ", " << intarr[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "Generating random float array: " << std::endl << "[";
    auto floatarr = random_array<float, 10>();
    std::cout << floatarr[0];
    for(size_t i = 1; i < floatarr.size(); ++i)
    {
        std::cout << ", " << floatarr[i];
    }
    std::cout << "]" << std::endl;

    return 0;
}
