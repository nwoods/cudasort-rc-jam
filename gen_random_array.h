#ifndef CUDASORT_RC_JAM_GEN_RANDOM_ARRAY__
#define CUDASORT_RC_JAM_GEN_RANDOM_ARRAY__

// Template function to generate arrays of random numbers.
// All the "concepts" and template stuff is to allow one template to handle all cases allowed by the STL random library (but really I just wanted to learn c++20 concepts/requires)

#include<random>
#include<array>
#include<limits>
#include<type_traits>
#include<concepts>

template<class T>
concept int_for_distrib = std::is_integral_v<T> && !(std::is_same_v<T, bool> || std::is_same_v<std::make_unsigned_t<T>, unsigned char>);

template<typename T>
requires int_for_distrib<T> || std::floating_point<T>
struct DistType{};

template<int_for_distrib T>
struct DistType<T>{using type = std::uniform_int_distribution<T>;};

template<std::floating_point T>
struct DistType<T>{using type = std::uniform_real_distribution<T>;};

template<typename T, size_t len>
requires int_for_distrib<T> || std::floating_point<T>
std::array<T, len> random_array(T minimum = std::numeric_limits<T>::min(), T maximum = std::numeric_limits<T>::max())
{
    std::mt19937 gen{std::random_device{}()};
    typename DistType<T>::type dist(minimum, maximum);

    std::array<T, len> out;

    for(size_t i = 0; i < out.size(); ++i)
    {
        out[i] = dist(gen);
    }

    return std::move(out);
}

#endif // header guard
