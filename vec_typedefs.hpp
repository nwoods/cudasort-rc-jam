#pragma once


template<typename T, size_t sz> struct Vec_t;

template<> struct Vec_t<float, 1>{typedef float1 type;};
template<> struct Vec_t<float, 2>{typedef float2 type;};
template<> struct Vec_t<float, 3>{typedef float3 type;};
template<> struct Vec_t<float, 4>{typedef float4 type;};

template<> struct Vec_t<double, 1>{typedef double1 type;};
template<> struct Vec_t<double, 2>{typedef double2 type;};
template<> struct Vec_t<double, 3>{typedef double3 type;};
template<> struct Vec_t<double, 4>{typedef double4 type;};

template<> struct Vec_t<char, 1>{typedef char1 type;};
template<> struct Vec_t<char, 2>{typedef char2 type;};
template<> struct Vec_t<char, 3>{typedef char3 type;};
template<> struct Vec_t<char, 4>{typedef char4 type;};

template<> struct Vec_t<unsigned char, 1>{typedef uchar1 type;};
template<> struct Vec_t<unsigned char, 2>{typedef uchar2 type;};
template<> struct Vec_t<unsigned char, 3>{typedef uchar3 type;};
template<> struct Vec_t<unsigned char, 4>{typedef uchar4 type;};

template<> struct Vec_t<short, 1>{typedef short1 type;};
template<> struct Vec_t<short, 2>{typedef short2 type;};
template<> struct Vec_t<short, 3>{typedef short3 type;};
template<> struct Vec_t<short, 4>{typedef short4 type;};

template<> struct Vec_t<unsigned short, 1>{typedef ushort1 type;};
template<> struct Vec_t<unsigned short, 2>{typedef ushort2 type;};
template<> struct Vec_t<unsigned short, 3>{typedef ushort3 type;};
template<> struct Vec_t<unsigned short, 4>{typedef ushort4 type;};

template<> struct Vec_t<int, 1>{typedef int1 type;};
template<> struct Vec_t<int, 2>{typedef int2 type;};
template<> struct Vec_t<int, 3>{typedef int3 type;};
template<> struct Vec_t<int, 4>{typedef int4 type;};

template<> struct Vec_t<unsigned int, 1>{typedef uint1 type;};
template<> struct Vec_t<unsigned int, 2>{typedef uint2 type;};
template<> struct Vec_t<unsigned int, 3>{typedef uint3 type;};
template<> struct Vec_t<unsigned int, 4>{typedef uint4 type;};

template<> struct Vec_t<long, 1>{typedef long1 type;};
template<> struct Vec_t<long, 2>{typedef long2 type;};
template<> struct Vec_t<long, 3>{typedef long3 type;};
template<> struct Vec_t<long, 4>{typedef long4 type;};

template<> struct Vec_t<unsigned long, 1>{typedef ulong1 type;};
template<> struct Vec_t<unsigned long, 2>{typedef ulong2 type;};
template<> struct Vec_t<unsigned long, 3>{typedef ulong3 type;};
template<> struct Vec_t<unsigned long, 4>{typedef ulong4 type;};

template<> struct Vec_t<long long, 1>{typedef longlong1 type;};
template<> struct Vec_t<long long, 2>{typedef longlong2 type;};
template<> struct Vec_t<long long, 3>{typedef longlong3 type;};
template<> struct Vec_t<long long, 4>{typedef longlong4 type;};

template<> struct Vec_t<unsigned long long, 1>{typedef ulonglong1 type;};
template<> struct Vec_t<unsigned long long, 2>{typedef ulonglong2 type;};
template<> struct Vec_t<unsigned long long, 3>{typedef ulonglong3 type;};
template<> struct Vec_t<unsigned long long, 4>{typedef ulonglong4 type;};
