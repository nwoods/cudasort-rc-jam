#ifndef CUDASORT_RC_JAM_SERIAL_RADIX_SORT__
#define CUDASORT_RC_JAM_SERIAL_RADIX_SORT__

#include <iostream>
#include <cstring>

uint32_t get_bucket_index(uint32_t num, uint32_t shift, uint32_t mask)
{
    return (num >> shift) & mask;
}

void count_elements(uint32_t *in, uint32_t *counts, size_t size, uint32_t shift, uint32_t num_buckets, uint32_t mask)
{
    // initialize counts to zero
    std::memset(counts, 0, num_buckets * sizeof(counts));

    for (uint32_t i = 0; i < size; i++)
    {
        counts[get_bucket_index(in[i], shift, mask)]++;
    }

    // cumulative counts
    uint32_t total = 0;
    for (uint32_t i = 0; i < num_buckets; i++)
    {
        uint32_t count = counts[i];
        counts[i] = total;
        total += count;
    }
}

void distribute(const uint32_t *in, uint32_t *out, uint32_t *counts, uint32_t size, uint32_t shift, uint32_t mask)
{
    for (uint32_t i = 0; i < size; i++)
    {
        uint32_t num = in[i];
        uint32_t bucket = get_bucket_index(num, shift, mask);
        out[counts[bucket]++] = num;
    }
}

void radix_sort_serial(const uint32_t *in, uint32_t *out, size_t len, uint32_t bits_per_pass)
{
    uint32_t num_buckets = (1 << bits_per_pass), mask = (1 << bits_per_pass) - 1;
    uint32_t *counts = new uint32_t[num_buckets];
    uint32_t *temp = new uint32_t[len];

    uint32_t num_passes = (32 + bits_per_pass - 1) / bits_per_pass;

    std::memcpy(out, in, len * sizeof(uint32_t));
    uint32_t* source = out;
    uint32_t* dest = temp;

    for (uint32_t pass = 0; pass < num_passes; pass++)
    {
        uint32_t shift = pass * bits_per_pass;
        count_elements(source, counts, len, shift, num_buckets, mask);
        distribute(source, dest, counts, len, shift, mask);
        uint32_t *t = source;
        source = dest;
        dest = t;
    }

    // if we have data in temp copy it out to result
    if (source != out)
    {
        std::memcpy(out, temp, len * sizeof(uint32_t));
    }
}

#endif // header guard