#ifndef _PCG_TINY_H_
#define _PCG_TINY_H_

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;

#define PCG32_INITIALIZER { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

extern pcg32_random_t rng_global;

uint32_t pcg32_random_r(pcg32_random_t* rng);

uint32_t pcg32_random();

void pcg32_srandom_r(pcg32_random_t* rng,
                     uint64_t initstate,
                     uint64_t initseq);

#endif
