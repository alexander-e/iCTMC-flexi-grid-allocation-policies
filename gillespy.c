/*  A C implementation (with a Python interface) of the Gillespie method with
    batch-mean estimation method to determine estimates of the blocking probabilities
    as studied in [1].

    The recommended way to compile this .c file for use in Python is:
    gcc -shared -fPIC -Wall -O3 -funroll-loops gillespy.c -I/usr/include/python3.5 -o gillespy.so

    This code is loosely based on https://www.r-bloggers.com/vanilla-c-code-for-the-stochastic-simulation-algorithm/

    References
    ----------
    .. [1] Alexander Erreygers, Cristina Rottondi, Giacomo Verticale
           and Jasper De Bock. ``Imprecise Markov Models for Scalable
           and Robust Performance Evaluation of Flexi-Grid Spectrum
           Allocation Policies''. arXiv:?.

    Copyright (C) 2018 Alexander Erreygers

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "Python.h"

/*  Random number generators.

    This file can use two Random number generators: pcg_32 or
    xoroshiro128+. The second one is probably preferred.
*/

typedef struct { uint64_t s0;  uint64_t s1; } rng_t;

// Minimal pcg32 code
/* This code is slightly adapted from http://www.pcg-random.org/download.html.

   Original copyright:

   *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
   Licensed under Apache License 2.0 (NO WARRANTY, etc. see website) */

double pcg32(rng_t* rng)
{
    uint64_t oldstate = rng->s0;
    // Advance internal state
    rng->s0 = oldstate * 6364136223846793005ULL + (rng->s1|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    const uint32_t result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

    // Conversion based on random_real_64 of http://xoroshiro.di.unimi.it/random_real.c
    // Only 32 bits instead of 64!
    return ldexp((double) result, -32);
}

// Minimal xoroshiro128+ code
/* This code is slightly adapted from http://xoroshiro.di.unimi.it/xoroshiro128plus.c

   Original copyright:

   Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org)
   To the extent possible under law, the author has dedicated all copyright
   and related and neighboring rights to this software to the public domain
   worldwide. This software is distributed without any warranty.

   See <http://creativecommons.org/publicdomain/zero/1.0/>.

   This is the successor to xorshift128+. It is the fastest full-period
   generator passing BigCrush without systematic failures, but due to the
   relatively short period it is acceptable only for applications with a
   mild amount of parallelism; otherwise, use a xorshift1024* generator.

   Beside passing BigCrush, this generator passes the PractRand test suite
   up to (and included) 16TB, with the exception of binary rank tests,
   which fail due to the lowest bit being an LFSR; all other bits pass all
   tests. We suggest to use a sign test to extract a random Boolean value.

   Note that the generator uses a simulated rotate operation, which most C
   compilers will turn into a single instruction. In Java, you can use
   Long.rotateLeft(). In languages that do not make low-level rotation
   instructions accessible xorshift128+ could be faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

double xoroshiro128plus(rng_t* rng) {
    const uint64_t s0 = rng->s0;
    uint64_t s1 = rng->s1;
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    rng->s0 = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    rng->s1 = rotl(s1, 36); // c

    // Conversion to [0, 1) based on http://xoroshiro.di.unimi.it/random_real.c
    return ldexp((double)(result & ((1ULL << 53) - 1)), -53);
}

/* Various methods used throughout the code. */

// Use binary search as the arrays are sorted
unsigned int bisect_left_double(double arr[], const double maxi, const unsigned int len, const double u)
{
    unsigned int lo = 0, mid, hi = len-1;

    // Based on https://hg.python.org/cpython/file/3.5/Lib/bisect.py
    while (lo < hi) {
        mid = ((unsigned int) lo + hi) / 2;
        if (arr[mid] < u * maxi) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// Determine the mean and standard deviation of batches in the batch mean method
void determine_mean_and_sd(double bp1s[], double bp2s[], const unsigned int num, double res[]) {
    double bp1=0, bp2=0, sd1=0, sd2=0;
    unsigned int i;

    for (i=0; i < num; i++) {
        bp1 += bp1s[i];
        bp2 += bp2s[i];
    }
    bp1 = bp1 / num;
    bp2 = bp2 / num;

    for (i=0; i < num; i++) {
        sd1 += pow(bp1s[i] - bp1, 2);
        sd2 += pow(bp2s[i] - bp2, 2);
    }

    res[0] = bp1;
    res[1] = sqrt(sd1 / (num - 1));
    res[2] = bp2;
    res[3] = sqrt(sd2 / (num - 1));
}

/* Functions concerning allocation policies. */

// Random allocation policy
unsigned int allocate_RA(unsigned int state[], const unsigned int n2, rng_t* rng) {
    unsigned int j, sumstates = 0, cdf2[n2];
    unsigned int lo = 0, mid, hi = n2-1;

    // Generate a random number
    // double v = pcg32(&rng);
    double v = xoroshiro128plus(rng);
    // double v = (double) rand() / (double) RAND_MAX;

    // Get new random allocation
    for (j=0; j < n2; j++){
        sumstates += (unsigned int) (n2 - j) * state[j];
        cdf2[j] = sumstates;
    }

    // Based on https://hg.python.org/cpython/file/3.5/Lib/bisect.py
    while (lo < hi) {
        mid = (unsigned int) (lo + hi) / 2;
        if (cdf2[mid] < (double) v * sumstates) lo = mid + 1;
        else hi = mid;
    }

    return lo;
}

// Least filled allocation policy
unsigned int allocate_LF(unsigned int state[], const unsigned int n2, rng_t* rng) {
    unsigned int j, alloc = 0;

    // Look for the least-filled slot
    for (j=1; j < n2; j++) {
        if (state[j] > 0){
            alloc = j;
            break;
        }
    }
    return alloc;
}

// Most filled allocation policy
unsigned int allocate_MF(unsigned int state[], const unsigned int n2, rng_t* rng) {
    unsigned int j, alloc = 0;

    // Look for the least-filled slot
    for (j=n2-1; j > 0; j--) {
        if (state[j] > 0){
            alloc = j;
            break;
        }
    }
    return alloc;
}

// Naive Gillespie approximation, solely for testing purposes
static PyObject* py_gillespie_naive(PyObject* self, PyObject* args)
{

    unsigned int m1, n2;
    double lambda1, lambda2, mu1, mu2;
    unsigned long num_events;
    char *pol;
    const uint64_t useed1, useed2, vseed1, vseed2;

    if (!PyArg_ParseTuple(args, "IIddddskKKKK", &m1, &n2, &lambda1, &lambda2, &mu1, &mu2, &pol, &num_events, &useed1, &useed2, &vseed1, &vseed2)) {
        return NULL;
    }

    // Allocation
    unsigned int (*allocate)(unsigned int*, const unsigned int, rng_t*);

    if (strcmp(pol, "RA") == 0) {
        allocate = &allocate_RA;
    } else if (strcmp(pol, "LF") == 0) {
        allocate = &allocate_LF;
    } else {
        allocate = &allocate_MF;
    }

    unsigned long i, arrivals = 0, loss1 = 0, loss2 = 0;
    // unsigned long a_arr = 0, a_loss1 = 0, a_loss2 = 0;
    // unsigned long b_arr = 0, b_loss1 = 0, b_loss2 = 0;
    unsigned int alloc, j, sumstates;
    int the_event;
    double u, bp1 = 0., bp2 = 0., cumsum;
    // double a_bp1=0., a_bp2 = 0., b_bp1 = 0., b_bp2 = 0.;

    double weights[n2+3], cdf1[n2+3];

    unsigned int state[n2+1];

    unsigned int m2 = m1 / n2;

    // Initialise the weights
    weights[0] = lambda1;
    weights[1] = lambda2;

    state[0] = m2;
    for (j=1; j < n2+1; j++) {
        state[j] = 0;
    }

    /* Initialise rngs:
       Four options! Pick one
    */

    rng_t rng1, rng2;
    if ((useed1 == 0) && (useed2 == 0) && (vseed1 == 0) && (vseed2 == 0)) {
        rng1.s0 = time(NULL);
        rng1.s1 = (uintptr_t) &rng1;
        rng2.s0 = time(NULL);
        rng2.s1 = (uintptr_t) &rng2;
    } else {
        rng1.s0 = useed1;
        rng1.s1 = useed2;
        rng2.s0 = vseed1;
        rng2.s1 = vseed2;
    }

    for(i=0; i < num_events; i++) {
        // for (j=0; j<n2+1; j++) {
        //     printf("%u, ", state[j]);
        // }
        // printf("\n");

        // Generate a random number
        // u = pcg32(&rng1);
        u = xoroshiro128plus(&rng1);
        // u = (double) rand() / (double) RAND_MAX;

        // Correct the weights
        sumstates = 0;
        for (j=0; j < n2+1; j++) {
            sumstates += state[j];
        }
        weights[2] = (double) mu2 * (m2 - sumstates);
        for (j=3; j < n2+3; j++) {
            weights[j] = (double) mu1 * (j-2) * state[j-2];
        }

        // Determine the cdf
        cumsum = 0.0;
        for (j=0; j < n2+3; j++){
            cumsum += weights[j];
            cdf1[j] = cumsum;
        }
        // // Alternative to using the bisection method
        // for (j=0; j < n2+3; j++) {
        //     if (u < (double) cdf1[j] / cumsum) {
        //         the_event = (int) j - 2;
        //         break;
        //     }
        // }
        // Bisection method
        the_event = (int) bisect_left_double(cdf1, cumsum, n2+3, u) - 2;

        switch (the_event) {
            case -2 :
                // // Code with two estimators
                // a_arr += 1;
                // if (sumstates - state[n2] < 1)
                //     a_loss1 += 1;
                // if (state[0] < 1)
                //     a_loss2 += 1;
                // Original code
                arrivals += 1;
                if (state[0] < 1) {
                    loss2 += 1;
                }
                // sumstates -= state[n2];
                if (sumstates - state[n2] < 1) {
                    // printf("Blocking of type 1\n");
                    loss1 += 1;
                } else {
                    // Get new random allocation
                    sumstates = 0;
                    alloc = allocate(state, n2, &rng2);

                    // printf("Arrival of type 1 at %i \n", alloc);
                    state[alloc] -= 1;
                    state[alloc+1] += 1;
                }
                break;
            case -1:
                // // Code with two estimators
                // b_arr += 1;
                // if (sumstates - state[n2] < 1)
                //     b_loss1 += 1;
                // if (state[0] < 1)
                //     b_loss2 += 1;
                // Original code
                arrivals += 1;
                if (sumstates - state[n2] < 1) {
                    loss1 += 1;
                }
                if (state[0] < 1) {
                    // printf("Blocking of type 2\n");
                    loss2 += 1;
                } else {
                    // printf("Arrival of type 2\n");
                    state[0] -= 1;
                }
                break;
            case 0 :
                // printf("Departure of type 2\n");
                state[0] += 1;
                break;
            default:
                // printf("Departure of type 1 at %i \n", the_event);
                state[the_event] -= 1;
                state[the_event-1] += 1;
        }

    }

    if (arrivals > 0) {
        bp1 = (double) loss1 / arrivals;
        bp2 = (double) loss2 / arrivals;
    }

    printf("%3lu, %3lu, %3lu \n", loss1, loss2, arrivals);
    printf("%3.7g, %3.7g \n", bp1, bp2);


    // printf("Alternative: \n");
    // if (a_arr > 0) {
    //     a_bp1 = (double) a_loss1 / a_arr;
    //     a_bp2 = (double) a_loss2 / a_arr;
    // }
    // if (b_arr > 0) {
    //     b_bp1 = (double) b_loss1 / b_arr;
    //     b_bp2 = (double) b_loss2 / b_arr;
    // }

    // printf("%3lu, %3lu, %3lu \n", a_loss1, a_loss2, a_arr);
    // printf("%3.7g, %3.7g \n", a_bp1, a_bp2);
    // printf("%3lu, %3lu, %3lu \n", b_loss1, b_loss2, b_arr);
    // printf("%3.7g, %3.7g \n", b_bp1, b_bp2);

    return Py_BuildValue("(III)", loss1, loss2, arrivals);
}

// Batch mean Gillespie approximation method
static PyObject* py_gillespie_batchmean(PyObject* self, PyObject* args)
{

    const unsigned int m1, n2, min_batches, max_batches;
    const double lambda1, lambda2, mu1, mu2, rel_accur, conf_factor;
    const unsigned long batch_size;
    const char *pol;
    const uint64_t useed1, useed2, vseed1, vseed2;


    if (!PyArg_ParseTuple(args, "IIddddskIIddKKKK", &m1, &n2, &lambda1, &lambda2, &mu1, &mu2, &pol, &batch_size, &min_batches, &max_batches, &rel_accur, &conf_factor, &useed1, &useed2, &vseed1, &vseed2)) {
        return NULL;
    }

    if (min_batches > max_batches) {
        return NULL;
    }

    if ((mu1 < 0) || (mu2 < 0) || (lambda1 < 0) || (lambda2 < 0) || (rel_accur <= 0)) {
        return NULL;
    }

    const unsigned int m2 = m1 / n2;

    // Allocation
    unsigned int (*allocate)(unsigned int*, const unsigned int, rng_t*);

    if (strcmp(pol, "RA") == 0) {
        allocate = &allocate_RA;
    } else if (strcmp(pol, "LF") == 0) {
        allocate = &allocate_LF;
    } else {
        allocate = &allocate_MF;
    }

    // Variables used for the simulations
    unsigned int alloc, j, sumstates;
    int the_event;
    double u, cumsum;
    // double a_bp1=0., a_bp2 = 0., b_bp1 = 0., b_bp2 = 0.;

    double weights[n2+3], cdf1[n2+3];
    weights[0] = lambda1;
    weights[1] = lambda2;

    // The state space
    unsigned int state[n2+1];
    state[0] = m2;
    for (j=1; j < n2+1; j++) {
        state[j] = 0;
    }

    // Initialise rngs
    rng_t rng1, rng2;
    if ((useed1 == 0) && (useed2 == 0) && (vseed1 == 0) && (vseed2 == 0)) {
        rng1.s0 = time(NULL);
        rng1.s1 = (uintptr_t) &rng1;
        rng2.s0 = time(NULL);
        rng2.s1 = (uintptr_t) &rng2;
    } else {
        rng1.s0 = useed1;
        rng1.s1 = useed2;
        rng2.s0 = vseed1;
        rng2.s1 = vseed2;
    }

    // Variables used for the batches
    unsigned long arrivals = 0, loss1 = 0, loss2 = 0;
    // unsigned long a_arr = 0, a_loss1 = 0, a_loss2 = 0;
    // unsigned long b_arr = 0, b_loss1 = 0, b_loss2 = 0;
    unsigned int num_batches = 0;
    double bp1s[max_batches], bp2s[max_batches];
    int burn_in = 1, des_accur = 0;
    unsigned long bi_loss2[2] = {batch_size, 0};
    double results[4];

    // Results
    double bp1=0, bp2=0, sd1=0, sd2=0, err1=0, err2=0;

    // Simulating a single batch
    while (!des_accur && (num_batches < max_batches)) {

        arrivals = 0;
        loss1 = 0;
        loss2 = 0;
        while (arrivals < batch_size) {
            // Generate a random number
            // u = pcg32(&rng1);
            u = xoroshiro128plus(&rng1);
            // u = (double) rand() / (double) RAND_MAX;

            // Correct the weights
            sumstates = 0;
            for (j=0; j < n2+1; j++) {
                sumstates += state[j];
            }
            weights[2] = (double) mu2 * (m2 - sumstates);
            for (j=3; j < n2+3; j++) {
                weights[j] = (double) mu1 * (j-2) * state[j-2];
            }

            // Determine the cdf
            cumsum = 0.0;
            for (j=0; j < n2+3; j++) {
                cumsum += weights[j];
                cdf1[j] = cumsum;
            }
            // // Alternative to using the bisection method
            // for (j=0; j < n2+3; j++) {
            //     if (u < (double) cdf1[j] / cumsum) {
            //         the_event = (int) j - 2;
            //         break;
            //     }
            // }
            // Bisection method to determine the occuring event
            the_event = (int) bisect_left_double(cdf1, cumsum, n2+3, u) - 2;

            switch (the_event) {
                case -2 :
                    // // Code with two estimators
                    // a_arr += 1;
                    // if (sumstates - state[n2] < 1)
                    //     a_loss1 += 1;
                    // if (state[0] < 1)
                    //     a_loss2 += 1;
                    // Original code
                    arrivals += 1;
                    if (state[0] < 1) {
                        loss2 += 1;
                    }
                    // sumstates -= state[n2];
                    if (sumstates - state[n2] < 1) {
                        // printf("Blocking of type 1\n");
                        loss1 += 1;
                    } else {
                        // Get new random allocation
                        alloc = allocate(state, n2, &rng2);
                        // printf("Arrival of type 1 at %i \n", alloc);
                        state[alloc] -= 1;
                        state[alloc+1] += 1;
                    }
                    break;
                case -1:
                    // // Code with two estimators
                    // b_arr += 1;
                    // if (sumstates - state[n2] < 1)
                    //     b_loss1 += 1;
                    // if (state[0] < 1)
                    //     b_loss2 += 1;
                    // Original code
                    arrivals += 1;
                    if (sumstates - state[n2] < 1) {
                        loss1 += 1;
                    }
                    if (state[0] < 1) {
                        // printf("Blocking of type 2\n");
                        loss2 += 1;
                    } else {
                        // printf("Arrival of type 2\n");
                        state[0] -= 1;
                    }
                    break;
                case 0 :
                    // printf("Departure of type 2\n");
                    state[0] += 1;
                    break;
                default:
                    // printf("Departure of type 1 at %i \n", the_event);
                    state[the_event] -= 1;
                    state[the_event-1] += 1;
            }
        }

        if (burn_in) {
            // We use the arrivals of type 2 to check the burn in
            // This check is based on Heuristic R1 (p. 143) of (Pawlikwski, 1990),
            // see https://dl.acm.org/citation.cfm?id=78921.
            int bi_check1 = 0, bi_check2 = 0;

            if (loss2 < bi_loss2[0]) bi_loss2[0] = loss2;
            else bi_check1 = 1;

            if (loss2 > bi_loss2[1]) bi_loss2[1] = loss2;
            else bi_check2 = 1;

            // Check whether we can assume that the burn in phase was long enough
            if (bi_check1 && bi_check2) {
                burn_in = 0;
                printf("Burn-in completed after %i batches \n", num_batches+1);
                num_batches = 0;
            } else {
                if ((num_batches+1) == min_batches) {
                    printf("Burn-in aborted after %i batches \n", num_batches+1);
                    num_batches = 0;
                    burn_in = 0;
                } else num_batches += 1;
            }
        } else {
            bp1s[num_batches] = (double) loss1 / arrivals;
            bp2s[num_batches] = (double) loss2 / arrivals;
            num_batches += 1;
            // printf("%i : %g, %g \n", num_batches, bp1s[num_batches-1], bp2s[num_batches-1]);

            if (num_batches >= min_batches) {
                determine_mean_and_sd(bp1s, bp2s, num_batches, results);
                bp1 = results[0];
                sd1 = results[1];
                bp2 = results[2];
                sd2 = results[3];
                err1 = conf_factor * sd1 / sqrt(num_batches);
                err2 = conf_factor * sd2 / sqrt(num_batches);
                // printf("%g, %g, %g ; %g, %g, %g \n", bp1, sd1, ra1, bp2, sd2, ra2);
                if ((err1 <= rel_accur * bp1) && (err2 <= rel_accur * bp2)) {
                    des_accur = 1;
                     printf("Finished after %i batches \n", num_batches);
                }
            }
        }
    }

    if (!des_accur) {
        printf("Iterations stopped after the maximum of %i batches \n", num_batches);
    }

    return Py_BuildValue("(dddd)", bp1, err1, bp2, err2);
}

// Bind Python function names to our C function
static PyMethodDef gillespyMethods[] = {
    {"gillespie_naive", py_gillespie_naive, METH_VARARGS, "Naive Gillespie."},
    {"gillespie_batchmean", py_gillespie_batchmean, METH_VARARGS, "Batch mean Gillespie."},
    {NULL, NULL, 0, NULL}
};

// Module definition structure
static struct PyModuleDef gillespymodule = {
   PyModuleDef_HEAD_INIT,
   "gillespy",   /* name of module */
   "", /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   gillespyMethods
};

// Final step required
PyMODINIT_FUNC
PyInit_gillespy(void)
{
    return PyModule_Create(&gillespymodule);
}
