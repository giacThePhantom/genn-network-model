#include "definitionsInternal.h"
#include <iostream>
#include <random>
#include <cstdint>

struct MergedNeuronInitGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    double* inSynInSyn0;
    double* inSynInSyn1;
    double* inSynInSyn2;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    double* inSynInSyn0;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronInitGroup3
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    double* inSynInSyn0;
    double* inSynInSyn1;
    unsigned int numNeurons;
    
}
;
struct MergedSynapseDenseInitGroup0
 {
    scalar* g;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup1
 {
    unsigned int* rowLength;
    uint32_t* ind;
    scalar n_trg;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup2
 {
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, double* inSynInSyn1, double* inSynInSyn2, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, rng, inSynInSyn0, inSynInSyn1, inSynInSyn2, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup0, &group, sizeof(MergedNeuronInitGroup0), idx * sizeof(MergedNeuronInitGroup0)));
}
__device__ __constant__ MergedNeuronInitGroup1 d_mergedNeuronInitGroup1[1];
void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, unsigned int numNeurons) {
    MergedNeuronInitGroup1 group = {spkCnt, spk, rng, inSynInSyn0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup1, &group, sizeof(MergedNeuronInitGroup1), idx * sizeof(MergedNeuronInitGroup1)));
}
__device__ __constant__ MergedNeuronInitGroup2 d_mergedNeuronInitGroup2[1];
void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons) {
    MergedNeuronInitGroup2 group = {spkCnt, spk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup2, &group, sizeof(MergedNeuronInitGroup2), idx * sizeof(MergedNeuronInitGroup2)));
}
__device__ __constant__ MergedNeuronInitGroup3 d_mergedNeuronInitGroup3[1];
void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup3 group = {spkCnt, spk, rng, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup3, &group, sizeof(MergedNeuronInitGroup3), idx * sizeof(MergedNeuronInitGroup3)));
}
__device__ __constant__ MergedSynapseDenseInitGroup0 d_mergedSynapseDenseInitGroup0[2];
void pushMergedSynapseDenseInitGroup0ToDevice(unsigned int idx, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseDenseInitGroup0 group = {g, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseDenseInitGroup0, &group, sizeof(MergedSynapseDenseInitGroup0), idx * sizeof(MergedSynapseDenseInitGroup0)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup0 d_mergedSynapseConnectivityInitGroup0[1];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup0 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup0, &group, sizeof(MergedSynapseConnectivityInitGroup0), idx * sizeof(MergedSynapseConnectivityInitGroup0)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup1 d_mergedSynapseConnectivityInitGroup1[2];
void pushMergedSynapseConnectivityInitGroup1ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, scalar n_trg, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup1 group = {rowLength, ind, n_trg, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup1, &group, sizeof(MergedSynapseConnectivityInitGroup1), idx * sizeof(MergedSynapseConnectivityInitGroup1)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup2 d_mergedSynapseConnectivityInitGroup2[1];
void pushMergedSynapseConnectivityInitGroup2ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup2 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup2, &group, sizeof(MergedSynapseConnectivityInitGroup2), idx * sizeof(MergedSynapseConnectivityInitGroup2)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {832, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {10432, };
__device__ unsigned int d_mergedNeuronInitGroupStartID3[] = {10624, };
__device__ unsigned int d_mergedSynapseDenseInitGroupStartID0[] = {14656, 18688, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID0[] = {19520, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID1[] = {20352, 21184, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID2[] = {22016, };

extern "C" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed) {
    if(threadIdx.x == 0) {
        curand_init(deviceRNGSeed, 0, 0, &d_rng);
    }
}

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 64 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 832) {
        struct MergedNeuronInitGroup0 *group = &d_mergedNeuronInitGroup0[0]; 
        const unsigned int lid = id - 0;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            curand_init(deviceRNGSeed, id, 0, &group->rng[lid]);
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                group->inSynInSyn0[lid] = 0.00000000000000000e+00;
            }
             {
                group->inSynInSyn1[lid] = 0.00000000000000000e+00;
            }
             {
                group->inSynInSyn2[lid] = 0.00000000000000000e+00;
            }
            // current source variables
        }
    }
    // merged1
    if(id >= 832 && id < 10432) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 832;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            curand_init(deviceRNGSeed, id, 0, &group->rng[lid]);
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                group->inSynInSyn0[lid] = 0.00000000000000000e+00;
            }
            // current source variables
        }
    }
    // merged2
    if(id >= 10432 && id < 10624) {
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[0]; 
        const unsigned int lid = id - 10432;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
            // current source variables
        }
    }
    // merged3
    if(id >= 10624 && id < 14656) {
        struct MergedNeuronInitGroup3 *group = &d_mergedNeuronInitGroup3[0]; 
        const unsigned int lid = id - 10624;
        // only do this for existing neurons
        if(lid < group->numNeurons) {
            curand_init(deviceRNGSeed, id, 0, &group->rng[lid]);
            if(lid == 0) {
                group->spkCnt[0] = 0;
            }
            group->spk[lid] = 0;
             {
                group->inSynInSyn0[lid] = 0.00000000000000000e+00;
            }
             {
                group->inSynInSyn1[lid] = 0.00000000000000000e+00;
            }
            // current source variables
        }
    }
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // merged0
    if(id >= 14656 && id < 19520) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseDenseInitGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseDenseInitGroup0 *group = &d_mergedSynapseDenseInitGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseDenseInitGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing postsynaptic neurons
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    scalar initVal;
                    initVal = (0.00000000000000000e+00);
                    group->g[(i * group->rowStride) + lid] = initVal;
                }
            }
        }
    }
    
    // ------------------------------------------------------------------------
    // Synapse groups with kernel connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with sparse connectivity
    // merged0
    if(id >= 19520 && id < 20352) {
        struct MergedSynapseConnectivityInitGroup0 *group = &d_mergedSynapseConnectivityInitGroup0[0]; 
        const unsigned int lid = id - 19520;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            // Build sparse connectivity
            while(true) {
                const unsigned int offset= (unsigned int) lid/((unsigned int) (5.00000000000000000e+00))*(2.50000000000000000e+01);
                for (unsigned int k= 0; k < (2.50000000000000000e+01); k++) {
                do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = (offset+k);
                    group->rowLength[lid]++;
                }
                while(false);
                }
                break;
            }
        }
    }
    // merged1
    if(id >= 20352 && id < 22016) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseConnectivityInitGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseConnectivityInitGroup1 *group = &d_mergedSynapseConnectivityInitGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseConnectivityInitGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        // only do this for existing postsynaptic neurons
        if(lid < group->numTrgNeurons) {
            curandStatePhilox4_32_10_t localRNG = d_rng;
            skipahead_sequence((unsigned long long)id, &localRNG);
            // Build sparse connectivity
            unsigned int c = (1.20000000000000000e+01);
            while(true) {
                if (c == 0) {
                break;
                }
                const unsigned int glo= lid/((unsigned int) group->n_trg);
                const unsigned int offset= (6.00000000000000000e+01)*glo;
                const unsigned int tid= curand_uniform_double(&localRNG)*(6.00000000000000000e+01);
                do {
                    const unsigned int idx = ((offset+tid+0) * group->rowStride) + group->rowLength[offset+tid+0];
                    group->ind[((offset+tid+0) * group->rowStride) + atomicAdd(&group->rowLength[offset+tid+0], 1)] = lid;}
                while(false);
                c--;
            }
        }
    }
    // merged2
    if(id >= 22016 && id < 22208) {
        struct MergedSynapseConnectivityInitGroup2 *group = &d_mergedSynapseConnectivityInitGroup2[0]; 
        const unsigned int lid = id - 22016;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            // Build sparse connectivity
            while(true) {
                const unsigned int row_length= group->numTrgNeurons/group->numSrcNeurons;
                const unsigned int offset= lid*row_length;
                for (unsigned int k= 0; k < row_length; k++) {
                    do {
                    const unsigned int idx = (lid * group->rowStride) + group->rowLength[lid];
                    group->ind[idx] = (offset + k);
                    group->rowLength[lid]++;
                }
                while(false);
                }
                break;
            }
        }
    }
    
}
void initialize() {
    unsigned long long deviceRNGSeed = 0;
     {
        std::random_device seedSource;
        uint32_t *deviceRNGSeedWord = reinterpret_cast<uint32_t*>(&deviceRNGSeed);
        for(int i = 0; i < 2; i++) {
            deviceRNGSeedWord[i] = seedSource();
        }
    }
    initializeRNGKernel<<<1, 1>>>(deviceRNGSeed);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    CHECK_CUDA_ERRORS(cudaMemset(d_rowLengthorn_ln, 0, 9600 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMemset(d_rowLengthorn_pn, 0, 9600 * sizeof(unsigned int)));
     {
        const dim3 threads(64, 1);
        const dim3 grid(347, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
}
