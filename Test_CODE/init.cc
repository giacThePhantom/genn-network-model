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
    double* inSynInSyn2;
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
struct MergedSynapseDenseInitGroup1
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
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseConnectivityInitGroup2
 {
    unsigned int* rowLength;
    uint32_t* ind;
    scalar n_orn;
    scalar n_trg;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseSparseInitGroup0
 {
    unsigned int* rowLength;
    uint32_t* ind;
    scalar* g;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    unsigned int colStride;
    
}
;
__device__ __constant__ MergedNeuronInitGroup0 d_mergedNeuronInitGroup0[1];
void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronInitGroup0 group = {spkCnt, spk, rng, inSynInSyn0, inSynInSyn1, numNeurons, };
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
void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, double* inSynInSyn1, double* inSynInSyn2, unsigned int numNeurons) {
    MergedNeuronInitGroup3 group = {spkCnt, spk, rng, inSynInSyn0, inSynInSyn1, inSynInSyn2, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronInitGroup3, &group, sizeof(MergedNeuronInitGroup3), idx * sizeof(MergedNeuronInitGroup3)));
}
__device__ __constant__ MergedSynapseDenseInitGroup0 d_mergedSynapseDenseInitGroup0[1];
void pushMergedSynapseDenseInitGroup0ToDevice(unsigned int idx, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseDenseInitGroup0 group = {g, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseDenseInitGroup0, &group, sizeof(MergedSynapseDenseInitGroup0), idx * sizeof(MergedSynapseDenseInitGroup0)));
}
__device__ __constant__ MergedSynapseDenseInitGroup1 d_mergedSynapseDenseInitGroup1[1];
void pushMergedSynapseDenseInitGroup1ToDevice(unsigned int idx, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseDenseInitGroup1 group = {g, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseDenseInitGroup1, &group, sizeof(MergedSynapseDenseInitGroup1), idx * sizeof(MergedSynapseDenseInitGroup1)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup0 d_mergedSynapseConnectivityInitGroup0[1];
void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup0 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup0, &group, sizeof(MergedSynapseConnectivityInitGroup0), idx * sizeof(MergedSynapseConnectivityInitGroup0)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup1 d_mergedSynapseConnectivityInitGroup1[1];
void pushMergedSynapseConnectivityInitGroup1ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup1 group = {rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup1, &group, sizeof(MergedSynapseConnectivityInitGroup1), idx * sizeof(MergedSynapseConnectivityInitGroup1)));
}
__device__ __constant__ MergedSynapseConnectivityInitGroup2 d_mergedSynapseConnectivityInitGroup2[2];
void pushMergedSynapseConnectivityInitGroup2ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, scalar n_orn, scalar n_trg, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseConnectivityInitGroup2 group = {rowLength, ind, n_orn, n_trg, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseConnectivityInitGroup2, &group, sizeof(MergedSynapseConnectivityInitGroup2), idx * sizeof(MergedSynapseConnectivityInitGroup2)));
}
__device__ __constant__ MergedSynapseSparseInitGroup0 d_mergedSynapseSparseInitGroup0[1];
void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int colStride) {
    MergedSynapseSparseInitGroup0 group = {rowLength, ind, g, rowStride, numSrcNeurons, numTrgNeurons, colStride, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseSparseInitGroup0, &group, sizeof(MergedSynapseSparseInitGroup0), idx * sizeof(MergedSynapseSparseInitGroup0)));
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
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ unsigned int d_mergedNeuronInitGroupStartID0[] = {0, };
__device__ unsigned int d_mergedNeuronInitGroupStartID1[] = {32, };
__device__ unsigned int d_mergedNeuronInitGroupStartID2[] = {64, };
__device__ unsigned int d_mergedNeuronInitGroupStartID3[] = {224, };
__device__ unsigned int d_mergedSynapseDenseInitGroupStartID0[] = {256, };
__device__ unsigned int d_mergedSynapseDenseInitGroupStartID1[] = {288, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID0[] = {320, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID1[] = {352, };
__device__ unsigned int d_mergedSynapseConnectivityInitGroupStartID2[] = {512, 544, };
__device__ unsigned int d_mergedSynapseSparseInitGroupStartID0[] = {0, };

extern "C" __global__ void initializeRNGKernel(unsigned long long deviceRNGSeed) {
    if(threadIdx.x == 0) {
        curand_init(deviceRNGSeed, 0, 0, &d_rng);
    }
}

extern "C" __global__ void initializeKernel(unsigned long long deviceRNGSeed) {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // ------------------------------------------------------------------------
    // Local neuron groups
    // merged0
    if(id < 32) {
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
            // current source variables
        }
    }
    // merged1
    if(id >= 32 && id < 64) {
        struct MergedNeuronInitGroup1 *group = &d_mergedNeuronInitGroup1[0]; 
        const unsigned int lid = id - 32;
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
    if(id >= 64 && id < 224) {
        struct MergedNeuronInitGroup2 *group = &d_mergedNeuronInitGroup2[0]; 
        const unsigned int lid = id - 64;
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
    if(id >= 224 && id < 256) {
        struct MergedNeuronInitGroup3 *group = &d_mergedNeuronInitGroup3[0]; 
        const unsigned int lid = id - 224;
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
    
    // ------------------------------------------------------------------------
    // Custom update groups
    
    // ------------------------------------------------------------------------
    // Custom WU update groups with dense connectivity
    
    // ------------------------------------------------------------------------
    // Synapse groups with dense connectivity
    // merged0
    if(id >= 256 && id < 288) {
        struct MergedSynapseDenseInitGroup0 *group = &d_mergedSynapseDenseInitGroup0[0]; 
        const unsigned int lid = id - 256;
        // only do this for existing postsynaptic neurons
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    scalar initVal;
                    const unsigned int npn= (unsigned int) (2.10000000000000000e+01);
                    const unsigned int nln= (unsigned int) (1.50000000000000000e+01);
                    initVal=(i/nln == lid/npn) ? 0.0 : (2.00000000000000016e-05);
                    group->g[(i * group->rowStride) + lid] = initVal;
                }
            }
        }
    }
    // merged1
    if(id >= 288 && id < 320) {
        struct MergedSynapseDenseInitGroup1 *group = &d_mergedSynapseDenseInitGroup1[0]; 
        const unsigned int lid = id - 288;
        // only do this for existing postsynaptic neurons
        if(lid < group->numTrgNeurons) {
            for(unsigned int i = 0; i < group->numSrcNeurons; i++) {
                 {
                    scalar initVal;
                    const unsigned int nln= (unsigned int) (1.50000000000000000e+01);
                    initVal=(i/nln == lid/nln) ? 0.0 : (2.00000000000000016e-05);
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
    if(id >= 320 && id < 352) {
        struct MergedSynapseConnectivityInitGroup0 *group = &d_mergedSynapseConnectivityInitGroup0[0]; 
        const unsigned int lid = id - 320;
        // only do this for existing presynaptic neurons
        if(lid < group->numSrcNeurons) {
            group->rowLength[lid] = 0;
            // Build sparse connectivity
            while(true) {
                const unsigned int offset= (unsigned int) lid/((unsigned int) (2.10000000000000000e+01))*(1.50000000000000000e+01);
                for (unsigned int k= 0; k < (1.50000000000000000e+01); k++) {
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
    if(id >= 352 && id < 512) {
        struct MergedSynapseConnectivityInitGroup1 *group = &d_mergedSynapseConnectivityInitGroup1[0]; 
        const unsigned int lid = id - 352;
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
    // merged2
    if(id >= 512 && id < 576) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedSynapseConnectivityInitGroupStartID2[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedSynapseConnectivityInitGroup2 *group = &d_mergedSynapseConnectivityInitGroup2[lo - 1]; 
        const unsigned int groupStartID = d_mergedSynapseConnectivityInitGroupStartID2[lo - 1];
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
                const unsigned int offset= group->n_orn*glo;
                const unsigned int tid= curand_uniform_double(&localRNG)*group->n_orn;
                do {
                    const unsigned int idx = ((offset+tid+0) * group->rowStride) + group->rowLength[offset+tid+0];
                    group->ind[((offset+tid+0) * group->rowStride) + atomicAdd(&group->rowLength[offset+tid+0], 1)] = lid;}
                while(false);
                c--;
            }
        }
    }
    
}
extern "C" __global__ void initializeSparseKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shRowLength[32];
    // merged0
    if(id < 32) {
        struct MergedSynapseSparseInitGroup0 *group = &d_mergedSynapseSparseInitGroup0[0]; 
        const unsigned int lid = id - 0;
        const unsigned int numBlocks = (group->numSrcNeurons + 32 - 1) / 32;
        unsigned int idx = lid;
        for(unsigned int r = 0; r < numBlocks; r++) {
            const unsigned numRowsInBlock = (r == (numBlocks - 1)) ? ((group->numSrcNeurons - 1) % 32) + 1 : 32;
            __syncthreads();
            if (threadIdx.x < numRowsInBlock) {
                shRowLength[threadIdx.x] = group->rowLength[(r * 32) + threadIdx.x];
            }
            __syncthreads();
            for(unsigned int i = 0; i < numRowsInBlock; i++) {
                if(lid < shRowLength[i]) {
                     {
                        scalar initVal;
                        initVal = (1.00000000000000002e-03);
                        group->g[(((r * 32) + i) * group->rowStride) + lid] = initVal;
                    }
                }
                idx += group->rowStride;
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
    CHECK_CUDA_ERRORS(cudaMemset(d_rowLengthor_ln, 0, 160 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMemset(d_rowLengthor_pn, 0, 160 * sizeof(unsigned int)));
     {
        const dim3 threads(32, 1);
        const dim3 grid(18, 1);
        initializeKernel<<<grid, threads>>>(deviceRNGSeed);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}

void initializeSparse() {
    copyStateToDevice(true);
    copyConnectivityToDevice(true);
    
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        initializeSparseKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
