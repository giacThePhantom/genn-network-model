#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedPresynapticUpdateGroup0
 {
    double* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    unsigned int* rowLength;
    uint32_t* ind;
    scalar g;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedPresynapticUpdateGroup1
 {
    double* inSyn;
    unsigned int* srcSpkCnt;
    unsigned int* srcSpk;
    scalar* g;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
struct MergedSynapseDynamicsGroup0
 {
    double* inSyn;
    scalar* raPre;
    unsigned int* rowLength;
    uint32_t* ind;
    unsigned int rowStride;
    unsigned int numSrcNeurons;
    unsigned int numTrgNeurons;
    
}
;
__device__ __constant__ MergedPresynapticUpdateGroup0 d_mergedPresynapticUpdateGroup0[3];
void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, scalar g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup0 group = {inSyn, srcSpkCnt, srcSpk, rowLength, ind, g, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup0, &group, sizeof(MergedPresynapticUpdateGroup0), idx * sizeof(MergedPresynapticUpdateGroup0)));
}
__device__ __constant__ MergedPresynapticUpdateGroup1 d_mergedPresynapticUpdateGroup1[2];
void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedPresynapticUpdateGroup1 group = {inSyn, srcSpkCnt, srcSpk, g, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedPresynapticUpdateGroup1, &group, sizeof(MergedPresynapticUpdateGroup1), idx * sizeof(MergedPresynapticUpdateGroup1)));
}
__device__ __constant__ MergedSynapseDynamicsGroup0 d_mergedSynapseDynamicsGroup0[1];
void pushMergedSynapseDynamicsGroup0ToDevice(unsigned int idx, double* inSyn, scalar* raPre, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons) {
    MergedSynapseDynamicsGroup0 group = {inSyn, raPre, rowLength, ind, rowStride, numSrcNeurons, numTrgNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedSynapseDynamicsGroup0, &group, sizeof(MergedSynapseDynamicsGroup0), idx * sizeof(MergedSynapseDynamicsGroup0)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID0[] = {0, 4000, 4800, };
__device__ __constant__ unsigned int d_mergedPresynapticUpdateGroupStartID1[] = {4832, 8832, };
__device__ __constant__ unsigned int d_mergedSynapseDynamicsGroupStartID0[] = {0, };
extern "C" __global__ void updatePresynapticKernel(double t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shRowLength[32];
    __shared__ unsigned int shSpk[32];
    // merged0
    if(id < 4832) {
        unsigned int lo = 0;
        unsigned int hi = 3;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedPresynapticUpdateGroupStartID0[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedPresynapticUpdateGroup0 *group = &d_mergedPresynapticUpdateGroup0[lo - 1]; 
        const unsigned int groupStartID = d_mergedPresynapticUpdateGroupStartID0[lo - 1];
        const unsigned int lid = id - groupStartID;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                    shRowLength[threadIdx.x] = group->rowLength[spk];
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        const unsigned int npost = shRowLength[j];
                        if (lid < npost) {
                            const unsigned int ipost = group->ind[synAddress];
                            atomicAdd(&group->inSyn[ipost], group->g);
                        }
                    }
                }
            }
        }
        
    }
    // merged1
    if(id >= 4832 && id < 9632) {
        unsigned int lo = 0;
        unsigned int hi = 2;
        while(lo < hi)
         {
            const unsigned int mid = (lo + hi) / 2;
            if(id < d_mergedPresynapticUpdateGroupStartID1[mid]) {
                hi = mid;
            }
            else {
                lo = mid + 1;
            }
        }
        struct MergedPresynapticUpdateGroup1 *group = &d_mergedPresynapticUpdateGroup1[lo - 1]; 
        const unsigned int groupStartID = d_mergedPresynapticUpdateGroupStartID1[lo - 1];
        const unsigned int lid = id - groupStartID;
        double linSyn = 0;
         {
            const unsigned int numSpikes = group->srcSpkCnt[0];
            const unsigned int numSpikeBlocks = (numSpikes + 32 - 1) / 32;
            for (unsigned int r = 0; r < numSpikeBlocks; r++) {
                const unsigned int numSpikesInBlock = (r == numSpikeBlocks - 1) ? ((numSpikes - 1) % 32) + 1 : 32;
                __syncthreads();
                if (threadIdx.x < numSpikesInBlock) {
                    const unsigned int spk = group->srcSpk[(r * 32) + threadIdx.x];
                    shSpk[threadIdx.x] = spk;
                }
                __syncthreads();
                // loop through all incoming spikes
                for (unsigned int j = 0; j < numSpikesInBlock; j++) {
                    // only work on existing neurons
                    if (lid < group->rowStride) {
                        const unsigned int synAddress = (shSpk[j] * group->rowStride) + lid;
                        linSyn += group->g[synAddress];
                    }
                }
            }
        }
        
        // only do this for existing neurons
        if (lid < group->numTrgNeurons) {
            group->inSyn[lid] += linSyn;
        }
    }
}
extern "C" __global__ void updateSynapseDynamicsKernel(double t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    // merged0
    if(id < 9600) {
        struct MergedSynapseDynamicsGroup0 *group = &d_mergedSynapseDynamicsGroup0[0]; 
        const unsigned int lid = id - 0;
        if (lid < (group->numSrcNeurons * group->rowStride)) {
            const unsigned int row = lid / group->rowStride;
            const unsigned int col = lid % group->rowStride;
            if(col < group->rowLength[row]) {
                atomicAdd(&group->inSyn[group->ind[lid]], group->raPre[row]);}
        }
    }
}
void updateSynapses(double t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(300, 1);
        updateSynapseDynamicsKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(301, 1);
        updatePresynapticKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
