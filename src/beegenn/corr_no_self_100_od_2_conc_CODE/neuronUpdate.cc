#include "definitionsInternal.h"
#include "supportCode.h"

struct MergedNeuronUpdateGroup0
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    scalar* V;
    scalar* a;
    double* inSynInSyn0;
    double* inSynInSyn1;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    scalar* V;
    scalar* a;
    double* inSynInSyn0;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    scalar* kp1cn_0;
    scalar* km2_2;
    scalar* kp2_2;
    scalar* km1_2;
    scalar* kp1cn_2;
    scalar* km2_1;
    scalar* kp2_1;
    scalar* km1_1;
    scalar* kp1cn_1;
    scalar* km2_0;
    scalar* kp2_0;
    scalar* km1_0;
    scalar* ra;
    scalar* ra_2;
    scalar* rb_2;
    scalar* ra_1;
    scalar* rb_1;
    scalar* ra_0;
    scalar* rb_0;
    scalar* r0;
    unsigned int* spk;
    unsigned int* spkCnt;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup3
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    scalar* V;
    scalar* a;
    double* inSynInSyn0;
    double* inSynInSyn1;
    double* inSynInSyn2;
    uint32_t* recordSpk;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup0
 {
    unsigned int* spkCnt;
    
}
;
struct MergedNeuronSpikeQueueUpdateGroup1
 {
    unsigned int* spkCnt;
    
}
;
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup0 d_mergedNeuronSpikeQueueUpdateGroup0[1];
void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup0 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup0, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup0), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup0)));
}
__device__ __constant__ MergedNeuronSpikeQueueUpdateGroup1 d_mergedNeuronSpikeQueueUpdateGroup1[3];
void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt) {
    MergedNeuronSpikeQueueUpdateGroup1 group = {spkCnt, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronSpikeQueueUpdateGroup1, &group, sizeof(MergedNeuronSpikeQueueUpdateGroup1), idx * sizeof(MergedNeuronSpikeQueueUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup0 d_mergedNeuronUpdateGroup0[1];
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, double* inSynInSyn1, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {spkCnt, spk, rng, V, a, inSynInSyn0, inSynInSyn1, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {spkCnt, spk, rng, V, a, inSynInSyn0, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup2 d_mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, scalar* kp1cn_0, scalar* km2_2, scalar* kp2_2, scalar* km1_2, scalar* kp1cn_2, scalar* km2_1, scalar* kp2_1, scalar* km1_1, scalar* kp1cn_1, scalar* km2_0, scalar* kp2_0, scalar* km1_0, scalar* ra, scalar* ra_2, scalar* rb_2, scalar* ra_1, scalar* rb_1, scalar* ra_0, scalar* rb_0, scalar* r0, unsigned int* spk, unsigned int* spkCnt, unsigned int numNeurons) {
    MergedNeuronUpdateGroup2 group = {kp1cn_0, km2_2, kp2_2, km1_2, kp1cn_2, km2_1, kp2_1, km1_1, kp1cn_1, km2_0, kp2_0, km1_0, ra, ra_2, rb_2, ra_1, rb_1, ra_0, rb_0, r0, spk, spkCnt, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &group, sizeof(MergedNeuronUpdateGroup2), idx * sizeof(MergedNeuronUpdateGroup2)));
}
__device__ __constant__ MergedNeuronUpdateGroup3 d_mergedNeuronUpdateGroup3[1];
void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, double* inSynInSyn1, double* inSynInSyn2, uint32_t* recordSpk, unsigned int numNeurons) {
    MergedNeuronUpdateGroup3 group = {spkCnt, spk, rng, V, a, inSynInSyn0, inSynInSyn1, inSynInSyn2, recordSpk, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup3, &group, sizeof(MergedNeuronUpdateGroup3), idx * sizeof(MergedNeuronUpdateGroup3)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
void pushMergedNeuronUpdate0recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup0) * (idx)) + offsetof(MergedNeuronUpdateGroup0, recordSpk)));
}

void pushMergedNeuronUpdate1recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup1) * (idx)) + offsetof(MergedNeuronUpdateGroup1, recordSpk)));
}

void pushMergedNeuronUpdate3recordSpkToDevice(unsigned int idx, uint32_t* value) {
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup3, &value, sizeof(value), (sizeof(MergedNeuronUpdateGroup3) * (idx)) + offsetof(MergedNeuronUpdateGroup3, recordSpk)));
}

__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {800, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID2[] = {10400, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID3[] = {10560, };

extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
    if(id < 1) {
        struct MergedNeuronSpikeQueueUpdateGroup0 *group = &d_mergedNeuronSpikeQueueUpdateGroup0[id - 0]; 
        group->spkCnt[0] = 0;
    }
    if(id >= 1 && id < 4) {
        struct MergedNeuronSpikeQueueUpdateGroup1 *group = &d_mergedNeuronSpikeQueueUpdateGroup1[id - 1]; 
        group->spkCnt[0] = 0;
    }
}

extern "C" __global__ void updateNeuronsKernel(double t, unsigned int recordingTimestep)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0) {
        shSpkCount = 0;
    }
    
    __shared__ uint32_t shSpkRecord;
    if (threadIdx.x == 0) {
        shSpkRecord = 0;
    }
    __syncthreads();
    // merged0
    if(id < 800) {
        struct MergedNeuronUpdateGroup0 *group = &d_mergedNeuronUpdateGroup0[0]; 
        const unsigned int lid = id - 0;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar la = group->a[lid];
            
            double Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn * ((0.00000000000000000e+00) - lV);
                linSyn*=(9.80198673306755253e-01);
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn * ((-8.00000000000000000e+01) - lV);
                linSyn*=(9.90049833749168107e-01);
                group->inSynInSyn1[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike = (lV >= (-4.00000000000000000e+01));
            // calculate membrane potential
            lV+= (-(1.00000000000000002e-02)*(lV-(-6.00000000000000000e+01)) - (0.00000000000000000e+00)*la*(lV-(-7.00000000000000000e+01)) + (1.00000000000000000e+00)*Isyn+(3.13049516849970555e+00)*curand_normal_double(&group->rng[lid]))*DT/(1.00000000000000000e+00);
            la+= -la*DT/(1.00000000000000000e+03);
            // test for and register a true spike
            if ((lV >= (-4.00000000000000000e+01)) && !(oldSpike)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
                // spike reset code
                lV= (-7.00000000000000000e+01);
                la+= 0.5;
            }
            group->V[lid] = lV;
            group->a[lid] = la;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
    // merged1
    if(id >= 800 && id < 10400) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 800;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar la = group->a[lid];
            
            double Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn0[lid];
                Isyn+= linSyn;
                linSyn= 0.0;
                
                group->inSynInSyn0[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike = (lV >= (-4.00000000000000000e+01));
            // calculate membrane potential
            lV+= (-(1.00000000000000002e-02)*(lV-(-6.00000000000000000e+01)) - (1.50000000000000003e-03)*la*(lV-(-7.00000000000000000e+01)) + (1.00000000000000000e+01)*Isyn+(3.13049516849970555e+00)*curand_normal_double(&group->rng[lid]))*DT/(1.00000000000000000e+00);
            la+= -la*DT/(1.00000000000000000e+03);
            // test for and register a true spike
            if ((lV >= (-4.00000000000000000e+01)) && !(oldSpike)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
                // spike reset code
                lV= (-7.00000000000000000e+01);
                la+= 0.5;
            }
            group->V[lid] = lV;
            group->a[lid] = la;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
    // merged2
    if(id >= 10400 && id < 10560) {
        struct MergedNeuronUpdateGroup2 *group = &d_mergedNeuronUpdateGroup2[0]; 
        const unsigned int lid = id - 10400;
        
        if(lid < group->numNeurons) {
            scalar lr0 = group->r0[lid];
            scalar lrb_0 = group->rb_0[lid];
            scalar lra_0 = group->ra_0[lid];
            scalar lrb_1 = group->rb_1[lid];
            scalar lra_1 = group->ra_1[lid];
            scalar lrb_2 = group->rb_2[lid];
            scalar lra_2 = group->ra_2[lid];
            scalar lra = group->ra[lid];
            scalar lkp1cn_0 = group->kp1cn_0[lid];
            scalar lkm1_0 = group->km1_0[lid];
            scalar lkp2_0 = group->kp2_0[lid];
            scalar lkm2_0 = group->km2_0[lid];
            scalar lkp1cn_1 = group->kp1cn_1[lid];
            scalar lkm1_1 = group->km1_1[lid];
            scalar lkp2_1 = group->kp2_1[lid];
            scalar lkm2_1 = group->km2_1[lid];
            scalar lkp1cn_2 = group->kp1cn_2[lid];
            scalar lkm1_2 = group->km1_2[lid];
            scalar lkp2_2 = group->kp2_2[lid];
            scalar lkm2_2 = group->km2_2[lid];
            
            // calculate membrane potential
            // update all bound receptors and activated receptors
            lrb_0+= (lkp1cn_0*lr0 - lkm1_0*lrb_0 + lkm2_0*lra_0 - lkp2_0*lrb_0)*DT;
            if (lrb_0 > 1.0) lrb_0= 1.0;
            lra_0+= (lkp2_0*lrb_0 - lkm2_0*lra_0)*DT;
            if (lra_0 > 1.0) lra_0= 1.0;
            lrb_1+= (lkp1cn_1*lr0 - lkm1_1*lrb_1 + lkm2_1*lra_1 - lkp2_1*lrb_1)*DT;
            if (lrb_1 > 1.0) lrb_1= 1.0;
            lra_1+= (lkp2_1*lrb_1 - lkm2_1*lra_1)*DT;
            if (lra_1 > 1.0) lra_1= 1.0;
            lrb_2+= (lkp1cn_2*lr0 - lkm1_2*lrb_2 + lkm2_2*lra_2 - lkp2_2*lrb_2)*DT;
            if (lrb_2 > 1.0) lrb_2= 1.0;
            lra_2+= (lkp2_2*lrb_2 - lkm2_2*lra_2)*DT;
            if (lra_2 > 1.0) lra_2= 1.0;
            // now update ra and calculate the sum of bound receptors
            scalar rb= lrb_0 + lrb_1 + lrb_2;
            if (rb > 1.0) rb= 1.0;
            lra= lra_0 + lra_1 + lra_2;
            if (lra > 1.0) lra= 1.0;
            // then update r0 as a function of rb and ra
            lr0= 1.0 - rb - lra;
            if (lr0 < 0.0) lr0= 0.0;
            group->r0[lid] = lr0;
            group->rb_0[lid] = lrb_0;
            group->ra_0[lid] = lra_0;
            group->rb_1[lid] = lrb_1;
            group->ra_1[lid] = lra_1;
            group->rb_2[lid] = lrb_2;
            group->ra_2[lid] = lra_2;
            group->ra[lid] = lra;
            group->kp1cn_0[lid] = lkp1cn_0;
            group->km1_0[lid] = lkm1_0;
            group->kp2_0[lid] = lkp2_0;
            group->km2_0[lid] = lkm2_0;
            group->kp1cn_1[lid] = lkp1cn_1;
            group->km1_1[lid] = lkm1_1;
            group->kp2_1[lid] = lkp2_1;
            group->km2_1[lid] = lkm2_1;
            group->kp1cn_2[lid] = lkp1cn_2;
            group->km1_2[lid] = lkm1_2;
            group->kp2_2[lid] = lkp2_2;
            group->km2_2[lid] = lkm2_2;
        }
        __syncthreads();
    }
    // merged3
    if(id >= 10560 && id < 14560) {
        struct MergedNeuronUpdateGroup3 *group = &d_mergedNeuronUpdateGroup3[0]; 
        const unsigned int lid = id - 10560;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar la = group->a[lid];
            
            double Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn * ((0.00000000000000000e+00) - lV);
                linSyn*=(9.80198673306755253e-01);
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn * ((0.00000000000000000e+00) - lV);
                linSyn*=(9.80198673306755253e-01);
                group->inSynInSyn1[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn2[lid];
                Isyn += linSyn * ((-8.00000000000000000e+01) - lV);
                linSyn*=(9.90049833749168107e-01);
                group->inSynInSyn2[lid] = linSyn;
            }
            // test whether spike condition was fulfilled previously
            const bool oldSpike = (lV >= (-4.00000000000000000e+01));
            // calculate membrane potential
            lV+= (-(1.00000000000000002e-02)*(lV-(-6.00000000000000000e+01)) - (5.00000000000000010e-04)*la*(lV-(-7.00000000000000000e+01)) + (1.00000000000000000e+00)*Isyn+(3.13049516849970555e+00)*curand_normal_double(&group->rng[lid]))*DT/(1.00000000000000000e+00);
            la+= -la*DT/(1.00000000000000000e+03);
            // test for and register a true spike
            if ((lV >= (-4.00000000000000000e+01)) && !(oldSpike)) {
                const unsigned int spkIdx = atomicAdd(&shSpkCount, 1);
                shSpk[spkIdx] = lid;
                atomicOr(&shSpkRecord, 1 << threadIdx.x);
                // spike reset code
                lV= (-7.00000000000000000e+01);
                la+= 0.5;
            }
            group->V[lid] = lV;
            group->a[lid] = la;
        }
        __syncthreads();
        if(threadIdx.x == 0) {
            if (shSpkCount > 0) {
                shPosSpk = atomicAdd(&group->spkCnt[0], shSpkCount);
            }
        }
        __syncthreads();
        if(threadIdx.x < shSpkCount) {
            const unsigned int n = shSpk[threadIdx.x];
            group->spk[shPosSpk + threadIdx.x] = n;
        }
        if(threadIdx.x < 1) {
            const unsigned int numRecordingWords = (group->numNeurons + 31) / 32;
            const unsigned int popWordIdx = (lid / 32) + threadIdx.x;
            if(popWordIdx < numRecordingWords) {
                group->recordSpk[(recordingTimestep * numRecordingWords * 1) + popWordIdx] = shSpkRecord;
            }
        }
    }
}
void updateNeurons(double t, unsigned int recordingTimestep) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(455, 1);
        updateNeuronsKernel<<<grid, threads>>>(t, recordingTimestep);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
