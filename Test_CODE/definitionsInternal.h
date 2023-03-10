#pragma once
#include "definitions.h"

// CUDA includes
#include <curand_kernel.h>
#include <cuda_fp16.h>

// ------------------------------------------------------------------------
// Helper macro for error-checking CUDA calls
#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": cuda error " + std::to_string(error) + ": " + cudaGetErrorString(error));\
    }\
}

#define SUPPORT_CODE_FUNC __device__ __host__ inline


template<typename RNG>
__device__ inline float exponentialDistFloat(RNG *rng) {
    while (true) {
        const float u = curand_uniform(rng);
        if (u != 0.0f) {
            return -logf(u);
        }
    }
}

template<typename RNG>
__device__ inline double exponentialDistDouble(RNG *rng) {
    while (true) {
        const double u = curand_uniform_double(rng);
        if (u != 0.0) {
            return -log(u);
        }
    }
}

template<typename RNG>
__device__ inline float gammaDistFloatInternal(RNG *rng, float c, float d)
 {
    float x, v, u;
    while (true) {
        do {
            x = curand_normal(rng);
            v = 1.0f + c*x;
        }
        while (v <= 0.0f);
        
        v = v*v*v;
        do {
            u = curand_uniform(rng);
        }
        while (u == 1.0f);
        
        if (u < 1.0f - 0.0331f*x*x*x*x) break;
        if (logf(u) < 0.5f*x*x + d*(1.0f - v + logf(v))) break;
    }
    
    return d*v;
}

template<typename RNG>
__device__ inline float gammaDistFloat(RNG *rng, float a)
 {
    if (a > 1)
     {
        const float u = curand_uniform (rng);
        const float d = (1.0f + a) - 1.0f / 3.0f;
        const float c = (1.0f / 3.0f) / sqrtf(d);
        return gammaDistFloatInternal (rng, c, d) * powf(u, 1.0f / a);
    }
    else
     {
        const float d = a - 1.0f / 3.0f;
        const float c = (1.0f / 3.0f) / sqrtf(d);
        return gammaDistFloatInternal(rng, c, d);
    }
}

template<typename RNG>
__device__ inline float gammaDistDoubleInternal(RNG *rng, double c, double d)
 {
    double x, v, u;
    while (true) {
        do {
            x = curand_normal_double(rng);
            v = 1.0 + c*x;
        }
        while (v <= 0.0);
        
        v = v*v*v;
        do {
            u = curand_uniform_double(rng);
        }
        while (u == 1.0);
        
        if (u < 1.0 - 0.0331*x*x*x*x) break;
        if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))) break;
    }
    
    return d*v;
}

template<typename RNG>
__device__ inline float gammaDistDouble(RNG *rng, double a)
 {
    if (a > 1.0)
     {
        const double u = curand_uniform (rng);
        const double d = (1.0 + a) - 1.0 / 3.0;
        const double c = (1.0 / 3.0) / sqrt(d);
        return gammaDistDoubleInternal (rng, c, d) * pow(u, 1.0 / a);
    }
    else
     {
        const float d = a - 1.0 / 3.0;
        const float c = (1.0 / 3.0) / sqrt(d);
        return gammaDistDoubleInternal(rng, c, d);
    }
}

template<typename RNG>
__device__ inline unsigned int binomialDistFloatInternal(RNG *rng, unsigned int n, float p)
 {
    const float q = 1.0f - p;
    const float qn = expf(n * logf(q));
    const float np = n * p;
    const unsigned int bound = min(n, (unsigned int)(np + (10.0f * sqrtf((np * q) + 1.0f))));
    unsigned int x = 0;
    float px = qn;
    float u = curand_uniform(rng);
    while(u > px)
     {
        x++;
        if(x > bound) {
            x = 0;
            px = qn;
            u = curand_uniform(rng);
        }
        else {
            u -= px;
            px = ((n - x + 1) * p * px) / (x * q);
        }
    }
    return x;
}

template<typename RNG>
__device__ inline unsigned int binomialDistFloat(RNG *rng, unsigned int n, float p)
 {
    if(p <= 0.5f) {
        return binomialDistFloatInternal(rng, n, p);
    }
    else {
        return (n - binomialDistFloatInternal(rng, n, 1.0f - p));
    }
}
template<typename RNG>
__device__ inline unsigned int binomialDistDoubleInternal(RNG *rng, unsigned int n, double p)
 {
    const double q = 1.0 - p;
    const double qn = exp(n * log(q));
    const double np = n * p;
    const unsigned int bound = min(n, (unsigned int)(np + (10.0 * sqrt((np * q) + 1.0))));
    unsigned int x = 0;
    double px = qn;
    double u = curand_uniform_double(rng);
    while(u > px)
     {
        x++;
        if(x > bound) {
            x = 0;
            px = qn;
            u = curand_uniform_double(rng);
        }
        else {
            u -= px;
            px = ((n - x + 1) * p * px) / (x * q);
        }
    }
    return x;
}

template<typename RNG>
__device__ inline unsigned int binomialDistDouble(RNG *rng, unsigned int n, double p)
 {
    if(p <= 0.5) {
        return binomialDistDoubleInternal(rng, n, p);
    }
    else {
        return (n - binomialDistDoubleInternal(rng, n, 1.0 - p));
    }
}
// ------------------------------------------------------------------------
// merged group structures
// ------------------------------------------------------------------------
extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR __device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged group arrays for host initialisation
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
EXPORT_VAR curandState* d_rngln;
EXPORT_VAR curandState* d_rngorn;
EXPORT_VAR curandState* d_rngpn;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying merged group structures to device
// ------------------------------------------------------------------------
EXPORT_FUNC void pushMergedNeuronInitGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup1ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronInitGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, double* inSynInSyn0, double* inSynInSyn1, double* inSynInSyn2, unsigned int numNeurons);
EXPORT_FUNC void pushMergedSynapseDenseInitGroup0ToDevice(unsigned int idx, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedSynapseDenseInitGroup1ToDevice(unsigned int idx, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedSynapseConnectivityInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedSynapseConnectivityInitGroup1ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedSynapseConnectivityInitGroup2ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, scalar n_orn, scalar n_trg, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedSynapseSparseInitGroup0ToDevice(unsigned int idx, unsigned int* rowLength, uint32_t* ind, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons, unsigned int colStride);
EXPORT_FUNC void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, scalar* kp1cn_0_65, scalar* kp2_0_58, scalar* kp1cn_0_59, scalar* kp2_0_59, scalar* kp1cn_0_60, scalar* kp2_0_60, scalar* kp1cn_0_61, scalar* kp2_0_61, scalar* kp1cn_0_62, scalar* kp2_0_62, scalar* kp1cn_0_63, scalar* kp2_0_63, scalar* kp1cn_0_64, scalar* kp2_0_64, scalar* kp1cn_0_58, scalar* kp2_0_65, scalar* kp1cn_0_66, scalar* kp2_0_66, scalar* kp1cn_0_67, scalar* kp2_0_67, scalar* kp1cn_0_68, scalar* kp2_0_68, scalar* kp1cn_0_69, scalar* kp2_0_69, scalar* kp1cn_0_70, scalar* kp2_0_70, scalar* kp1cn_0_71, scalar* kp2_0_71, scalar* kp1cn_0_51, scalar* kp2_0_44, scalar* kp1cn_0_45, scalar* kp2_0_45, scalar* kp1cn_0_46, scalar* kp2_0_46, scalar* kp1cn_0_47, scalar* kp2_0_47, scalar* kp1cn_0_48, scalar* kp2_0_48, scalar* kp1cn_0_49, scalar* kp2_0_49, scalar* kp1cn_0_50, scalar* kp2_0_50, scalar* kp1cn_0_72, scalar* kp2_0_51, scalar* kp1cn_0_52, scalar* kp2_0_52, scalar* kp1cn_0_53, scalar* kp2_0_53, scalar* kp1cn_0_54, scalar* kp2_0_54, scalar* kp1cn_0_55, scalar* kp2_0_55, scalar* kp1cn_0_56, scalar* kp2_0_56, scalar* kp1cn_0_57, scalar* kp2_0_57, scalar* kp1cn_0_93, scalar* kp2_0_86, scalar* kp1cn_0_87, scalar* kp2_0_87, scalar* kp1cn_0_88, scalar* kp2_0_88, scalar* kp1cn_0_89, scalar* kp2_0_89, scalar* kp1cn_0_90, scalar* kp2_0_90, scalar* kp1cn_0_91, scalar* kp2_0_91, scalar* kp1cn_0_92, scalar* kp2_0_92, scalar* kp1cn_0_86, scalar* kp2_0_93, scalar* kp1cn_0_94, scalar* kp2_0_94, scalar* kp1cn_0_95, scalar* kp2_0_95, scalar* kp1cn_0_96, scalar* kp2_0_96, scalar* kp1cn_0_97, scalar* kp2_0_97, scalar* kp1cn_0_98, scalar* kp2_0_98, scalar* kp1cn_0_99, scalar* kp2_0_99, scalar* kp1cn_0_79, scalar* kp2_0_72, scalar* kp1cn_0_73, scalar* kp2_0_73, scalar* kp1cn_0_74, scalar* kp2_0_74, scalar* kp1cn_0_75, scalar* kp2_0_75, scalar* kp1cn_0_76, scalar* kp2_0_76, scalar* kp1cn_0_77, scalar* kp2_0_77, scalar* kp1cn_0_78, scalar* kp2_0_78, scalar* kp1cn_0_44, scalar* kp2_0_79, scalar* kp1cn_0_80, scalar* kp2_0_80, scalar* kp1cn_0_81, scalar* kp2_0_81, scalar* kp1cn_0_82, scalar* kp2_0_82, scalar* kp1cn_0_83, scalar* kp2_0_83, scalar* kp1cn_0_84, scalar* kp2_0_84, scalar* kp1cn_0_85, scalar* kp2_0_85, scalar* kp2_0_9, scalar* kp1cn_0_3, scalar* kp2_0_3, scalar* kp1cn_0_4, scalar* kp2_0_4, scalar* kp1cn_0_5, scalar* kp2_0_5, scalar* kp1cn_0_6, scalar* kp2_0_6, scalar* kp1cn_0_7, scalar* kp2_0_7, scalar* kp1cn_0_8, scalar* kp2_0_8, scalar* kp1cn_0_9, scalar* kp2_0_2, scalar* kp1cn_0_10, scalar* kp2_0_10, scalar* kp1cn_0_11, scalar* kp2_0_11, scalar* kp1cn_0_12, scalar* kp2_0_12, scalar* kp1cn_0_13, scalar* kp2_0_13, scalar* kp1cn_0_14, scalar* kp2_0_14, scalar* kp1cn_0_15, scalar* kp2_0_15, scalar* kp1cn_0_16, scalar* km2_0, unsigned int* spkCnt, unsigned int* spk, scalar* r0, scalar* rb_0, scalar* ra_0, scalar* rb_1, scalar* ra_1, scalar* rb_2, scalar* ra_2, scalar* ra, scalar* kp1cn_0, scalar* km1_0, scalar* kp2_0, scalar* kp2_0_16, scalar* kp1cn_1, scalar* km1_1, scalar* kp2_1, scalar* km2_1, scalar* kp1cn_2, scalar* km1_2, scalar* kp2_2, scalar* km2_2, scalar* kp1cn_0_0, scalar* kp2_0_0, scalar* kp1cn_0_1, scalar* kp2_0_1, scalar* kp1cn_0_2, scalar* kp1cn_0_37, scalar* kp2_0_30, scalar* kp1cn_0_31, scalar* kp2_0_31, scalar* kp1cn_0_32, scalar* kp2_0_32, scalar* kp1cn_0_33, scalar* kp2_0_33, scalar* kp1cn_0_34, scalar* kp2_0_34, scalar* kp1cn_0_35, scalar* kp2_0_35, scalar* kp1cn_0_36, scalar* kp2_0_36, scalar* kp1cn_0_30, scalar* kp2_0_37, scalar* kp1cn_0_38, scalar* kp2_0_38, scalar* kp1cn_0_39, scalar* kp2_0_39, scalar* kp1cn_0_40, scalar* kp2_0_40, scalar* kp1cn_0_41, scalar* kp2_0_41, scalar* kp1cn_0_42, scalar* kp2_0_42, scalar* kp1cn_0_43, scalar* kp2_0_43, scalar* kp2_0_23, scalar* kp1cn_0_17, scalar* kp2_0_17, scalar* kp1cn_0_18, scalar* kp2_0_18, scalar* kp1cn_0_19, scalar* kp2_0_19, scalar* kp1cn_0_20, scalar* kp2_0_20, scalar* kp1cn_0_21, scalar* kp2_0_21, scalar* kp1cn_0_22, scalar* kp2_0_22, scalar* kp1cn_0_23, scalar* kp1cn_0_24, scalar* kp2_0_24, scalar* kp1cn_0_25, scalar* kp2_0_25, scalar* kp1cn_0_26, scalar* kp2_0_26, scalar* kp1cn_0_27, scalar* kp2_0_27, scalar* kp1cn_0_28, scalar* kp2_0_28, scalar* kp1cn_0_29, scalar* kp2_0_29, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, unsigned int numNeurons);
EXPORT_FUNC void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, double* inSynInSyn1, double* inSynInSyn2, unsigned int numNeurons);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup0ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup1ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, unsigned int* rowLength, uint32_t* ind, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedPresynapticUpdateGroup2ToDevice(unsigned int idx, double* inSyn, unsigned int* srcSpkCnt, unsigned int* srcSpk, scalar* g, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedSynapseDynamicsGroup0ToDevice(unsigned int idx, double* inSyn, scalar* raPre, unsigned int* rowLength, uint32_t* ind, unsigned int rowStride, unsigned int numSrcNeurons, unsigned int numTrgNeurons);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt);
EXPORT_FUNC void pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(unsigned int idx, unsigned int* spkCnt);
}  // extern "C"
