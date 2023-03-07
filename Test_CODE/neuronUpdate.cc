#include "definitionsInternal.h"
#include "supportCode.h"


extern "C" __global__ void neuronSpikeQueueUpdateKernel() {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x;
}

extern "C" __global__ void updateNeuronsKernel(double t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __syncthreads();
}
void updateNeurons(double t) {
}
