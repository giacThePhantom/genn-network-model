#include "definitionsInternal.h"

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
unsigned long long iT;
double t;
unsigned long long numRecordingTimesteps = 0;
__device__ curandStatePhilox4_32_10_t d_rng;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
double initTime = 0.0;
double initSparseTime = 0.0;
double neuronUpdateTime = 0.0;
double presynapticUpdateTime = 0.0;
double postsynapticUpdateTime = 0.0;
double synapseDynamicsTime = 0.0;
// ------------------------------------------------------------------------
// merged group arrays
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
unsigned int* glbSpkCntln;
unsigned int* d_glbSpkCntln;
unsigned int* glbSpkln;
unsigned int* d_glbSpkln;
uint32_t* recordSpkln;
uint32_t* d_recordSpkln;
curandState* d_rngln;
scalar* Vln;
scalar* d_Vln;
scalar* aln;
scalar* d_aln;
unsigned int* glbSpkCntor;
unsigned int* d_glbSpkCntor;
unsigned int* glbSpkor;
unsigned int* d_glbSpkor;
scalar* r0or;
scalar* d_r0or;
scalar* rb_0or;
scalar* d_rb_0or;
scalar* ra_0or;
scalar* d_ra_0or;
scalar* rb_1or;
scalar* d_rb_1or;
scalar* ra_1or;
scalar* d_ra_1or;
scalar* rb_2or;
scalar* d_rb_2or;
scalar* ra_2or;
scalar* d_ra_2or;
scalar* raor;
scalar* d_raor;
scalar* kp1cn_0or;
scalar* d_kp1cn_0or;
scalar* km1_0or;
scalar* d_km1_0or;
scalar* kp2_0or;
scalar* d_kp2_0or;
scalar* km2_0or;
scalar* d_km2_0or;
scalar* kp1cn_1or;
scalar* d_kp1cn_1or;
scalar* km1_1or;
scalar* d_km1_1or;
scalar* kp2_1or;
scalar* d_kp2_1or;
scalar* km2_1or;
scalar* d_km2_1or;
scalar* kp1cn_2or;
scalar* d_kp1cn_2or;
scalar* km1_2or;
scalar* d_km1_2or;
scalar* kp2_2or;
scalar* d_kp2_2or;
scalar* km2_2or;
scalar* d_km2_2or;
unsigned int* glbSpkCntorn;
unsigned int* d_glbSpkCntorn;
unsigned int* glbSpkorn;
unsigned int* d_glbSpkorn;
uint32_t* recordSpkorn;
uint32_t* d_recordSpkorn;
curandState* d_rngorn;
scalar* Vorn;
scalar* d_Vorn;
scalar* aorn;
scalar* d_aorn;
unsigned int* glbSpkCntpn;
unsigned int* d_glbSpkCntpn;
unsigned int* glbSpkpn;
unsigned int* d_glbSpkpn;
uint32_t* recordSpkpn;
uint32_t* d_recordSpkpn;
curandState* d_rngpn;
scalar* Vpn;
scalar* d_Vpn;
scalar* apn;
scalar* d_apn;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
double* inSynpn_ln;
double* d_inSynpn_ln;
double* inSynorn_ln;
double* d_inSynorn_ln;
double* inSynln_ln;
double* d_inSynln_ln;
double* inSynor_orn;
double* d_inSynor_orn;
double* inSynln_pn;
double* d_inSynln_pn;
double* inSynorn_pn;
double* d_inSynorn_pn;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
const unsigned int maxRowLengthor_orn = 60;
unsigned int* rowLengthor_orn;
unsigned int* d_rowLengthor_orn;
uint32_t* indor_orn;
uint32_t* d_indor_orn;
const unsigned int maxRowLengthorn_ln = 4000;
unsigned int* rowLengthorn_ln;
unsigned int* d_rowLengthorn_ln;
uint32_t* indorn_ln;
uint32_t* d_indorn_ln;
const unsigned int maxRowLengthorn_pn = 800;
unsigned int* rowLengthorn_pn;
unsigned int* d_rowLengthorn_pn;
uint32_t* indorn_pn;
uint32_t* d_indorn_pn;
const unsigned int maxRowLengthpn_ln = 25;
unsigned int* rowLengthpn_ln;
unsigned int* d_rowLengthpn_ln;
uint32_t* indpn_ln;
uint32_t* d_indpn_ln;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
scalar* gln_ln;
scalar* d_gln_ln;
scalar* gln_pn;
scalar* d_gln_pn;

}  // extern "C"
// ------------------------------------------------------------------------
// extra global params
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// copying things to device
// ------------------------------------------------------------------------
void pushlnSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntln, glbSpkCntln, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkln, glbSpkln, 4000 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushlnCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntln, glbSpkCntln, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkln, glbSpkln, glbSpkCntln[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVlnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vln, Vln, 4000 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentVlnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vln, Vln, 4000 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushalnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aln, aln, 4000 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentalnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aln, aln, 4000 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushlnStateToDevice(bool uninitialisedOnly) {
    pushVlnToDevice(uninitialisedOnly);
    pushalnToDevice(uninitialisedOnly);
}

void pushorSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntor, glbSpkCntor, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkor, glbSpkor, 160 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushorCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntor, glbSpkCntor, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkor, glbSpkor, glbSpkCntor[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushr0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r0or, r0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentr0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_r0or, r0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushrb_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rb_0or, rb_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentrb_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rb_0or, rb_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushra_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ra_0or, ra_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentra_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ra_0or, ra_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushrb_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rb_1or, rb_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentrb_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rb_1or, rb_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushra_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ra_1or, ra_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentra_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ra_1or, ra_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushrb_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rb_2or, rb_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentrb_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_rb_2or, rb_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushra_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ra_2or, ra_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentra_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_ra_2or, ra_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushraorToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_raor, raor, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentraorToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_raor, raor, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0or, kp1cn_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0or, kp1cn_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkm1_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km1_0or, km1_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkm1_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km1_0or, km1_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0or, kp2_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0or, kp2_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkm2_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km2_0or, km2_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkm2_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km2_0or, km2_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_1or, kp1cn_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_1or, kp1cn_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkm1_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km1_1or, km1_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkm1_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km1_1or, km1_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_1or, kp2_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_1or, kp2_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkm2_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km2_1or, km2_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkm2_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km2_1or, km2_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_2or, kp1cn_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_2or, kp1cn_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkm1_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km1_2or, km1_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkm1_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km1_2or, km1_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_2or, kp2_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_2or, kp2_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkm2_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km2_2or, km2_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkm2_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_km2_2or, km2_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushorStateToDevice(bool uninitialisedOnly) {
    pushr0orToDevice(uninitialisedOnly);
    pushrb_0orToDevice(uninitialisedOnly);
    pushra_0orToDevice(uninitialisedOnly);
    pushrb_1orToDevice(uninitialisedOnly);
    pushra_1orToDevice(uninitialisedOnly);
    pushrb_2orToDevice(uninitialisedOnly);
    pushra_2orToDevice(uninitialisedOnly);
    pushraorToDevice(uninitialisedOnly);
    pushkp1cn_0orToDevice(uninitialisedOnly);
    pushkm1_0orToDevice(uninitialisedOnly);
    pushkp2_0orToDevice(uninitialisedOnly);
    pushkm2_0orToDevice(uninitialisedOnly);
    pushkp1cn_1orToDevice(uninitialisedOnly);
    pushkm1_1orToDevice(uninitialisedOnly);
    pushkp2_1orToDevice(uninitialisedOnly);
    pushkm2_1orToDevice(uninitialisedOnly);
    pushkp1cn_2orToDevice(uninitialisedOnly);
    pushkm1_2orToDevice(uninitialisedOnly);
    pushkp2_2orToDevice(uninitialisedOnly);
    pushkm2_2orToDevice(uninitialisedOnly);
}

void pushornSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntorn, glbSpkCntorn, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkorn, glbSpkorn, 9600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushornCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntorn, glbSpkCntorn, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkorn, glbSpkorn, glbSpkCntorn[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVornToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vorn, Vorn, 9600 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentVornToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vorn, Vorn, 9600 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushaornToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aorn, aorn, 9600 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentaornToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_aorn, aorn, 9600 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushornStateToDevice(bool uninitialisedOnly) {
    pushVornToDevice(uninitialisedOnly);
    pushaornToDevice(uninitialisedOnly);
}

void pushpnSpikesToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntpn, glbSpkCntpn, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkpn, glbSpkpn, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
}

void pushpnCurrentSpikesToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntpn, glbSpkCntpn, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkpn, glbSpkpn, glbSpkCntpn[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

void pushVpnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vpn, Vpn, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentVpnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_Vpn, Vpn, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushapnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_apn, apn, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentapnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_apn, apn, 800 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushpnStateToDevice(bool uninitialisedOnly) {
    pushVpnToDevice(uninitialisedOnly);
    pushapnToDevice(uninitialisedOnly);
}

void pushor_ornConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthor_orn, rowLengthor_orn, 160 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indor_orn, indor_orn, 9600 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushorn_lnConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthorn_ln, rowLengthorn_ln, 9600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indorn_ln, indorn_ln, 38400000 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushorn_pnConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthorn_pn, rowLengthorn_pn, 9600 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indorn_pn, indorn_pn, 7680000 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushpn_lnConnectivityToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_rowLengthpn_ln, rowLengthpn_ln, 800 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indpn_ln, indpn_ln, 20000 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
}

void pushgln_lnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gln_ln, gln_ln, 16000000 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSynln_lnToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynln_ln, inSynln_ln, 4000 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushln_lnStateToDevice(bool uninitialisedOnly) {
    pushgln_lnToDevice(uninitialisedOnly);
    pushinSynln_lnToDevice(uninitialisedOnly);
}

void pushgln_pnToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_gln_pn, gln_pn, 3200000 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushinSynln_pnToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynln_pn, inSynln_pn, 800 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushln_pnStateToDevice(bool uninitialisedOnly) {
    pushgln_pnToDevice(uninitialisedOnly);
    pushinSynln_pnToDevice(uninitialisedOnly);
}

void pushinSynor_ornToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynor_orn, inSynor_orn, 9600 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushor_ornStateToDevice(bool uninitialisedOnly) {
    pushinSynor_ornToDevice(uninitialisedOnly);
}

void pushinSynorn_lnToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynorn_ln, inSynorn_ln, 4000 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushorn_lnStateToDevice(bool uninitialisedOnly) {
    pushinSynorn_lnToDevice(uninitialisedOnly);
}

void pushinSynorn_pnToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynorn_pn, inSynorn_pn, 800 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushorn_pnStateToDevice(bool uninitialisedOnly) {
    pushinSynorn_pnToDevice(uninitialisedOnly);
}

void pushinSynpn_lnToDevice(bool uninitialisedOnly) {
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynpn_ln, inSynpn_ln, 4000 * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void pushpn_lnStateToDevice(bool uninitialisedOnly) {
    pushinSynpn_lnToDevice(uninitialisedOnly);
}


// ------------------------------------------------------------------------
// copying things from device
// ------------------------------------------------------------------------
void pulllnSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntln, d_glbSpkCntln, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkln, d_glbSpkln, 4000 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pulllnCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntln, d_glbSpkCntln, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkln, d_glbSpkln, glbSpkCntln[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVlnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vln, d_Vln, 4000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVlnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vln, d_Vln, 4000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullalnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aln, d_aln, 4000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentalnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aln, d_aln, 4000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pulllnStateFromDevice() {
    pullVlnFromDevice();
    pullalnFromDevice();
}

void pullorSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntor, d_glbSpkCntor, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkor, d_glbSpkor, 160 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullorCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntor, d_glbSpkCntor, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkor, d_glbSpkor, glbSpkCntor[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullr0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r0or, d_r0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentr0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(r0or, d_r0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullrb_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rb_0or, d_rb_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrb_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rb_0or, d_rb_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullra_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ra_0or, d_ra_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentra_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ra_0or, d_ra_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullrb_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rb_1or, d_rb_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrb_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rb_1or, d_rb_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullra_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ra_1or, d_ra_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentra_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ra_1or, d_ra_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullrb_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rb_2or, d_rb_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentrb_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rb_2or, d_rb_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullra_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ra_2or, d_ra_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentra_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(ra_2or, d_ra_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullraorFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(raor, d_raor, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentraorFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(raor, d_raor, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0or, d_kp1cn_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0or, d_kp1cn_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkm1_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km1_0or, d_km1_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkm1_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km1_0or, d_km1_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0or, d_kp2_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0or, d_kp2_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkm2_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km2_0or, d_km2_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkm2_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km2_0or, d_km2_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_1or, d_kp1cn_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_1or, d_kp1cn_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkm1_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km1_1or, d_km1_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkm1_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km1_1or, d_km1_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_1or, d_kp2_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_1or, d_kp2_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkm2_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km2_1or, d_km2_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkm2_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km2_1or, d_km2_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_2or, d_kp1cn_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_2or, d_kp1cn_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkm1_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km1_2or, d_km1_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkm1_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km1_2or, d_km1_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_2or, d_kp2_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_2or, d_kp2_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkm2_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km2_2or, d_km2_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkm2_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(km2_2or, d_km2_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullorStateFromDevice() {
    pullr0orFromDevice();
    pullrb_0orFromDevice();
    pullra_0orFromDevice();
    pullrb_1orFromDevice();
    pullra_1orFromDevice();
    pullrb_2orFromDevice();
    pullra_2orFromDevice();
    pullraorFromDevice();
    pullkp1cn_0orFromDevice();
    pullkm1_0orFromDevice();
    pullkp2_0orFromDevice();
    pullkm2_0orFromDevice();
    pullkp1cn_1orFromDevice();
    pullkm1_1orFromDevice();
    pullkp2_1orFromDevice();
    pullkm2_1orFromDevice();
    pullkp1cn_2orFromDevice();
    pullkm1_2orFromDevice();
    pullkp2_2orFromDevice();
    pullkm2_2orFromDevice();
}

void pullornSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntorn, d_glbSpkCntorn, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkorn, d_glbSpkorn, 9600 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullornCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntorn, d_glbSpkCntorn, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkorn, d_glbSpkorn, glbSpkCntorn[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVornFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vorn, d_Vorn, 9600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVornFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vorn, d_Vorn, 9600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullaornFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aorn, d_aorn, 9600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentaornFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(aorn, d_aorn, 9600 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullornStateFromDevice() {
    pullVornFromDevice();
    pullaornFromDevice();
}

void pullpnSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntpn, d_glbSpkCntpn, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkpn, d_glbSpkpn, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullpnCurrentSpikesFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntpn, d_glbSpkCntpn, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkpn, d_glbSpkpn, glbSpkCntpn[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void pullVpnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vpn, d_Vpn, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentVpnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(Vpn, d_Vpn, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullapnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(apn, d_apn, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentapnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(apn, d_apn, 800 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullpnStateFromDevice() {
    pullVpnFromDevice();
    pullapnFromDevice();
}

void pullor_ornConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthor_orn, d_rowLengthor_orn, 160 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indor_orn, d_indor_orn, 9600 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullorn_lnConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthorn_ln, d_rowLengthorn_ln, 9600 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indorn_ln, d_indorn_ln, 38400000 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullorn_pnConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthorn_pn, d_rowLengthorn_pn, 9600 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indorn_pn, d_indorn_pn, 7680000 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullpn_lnConnectivityFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(rowLengthpn_ln, d_rowLengthpn_ln, 800 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(indpn_ln, d_indpn_ln, 20000 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void pullgln_lnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gln_ln, d_gln_ln, 16000000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynln_lnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynln_ln, d_inSynln_ln, 4000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullln_lnStateFromDevice() {
    pullgln_lnFromDevice();
    pullinSynln_lnFromDevice();
}

void pullgln_pnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(gln_pn, d_gln_pn, 3200000 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullinSynln_pnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynln_pn, d_inSynln_pn, 800 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullln_pnStateFromDevice() {
    pullgln_pnFromDevice();
    pullinSynln_pnFromDevice();
}

void pullinSynor_ornFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynor_orn, d_inSynor_orn, 9600 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullor_ornStateFromDevice() {
    pullinSynor_ornFromDevice();
}

void pullinSynorn_lnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynorn_ln, d_inSynorn_ln, 4000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullorn_lnStateFromDevice() {
    pullinSynorn_lnFromDevice();
}

void pullinSynorn_pnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynorn_pn, d_inSynorn_pn, 800 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullorn_pnStateFromDevice() {
    pullinSynorn_pnFromDevice();
}

void pullinSynpn_lnFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynpn_ln, d_inSynpn_ln, 4000 * sizeof(double), cudaMemcpyDeviceToHost));
}

void pullpn_lnStateFromDevice() {
    pullinSynpn_lnFromDevice();
}


// ------------------------------------------------------------------------
// helper getter functions
// ------------------------------------------------------------------------
unsigned int* getlnCurrentSpikes(unsigned int batch) {
    return (glbSpkln);
}

unsigned int& getlnCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntln[0];
}

scalar* getCurrentVln(unsigned int batch) {
    return Vln;
}

scalar* getCurrentaln(unsigned int batch) {
    return aln;
}

unsigned int* getorCurrentSpikes(unsigned int batch) {
    return (glbSpkor);
}

unsigned int& getorCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntor[0];
}

scalar* getCurrentr0or(unsigned int batch) {
    return r0or;
}

scalar* getCurrentrb_0or(unsigned int batch) {
    return rb_0or;
}

scalar* getCurrentra_0or(unsigned int batch) {
    return ra_0or;
}

scalar* getCurrentrb_1or(unsigned int batch) {
    return rb_1or;
}

scalar* getCurrentra_1or(unsigned int batch) {
    return ra_1or;
}

scalar* getCurrentrb_2or(unsigned int batch) {
    return rb_2or;
}

scalar* getCurrentra_2or(unsigned int batch) {
    return ra_2or;
}

scalar* getCurrentraor(unsigned int batch) {
    return raor;
}

scalar* getCurrentkp1cn_0or(unsigned int batch) {
    return kp1cn_0or;
}

scalar* getCurrentkm1_0or(unsigned int batch) {
    return km1_0or;
}

scalar* getCurrentkp2_0or(unsigned int batch) {
    return kp2_0or;
}

scalar* getCurrentkm2_0or(unsigned int batch) {
    return km2_0or;
}

scalar* getCurrentkp1cn_1or(unsigned int batch) {
    return kp1cn_1or;
}

scalar* getCurrentkm1_1or(unsigned int batch) {
    return km1_1or;
}

scalar* getCurrentkp2_1or(unsigned int batch) {
    return kp2_1or;
}

scalar* getCurrentkm2_1or(unsigned int batch) {
    return km2_1or;
}

scalar* getCurrentkp1cn_2or(unsigned int batch) {
    return kp1cn_2or;
}

scalar* getCurrentkm1_2or(unsigned int batch) {
    return km1_2or;
}

scalar* getCurrentkp2_2or(unsigned int batch) {
    return kp2_2or;
}

scalar* getCurrentkm2_2or(unsigned int batch) {
    return km2_2or;
}

unsigned int* getornCurrentSpikes(unsigned int batch) {
    return (glbSpkorn);
}

unsigned int& getornCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntorn[0];
}

scalar* getCurrentVorn(unsigned int batch) {
    return Vorn;
}

scalar* getCurrentaorn(unsigned int batch) {
    return aorn;
}

unsigned int* getpnCurrentSpikes(unsigned int batch) {
    return (glbSpkpn);
}

unsigned int& getpnCurrentSpikeCount(unsigned int batch) {
    return glbSpkCntpn[0];
}

scalar* getCurrentVpn(unsigned int batch) {
    return Vpn;
}

scalar* getCurrentapn(unsigned int batch) {
    return apn;
}


void copyStateToDevice(bool uninitialisedOnly) {
    pushlnStateToDevice(uninitialisedOnly);
    pushorStateToDevice(uninitialisedOnly);
    pushornStateToDevice(uninitialisedOnly);
    pushpnStateToDevice(uninitialisedOnly);
    pushln_lnStateToDevice(uninitialisedOnly);
    pushln_pnStateToDevice(uninitialisedOnly);
    pushor_ornStateToDevice(uninitialisedOnly);
    pushorn_lnStateToDevice(uninitialisedOnly);
    pushorn_pnStateToDevice(uninitialisedOnly);
    pushpn_lnStateToDevice(uninitialisedOnly);
}

void copyConnectivityToDevice(bool uninitialisedOnly) {
    pushor_ornConnectivityToDevice(uninitialisedOnly);
    pushorn_lnConnectivityToDevice(uninitialisedOnly);
    pushorn_pnConnectivityToDevice(uninitialisedOnly);
    pushpn_lnConnectivityToDevice(uninitialisedOnly);
}

void copyStateFromDevice() {
    pulllnStateFromDevice();
    pullorStateFromDevice();
    pullornStateFromDevice();
    pullpnStateFromDevice();
    pullln_lnStateFromDevice();
    pullln_pnStateFromDevice();
    pullor_ornStateFromDevice();
    pullorn_lnStateFromDevice();
    pullorn_pnStateFromDevice();
    pullpn_lnStateFromDevice();
}

void copyCurrentSpikesFromDevice() {
    pulllnCurrentSpikesFromDevice();
    pullorCurrentSpikesFromDevice();
    pullornCurrentSpikesFromDevice();
    pullpnCurrentSpikesFromDevice();
}

void copyCurrentSpikeEventsFromDevice() {
}

void allocateRecordingBuffers(unsigned int timesteps) {
    numRecordingTimesteps = timesteps;
     {
        const unsigned int numWords = 125 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkln, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkln, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate3recordSpkToDevice(0, d_recordSpkln);
        }
    }
     {
    }
     {
        const unsigned int numWords = 300 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkorn, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkorn, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate1recordSpkToDevice(0, d_recordSpkorn);
        }
    }
     {
        const unsigned int numWords = 25 * timesteps;
         {
            CHECK_CUDA_ERRORS(cudaHostAlloc(&recordSpkpn, numWords * sizeof(uint32_t), cudaHostAllocPortable));
            CHECK_CUDA_ERRORS(cudaMalloc(&d_recordSpkpn, numWords * sizeof(uint32_t)));
            pushMergedNeuronUpdate0recordSpkToDevice(0, d_recordSpkpn);
        }
    }
}

void pullRecordingBuffersFromDevice() {
    if(numRecordingTimesteps == 0) {
        throw std::runtime_error("Recording buffer not allocated - cannot pull from device");
    }
     {
        const unsigned int numWords = 125 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkln, d_recordSpkln, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
    }
     {
        const unsigned int numWords = 300 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkorn, d_recordSpkorn, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
     {
        const unsigned int numWords = 25 * numRecordingTimesteps;
         {
            CHECK_CUDA_ERRORS(cudaMemcpy(recordSpkpn, d_recordSpkpn, numWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
    }
}

void allocateMem() {
    int deviceID;
    CHECK_CUDA_ERRORS(cudaDeviceGetByPCIBusId(&deviceID, "0000:01:00.0"));
    CHECK_CUDA_ERRORS(cudaSetDevice(deviceID));
    
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntln, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntln, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkln, 4000 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkln, 4000 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rngln, 4000 * sizeof(curandState)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vln, 4000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vln, 4000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aln, 4000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aln, 4000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntor, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntor, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkor, 160 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkor, 160 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&r0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_r0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rb_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rb_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ra_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ra_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rb_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rb_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ra_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ra_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rb_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rb_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&ra_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_ra_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&raor, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_raor, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&km1_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_km1_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&km2_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_km2_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&km1_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_km1_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&km2_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_km2_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&km1_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_km1_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&km2_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_km2_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntorn, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntorn, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkorn, 9600 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkorn, 9600 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rngorn, 9600 * sizeof(curandState)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vorn, 9600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vorn, 9600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&aorn, 9600 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_aorn, 9600 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkCntpn, 1 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkCntpn, 1 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&glbSpkpn, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_glbSpkpn, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rngpn, 800 * sizeof(curandState)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&Vpn, 800 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_Vpn, 800 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&apn, 800 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_apn, 800 * sizeof(scalar)));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynpn_ln, 4000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynpn_ln, 4000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynorn_ln, 4000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynorn_ln, 4000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynln_ln, 4000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynln_ln, 4000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynor_orn, 9600 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynor_orn, 9600 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynln_pn, 800 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynln_pn, 800 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynorn_pn, 800 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynorn_pn, 800 * sizeof(double)));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthor_orn, 160 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthor_orn, 160 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indor_orn, 9600 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indor_orn, 9600 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthorn_ln, 9600 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthorn_ln, 9600 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indorn_ln, 38400000 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indorn_ln, 38400000 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthorn_pn, 9600 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthorn_pn, 9600 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indorn_pn, 7680000 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indorn_pn, 7680000 * sizeof(uint32_t)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&rowLengthpn_ln, 800 * sizeof(unsigned int), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_rowLengthpn_ln, 800 * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indpn_ln, 20000 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indpn_ln, 20000 * sizeof(uint32_t)));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gln_ln, 16000000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gln_ln, 16000000 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&gln_pn, 3200000 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gln_pn, 3200000 * sizeof(scalar)));
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntpn, d_glbSpkpn, d_rngpn, d_inSynln_pn, d_inSynorn_pn, 800);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntorn, d_glbSpkorn, d_rngorn, d_inSynor_orn, 9600);
    pushMergedNeuronInitGroup2ToDevice(0, d_glbSpkCntor, d_glbSpkor, 160);
    pushMergedNeuronInitGroup3ToDevice(0, d_glbSpkCntln, d_glbSpkln, d_rngln, d_inSynpn_ln, d_inSynorn_ln, d_inSynln_ln, 4000);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthpn_ln, d_indpn_ln, 25, 800, 4000);
    pushMergedSynapseConnectivityInitGroup1ToDevice(0, d_rowLengthorn_ln, d_indorn_ln, 2.50000000000000000e+01, 4000, 9600, 4000);
    pushMergedSynapseConnectivityInitGroup1ToDevice(1, d_rowLengthorn_pn, d_indorn_pn, 5.00000000000000000e+00, 800, 9600, 800);
    pushMergedSynapseConnectivityInitGroup2ToDevice(0, d_rowLengthor_orn, d_indor_orn, 60, 160, 9600);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntpn, d_glbSpkpn, d_rngpn, d_Vpn, d_apn, d_inSynorn_pn, d_inSynln_pn, d_recordSpkpn, 800);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_glbSpkCntorn, d_glbSpkorn, d_rngorn, d_Vorn, d_aorn, d_inSynor_orn, d_recordSpkorn, 9600);
    pushMergedNeuronUpdateGroup2ToDevice(0, d_kp1cn_0or, d_km2_2or, d_kp2_2or, d_km1_2or, d_kp1cn_2or, d_km2_1or, d_kp2_1or, d_km1_1or, d_kp1cn_1or, d_km2_0or, d_kp2_0or, d_km1_0or, d_raor, d_ra_2or, d_rb_2or, d_ra_1or, d_rb_1or, d_ra_0or, d_rb_0or, d_r0or, d_glbSpkor, d_glbSpkCntor, 160);
    pushMergedNeuronUpdateGroup3ToDevice(0, d_glbSpkCntln, d_glbSpkln, d_rngln, d_Vln, d_aln, d_inSynpn_ln, d_inSynorn_ln, d_inSynln_ln, d_recordSpkln, 4000);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynorn_ln, d_glbSpkCntorn, d_glbSpkorn, d_rowLengthorn_ln, d_indorn_ln, 8.00000000000000017e-03, 4000, 9600, 4000);
    pushMergedPresynapticUpdateGroup0ToDevice(1, d_inSynorn_pn, d_glbSpkCntorn, d_glbSpkorn, d_rowLengthorn_pn, d_indorn_pn, 8.00000000000000017e-03, 800, 9600, 800);
    pushMergedPresynapticUpdateGroup0ToDevice(2, d_inSynpn_ln, d_glbSpkCntpn, d_glbSpkpn, d_rowLengthpn_ln, d_indpn_ln, 1.00000000000000002e-03, 25, 800, 4000);
    pushMergedPresynapticUpdateGroup1ToDevice(0, d_inSynln_ln, d_glbSpkCntln, d_glbSpkln, d_gln_ln, 4000, 4000, 4000);
    pushMergedPresynapticUpdateGroup1ToDevice(1, d_inSynln_pn, d_glbSpkCntln, d_glbSpkln, d_gln_pn, 800, 4000, 800);
    pushMergedSynapseDynamicsGroup0ToDevice(0, d_inSynor_orn, d_raor, d_rowLengthor_orn, d_indor_orn, 60, 160, 9600);
    pushMergedNeuronSpikeQueueUpdateGroup0ToDevice(0, d_glbSpkCntor);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(0, d_glbSpkCntln);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(1, d_glbSpkCntorn);
    pushMergedNeuronSpikeQueueUpdateGroup1ToDevice(2, d_glbSpkCntpn);
}

void freeMem() {
    // ------------------------------------------------------------------------
    // global variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // timers
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // local neuron groups
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntln));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntln));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkln));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkln));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkln));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkln));
    CHECK_CUDA_ERRORS(cudaFree(d_rngln));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vln));
    CHECK_CUDA_ERRORS(cudaFree(d_Vln));
    CHECK_CUDA_ERRORS(cudaFreeHost(aln));
    CHECK_CUDA_ERRORS(cudaFree(d_aln));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntor));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntor));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkor));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkor));
    CHECK_CUDA_ERRORS(cudaFreeHost(r0or));
    CHECK_CUDA_ERRORS(cudaFree(d_r0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(rb_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_rb_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(ra_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_ra_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(rb_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_rb_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(ra_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_ra_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(rb_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_rb_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(ra_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_ra_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(raor));
    CHECK_CUDA_ERRORS(cudaFree(d_raor));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(km1_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_km1_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(km2_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_km2_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(km1_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_km1_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(km2_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_km2_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(km1_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_km1_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(km2_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_km2_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntorn));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntorn));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkorn));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkorn));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkorn));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkorn));
    CHECK_CUDA_ERRORS(cudaFree(d_rngorn));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vorn));
    CHECK_CUDA_ERRORS(cudaFree(d_Vorn));
    CHECK_CUDA_ERRORS(cudaFreeHost(aorn));
    CHECK_CUDA_ERRORS(cudaFree(d_aorn));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntpn));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntpn));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkpn));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkpn));
    CHECK_CUDA_ERRORS(cudaFreeHost(recordSpkpn));
    CHECK_CUDA_ERRORS(cudaFree(d_recordSpkpn));
    CHECK_CUDA_ERRORS(cudaFree(d_rngpn));
    CHECK_CUDA_ERRORS(cudaFreeHost(Vpn));
    CHECK_CUDA_ERRORS(cudaFree(d_Vpn));
    CHECK_CUDA_ERRORS(cudaFreeHost(apn));
    CHECK_CUDA_ERRORS(cudaFree(d_apn));
    
    // ------------------------------------------------------------------------
    // custom update variables
    // ------------------------------------------------------------------------
    
    // ------------------------------------------------------------------------
    // pre and postsynaptic variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynpn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynpn_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynorn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynorn_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynln_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynln_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynor_orn));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynor_orn));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynln_pn));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynln_pn));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynorn_pn));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynorn_pn));
    
    // ------------------------------------------------------------------------
    // synapse connectivity
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthor_orn));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthor_orn));
    CHECK_CUDA_ERRORS(cudaFreeHost(indor_orn));
    CHECK_CUDA_ERRORS(cudaFree(d_indor_orn));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthorn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthorn_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(indorn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_indorn_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthorn_pn));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthorn_pn));
    CHECK_CUDA_ERRORS(cudaFreeHost(indorn_pn));
    CHECK_CUDA_ERRORS(cudaFree(d_indorn_pn));
    CHECK_CUDA_ERRORS(cudaFreeHost(rowLengthpn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_rowLengthpn_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(indpn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_indpn_ln));
    
    // ------------------------------------------------------------------------
    // synapse variables
    // ------------------------------------------------------------------------
    CHECK_CUDA_ERRORS(cudaFreeHost(gln_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_gln_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(gln_pn));
    CHECK_CUDA_ERRORS(cudaFree(d_gln_pn));
    
}

size_t getFreeDeviceMemBytes() {
    size_t free;
    size_t total;
    CHECK_CUDA_ERRORS(cudaMemGetInfo(&free, &total));
    return free;
}

void stepTime() {
    updateSynapses(t);
    updateNeurons(t, (unsigned int)(iT % numRecordingTimesteps)); 
    iT++;
    t = iT*DT;
}

