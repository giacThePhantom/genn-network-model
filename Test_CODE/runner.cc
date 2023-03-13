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
scalar* kp1cn_0_0or;
scalar* d_kp1cn_0_0or;
scalar* kp2_0_0or;
scalar* d_kp2_0_0or;
scalar* kp1cn_0_1or;
scalar* d_kp1cn_0_1or;
scalar* kp2_0_1or;
scalar* d_kp2_0_1or;
scalar* kp1cn_0_2or;
scalar* d_kp1cn_0_2or;
scalar* kp2_0_2or;
scalar* d_kp2_0_2or;
scalar* kp1cn_0_3or;
scalar* d_kp1cn_0_3or;
scalar* kp2_0_3or;
scalar* d_kp2_0_3or;
scalar* kp1cn_0_4or;
scalar* d_kp1cn_0_4or;
scalar* kp2_0_4or;
scalar* d_kp2_0_4or;
scalar* kp1cn_0_5or;
scalar* d_kp1cn_0_5or;
scalar* kp2_0_5or;
scalar* d_kp2_0_5or;
scalar* kp1cn_0_6or;
scalar* d_kp1cn_0_6or;
scalar* kp2_0_6or;
scalar* d_kp2_0_6or;
scalar* kp1cn_0_7or;
scalar* d_kp1cn_0_7or;
scalar* kp2_0_7or;
scalar* d_kp2_0_7or;
scalar* kp1cn_0_8or;
scalar* d_kp1cn_0_8or;
scalar* kp2_0_8or;
scalar* d_kp2_0_8or;
scalar* kp1cn_0_9or;
scalar* d_kp1cn_0_9or;
scalar* kp2_0_9or;
scalar* d_kp2_0_9or;
scalar* kp1cn_0_10or;
scalar* d_kp1cn_0_10or;
scalar* kp2_0_10or;
scalar* d_kp2_0_10or;
scalar* kp1cn_0_11or;
scalar* d_kp1cn_0_11or;
scalar* kp2_0_11or;
scalar* d_kp2_0_11or;
scalar* kp1cn_0_12or;
scalar* d_kp1cn_0_12or;
scalar* kp2_0_12or;
scalar* d_kp2_0_12or;
scalar* kp1cn_0_13or;
scalar* d_kp1cn_0_13or;
scalar* kp2_0_13or;
scalar* d_kp2_0_13or;
scalar* kp1cn_0_14or;
scalar* d_kp1cn_0_14or;
scalar* kp2_0_14or;
scalar* d_kp2_0_14or;
scalar* kp1cn_0_15or;
scalar* d_kp1cn_0_15or;
scalar* kp2_0_15or;
scalar* d_kp2_0_15or;
scalar* kp1cn_0_16or;
scalar* d_kp1cn_0_16or;
scalar* kp2_0_16or;
scalar* d_kp2_0_16or;
scalar* kp1cn_0_17or;
scalar* d_kp1cn_0_17or;
scalar* kp2_0_17or;
scalar* d_kp2_0_17or;
scalar* kp1cn_0_18or;
scalar* d_kp1cn_0_18or;
scalar* kp2_0_18or;
scalar* d_kp2_0_18or;
scalar* kp1cn_0_19or;
scalar* d_kp1cn_0_19or;
scalar* kp2_0_19or;
scalar* d_kp2_0_19or;
scalar* kp1cn_0_20or;
scalar* d_kp1cn_0_20or;
scalar* kp2_0_20or;
scalar* d_kp2_0_20or;
scalar* kp1cn_0_21or;
scalar* d_kp1cn_0_21or;
scalar* kp2_0_21or;
scalar* d_kp2_0_21or;
scalar* kp1cn_0_22or;
scalar* d_kp1cn_0_22or;
scalar* kp2_0_22or;
scalar* d_kp2_0_22or;
scalar* kp1cn_0_23or;
scalar* d_kp1cn_0_23or;
scalar* kp2_0_23or;
scalar* d_kp2_0_23or;
scalar* kp1cn_0_24or;
scalar* d_kp1cn_0_24or;
scalar* kp2_0_24or;
scalar* d_kp2_0_24or;
scalar* kp1cn_0_25or;
scalar* d_kp1cn_0_25or;
scalar* kp2_0_25or;
scalar* d_kp2_0_25or;
scalar* kp1cn_0_26or;
scalar* d_kp1cn_0_26or;
scalar* kp2_0_26or;
scalar* d_kp2_0_26or;
scalar* kp1cn_0_27or;
scalar* d_kp1cn_0_27or;
scalar* kp2_0_27or;
scalar* d_kp2_0_27or;
scalar* kp1cn_0_28or;
scalar* d_kp1cn_0_28or;
scalar* kp2_0_28or;
scalar* d_kp2_0_28or;
scalar* kp1cn_0_29or;
scalar* d_kp1cn_0_29or;
scalar* kp2_0_29or;
scalar* d_kp2_0_29or;
scalar* kp1cn_0_30or;
scalar* d_kp1cn_0_30or;
scalar* kp2_0_30or;
scalar* d_kp2_0_30or;
scalar* kp1cn_0_31or;
scalar* d_kp1cn_0_31or;
scalar* kp2_0_31or;
scalar* d_kp2_0_31or;
scalar* kp1cn_0_32or;
scalar* d_kp1cn_0_32or;
scalar* kp2_0_32or;
scalar* d_kp2_0_32or;
scalar* kp1cn_0_33or;
scalar* d_kp1cn_0_33or;
scalar* kp2_0_33or;
scalar* d_kp2_0_33or;
scalar* kp1cn_0_34or;
scalar* d_kp1cn_0_34or;
scalar* kp2_0_34or;
scalar* d_kp2_0_34or;
scalar* kp1cn_0_35or;
scalar* d_kp1cn_0_35or;
scalar* kp2_0_35or;
scalar* d_kp2_0_35or;
scalar* kp1cn_0_36or;
scalar* d_kp1cn_0_36or;
scalar* kp2_0_36or;
scalar* d_kp2_0_36or;
scalar* kp1cn_0_37or;
scalar* d_kp1cn_0_37or;
scalar* kp2_0_37or;
scalar* d_kp2_0_37or;
scalar* kp1cn_0_38or;
scalar* d_kp1cn_0_38or;
scalar* kp2_0_38or;
scalar* d_kp2_0_38or;
scalar* kp1cn_0_39or;
scalar* d_kp1cn_0_39or;
scalar* kp2_0_39or;
scalar* d_kp2_0_39or;
scalar* kp1cn_0_40or;
scalar* d_kp1cn_0_40or;
scalar* kp2_0_40or;
scalar* d_kp2_0_40or;
scalar* kp1cn_0_41or;
scalar* d_kp1cn_0_41or;
scalar* kp2_0_41or;
scalar* d_kp2_0_41or;
scalar* kp1cn_0_42or;
scalar* d_kp1cn_0_42or;
scalar* kp2_0_42or;
scalar* d_kp2_0_42or;
scalar* kp1cn_0_43or;
scalar* d_kp1cn_0_43or;
scalar* kp2_0_43or;
scalar* d_kp2_0_43or;
scalar* kp1cn_0_44or;
scalar* d_kp1cn_0_44or;
scalar* kp2_0_44or;
scalar* d_kp2_0_44or;
scalar* kp1cn_0_45or;
scalar* d_kp1cn_0_45or;
scalar* kp2_0_45or;
scalar* d_kp2_0_45or;
scalar* kp1cn_0_46or;
scalar* d_kp1cn_0_46or;
scalar* kp2_0_46or;
scalar* d_kp2_0_46or;
scalar* kp1cn_0_47or;
scalar* d_kp1cn_0_47or;
scalar* kp2_0_47or;
scalar* d_kp2_0_47or;
scalar* kp1cn_0_48or;
scalar* d_kp1cn_0_48or;
scalar* kp2_0_48or;
scalar* d_kp2_0_48or;
scalar* kp1cn_0_49or;
scalar* d_kp1cn_0_49or;
scalar* kp2_0_49or;
scalar* d_kp2_0_49or;
scalar* kp1cn_0_50or;
scalar* d_kp1cn_0_50or;
scalar* kp2_0_50or;
scalar* d_kp2_0_50or;
scalar* kp1cn_0_51or;
scalar* d_kp1cn_0_51or;
scalar* kp2_0_51or;
scalar* d_kp2_0_51or;
scalar* kp1cn_0_52or;
scalar* d_kp1cn_0_52or;
scalar* kp2_0_52or;
scalar* d_kp2_0_52or;
scalar* kp1cn_0_53or;
scalar* d_kp1cn_0_53or;
scalar* kp2_0_53or;
scalar* d_kp2_0_53or;
scalar* kp1cn_0_54or;
scalar* d_kp1cn_0_54or;
scalar* kp2_0_54or;
scalar* d_kp2_0_54or;
scalar* kp1cn_0_55or;
scalar* d_kp1cn_0_55or;
scalar* kp2_0_55or;
scalar* d_kp2_0_55or;
scalar* kp1cn_0_56or;
scalar* d_kp1cn_0_56or;
scalar* kp2_0_56or;
scalar* d_kp2_0_56or;
scalar* kp1cn_0_57or;
scalar* d_kp1cn_0_57or;
scalar* kp2_0_57or;
scalar* d_kp2_0_57or;
scalar* kp1cn_0_58or;
scalar* d_kp1cn_0_58or;
scalar* kp2_0_58or;
scalar* d_kp2_0_58or;
scalar* kp1cn_0_59or;
scalar* d_kp1cn_0_59or;
scalar* kp2_0_59or;
scalar* d_kp2_0_59or;
scalar* kp1cn_0_60or;
scalar* d_kp1cn_0_60or;
scalar* kp2_0_60or;
scalar* d_kp2_0_60or;
scalar* kp1cn_0_61or;
scalar* d_kp1cn_0_61or;
scalar* kp2_0_61or;
scalar* d_kp2_0_61or;
scalar* kp1cn_0_62or;
scalar* d_kp1cn_0_62or;
scalar* kp2_0_62or;
scalar* d_kp2_0_62or;
scalar* kp1cn_0_63or;
scalar* d_kp1cn_0_63or;
scalar* kp2_0_63or;
scalar* d_kp2_0_63or;
scalar* kp1cn_0_64or;
scalar* d_kp1cn_0_64or;
scalar* kp2_0_64or;
scalar* d_kp2_0_64or;
scalar* kp1cn_0_65or;
scalar* d_kp1cn_0_65or;
scalar* kp2_0_65or;
scalar* d_kp2_0_65or;
scalar* kp1cn_0_66or;
scalar* d_kp1cn_0_66or;
scalar* kp2_0_66or;
scalar* d_kp2_0_66or;
scalar* kp1cn_0_67or;
scalar* d_kp1cn_0_67or;
scalar* kp2_0_67or;
scalar* d_kp2_0_67or;
scalar* kp1cn_0_68or;
scalar* d_kp1cn_0_68or;
scalar* kp2_0_68or;
scalar* d_kp2_0_68or;
scalar* kp1cn_0_69or;
scalar* d_kp1cn_0_69or;
scalar* kp2_0_69or;
scalar* d_kp2_0_69or;
scalar* kp1cn_0_70or;
scalar* d_kp1cn_0_70or;
scalar* kp2_0_70or;
scalar* d_kp2_0_70or;
scalar* kp1cn_0_71or;
scalar* d_kp1cn_0_71or;
scalar* kp2_0_71or;
scalar* d_kp2_0_71or;
scalar* kp1cn_0_72or;
scalar* d_kp1cn_0_72or;
scalar* kp2_0_72or;
scalar* d_kp2_0_72or;
scalar* kp1cn_0_73or;
scalar* d_kp1cn_0_73or;
scalar* kp2_0_73or;
scalar* d_kp2_0_73or;
scalar* kp1cn_0_74or;
scalar* d_kp1cn_0_74or;
scalar* kp2_0_74or;
scalar* d_kp2_0_74or;
scalar* kp1cn_0_75or;
scalar* d_kp1cn_0_75or;
scalar* kp2_0_75or;
scalar* d_kp2_0_75or;
scalar* kp1cn_0_76or;
scalar* d_kp1cn_0_76or;
scalar* kp2_0_76or;
scalar* d_kp2_0_76or;
scalar* kp1cn_0_77or;
scalar* d_kp1cn_0_77or;
scalar* kp2_0_77or;
scalar* d_kp2_0_77or;
scalar* kp1cn_0_78or;
scalar* d_kp1cn_0_78or;
scalar* kp2_0_78or;
scalar* d_kp2_0_78or;
scalar* kp1cn_0_79or;
scalar* d_kp1cn_0_79or;
scalar* kp2_0_79or;
scalar* d_kp2_0_79or;
scalar* kp1cn_0_80or;
scalar* d_kp1cn_0_80or;
scalar* kp2_0_80or;
scalar* d_kp2_0_80or;
scalar* kp1cn_0_81or;
scalar* d_kp1cn_0_81or;
scalar* kp2_0_81or;
scalar* d_kp2_0_81or;
scalar* kp1cn_0_82or;
scalar* d_kp1cn_0_82or;
scalar* kp2_0_82or;
scalar* d_kp2_0_82or;
scalar* kp1cn_0_83or;
scalar* d_kp1cn_0_83or;
scalar* kp2_0_83or;
scalar* d_kp2_0_83or;
scalar* kp1cn_0_84or;
scalar* d_kp1cn_0_84or;
scalar* kp2_0_84or;
scalar* d_kp2_0_84or;
scalar* kp1cn_0_85or;
scalar* d_kp1cn_0_85or;
scalar* kp2_0_85or;
scalar* d_kp2_0_85or;
scalar* kp1cn_0_86or;
scalar* d_kp1cn_0_86or;
scalar* kp2_0_86or;
scalar* d_kp2_0_86or;
scalar* kp1cn_0_87or;
scalar* d_kp1cn_0_87or;
scalar* kp2_0_87or;
scalar* d_kp2_0_87or;
scalar* kp1cn_0_88or;
scalar* d_kp1cn_0_88or;
scalar* kp2_0_88or;
scalar* d_kp2_0_88or;
scalar* kp1cn_0_89or;
scalar* d_kp1cn_0_89or;
scalar* kp2_0_89or;
scalar* d_kp2_0_89or;
scalar* kp1cn_0_90or;
scalar* d_kp1cn_0_90or;
scalar* kp2_0_90or;
scalar* d_kp2_0_90or;
scalar* kp1cn_0_91or;
scalar* d_kp1cn_0_91or;
scalar* kp2_0_91or;
scalar* d_kp2_0_91or;
scalar* kp1cn_0_92or;
scalar* d_kp1cn_0_92or;
scalar* kp2_0_92or;
scalar* d_kp2_0_92or;
scalar* kp1cn_0_93or;
scalar* d_kp1cn_0_93or;
scalar* kp2_0_93or;
scalar* d_kp2_0_93or;
scalar* kp1cn_0_94or;
scalar* d_kp1cn_0_94or;
scalar* kp2_0_94or;
scalar* d_kp2_0_94or;
scalar* kp1cn_0_95or;
scalar* d_kp1cn_0_95or;
scalar* kp2_0_95or;
scalar* d_kp2_0_95or;
scalar* kp1cn_0_96or;
scalar* d_kp1cn_0_96or;
scalar* kp2_0_96or;
scalar* d_kp2_0_96or;
scalar* kp1cn_0_97or;
scalar* d_kp1cn_0_97or;
scalar* kp2_0_97or;
scalar* d_kp2_0_97or;
scalar* kp1cn_0_98or;
scalar* d_kp1cn_0_98or;
scalar* kp2_0_98or;
scalar* d_kp2_0_98or;
scalar* kp1cn_0_99or;
scalar* d_kp1cn_0_99or;
scalar* kp2_0_99or;
scalar* d_kp2_0_99or;
unsigned int* glbSpkCntorn;
unsigned int* d_glbSpkCntorn;
unsigned int* glbSpkorn;
unsigned int* d_glbSpkorn;
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
double* inSynln_ln;
double* d_inSynln_ln;
double* inSynor_orn;
double* d_inSynor_orn;
double* inSynorn_ln;
double* d_inSynorn_ln;
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
const unsigned int maxRowLengthorn_ln = 800;
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

void pushkp1cn_0_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_0or, kp1cn_0_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_0or, kp1cn_0_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_0or, kp2_0_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_0orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_0or, kp2_0_0or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_1or, kp1cn_0_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_1or, kp1cn_0_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_1or, kp2_0_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_1orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_1or, kp2_0_1or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_2or, kp1cn_0_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_2or, kp1cn_0_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_2or, kp2_0_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_2orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_2or, kp2_0_2or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_3orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_3or, kp1cn_0_3or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_3orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_3or, kp1cn_0_3or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_3orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_3or, kp2_0_3or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_3orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_3or, kp2_0_3or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_4orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_4or, kp1cn_0_4or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_4orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_4or, kp1cn_0_4or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_4orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_4or, kp2_0_4or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_4orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_4or, kp2_0_4or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_5orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_5or, kp1cn_0_5or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_5orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_5or, kp1cn_0_5or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_5orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_5or, kp2_0_5or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_5orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_5or, kp2_0_5or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_6orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_6or, kp1cn_0_6or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_6orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_6or, kp1cn_0_6or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_6orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_6or, kp2_0_6or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_6orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_6or, kp2_0_6or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_7orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_7or, kp1cn_0_7or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_7orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_7or, kp1cn_0_7or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_7orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_7or, kp2_0_7or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_7orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_7or, kp2_0_7or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_8orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_8or, kp1cn_0_8or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_8orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_8or, kp1cn_0_8or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_8orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_8or, kp2_0_8or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_8orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_8or, kp2_0_8or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_9orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_9or, kp1cn_0_9or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_9orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_9or, kp1cn_0_9or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_9orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_9or, kp2_0_9or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_9orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_9or, kp2_0_9or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_10orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_10or, kp1cn_0_10or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_10orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_10or, kp1cn_0_10or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_10orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_10or, kp2_0_10or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_10orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_10or, kp2_0_10or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_11orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_11or, kp1cn_0_11or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_11orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_11or, kp1cn_0_11or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_11orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_11or, kp2_0_11or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_11orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_11or, kp2_0_11or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_12orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_12or, kp1cn_0_12or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_12orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_12or, kp1cn_0_12or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_12orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_12or, kp2_0_12or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_12orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_12or, kp2_0_12or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_13orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_13or, kp1cn_0_13or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_13orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_13or, kp1cn_0_13or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_13orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_13or, kp2_0_13or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_13orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_13or, kp2_0_13or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_14orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_14or, kp1cn_0_14or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_14orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_14or, kp1cn_0_14or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_14orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_14or, kp2_0_14or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_14orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_14or, kp2_0_14or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_15orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_15or, kp1cn_0_15or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_15orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_15or, kp1cn_0_15or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_15orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_15or, kp2_0_15or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_15orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_15or, kp2_0_15or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_16orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_16or, kp1cn_0_16or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_16orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_16or, kp1cn_0_16or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_16orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_16or, kp2_0_16or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_16orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_16or, kp2_0_16or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_17orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_17or, kp1cn_0_17or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_17orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_17or, kp1cn_0_17or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_17orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_17or, kp2_0_17or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_17orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_17or, kp2_0_17or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_18orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_18or, kp1cn_0_18or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_18orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_18or, kp1cn_0_18or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_18orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_18or, kp2_0_18or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_18orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_18or, kp2_0_18or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_19orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_19or, kp1cn_0_19or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_19orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_19or, kp1cn_0_19or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_19orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_19or, kp2_0_19or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_19orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_19or, kp2_0_19or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_20orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_20or, kp1cn_0_20or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_20orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_20or, kp1cn_0_20or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_20orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_20or, kp2_0_20or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_20orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_20or, kp2_0_20or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_21orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_21or, kp1cn_0_21or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_21orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_21or, kp1cn_0_21or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_21orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_21or, kp2_0_21or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_21orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_21or, kp2_0_21or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_22orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_22or, kp1cn_0_22or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_22orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_22or, kp1cn_0_22or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_22orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_22or, kp2_0_22or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_22orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_22or, kp2_0_22or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_23orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_23or, kp1cn_0_23or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_23orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_23or, kp1cn_0_23or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_23orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_23or, kp2_0_23or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_23orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_23or, kp2_0_23or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_24orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_24or, kp1cn_0_24or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_24orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_24or, kp1cn_0_24or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_24orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_24or, kp2_0_24or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_24orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_24or, kp2_0_24or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_25orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_25or, kp1cn_0_25or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_25orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_25or, kp1cn_0_25or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_25orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_25or, kp2_0_25or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_25orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_25or, kp2_0_25or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_26orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_26or, kp1cn_0_26or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_26orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_26or, kp1cn_0_26or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_26orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_26or, kp2_0_26or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_26orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_26or, kp2_0_26or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_27orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_27or, kp1cn_0_27or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_27orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_27or, kp1cn_0_27or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_27orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_27or, kp2_0_27or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_27orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_27or, kp2_0_27or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_28orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_28or, kp1cn_0_28or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_28orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_28or, kp1cn_0_28or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_28orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_28or, kp2_0_28or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_28orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_28or, kp2_0_28or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_29orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_29or, kp1cn_0_29or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_29orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_29or, kp1cn_0_29or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_29orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_29or, kp2_0_29or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_29orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_29or, kp2_0_29or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_30orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_30or, kp1cn_0_30or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_30orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_30or, kp1cn_0_30or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_30orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_30or, kp2_0_30or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_30orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_30or, kp2_0_30or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_31orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_31or, kp1cn_0_31or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_31orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_31or, kp1cn_0_31or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_31orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_31or, kp2_0_31or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_31orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_31or, kp2_0_31or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_32orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_32or, kp1cn_0_32or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_32orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_32or, kp1cn_0_32or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_32orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_32or, kp2_0_32or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_32orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_32or, kp2_0_32or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_33orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_33or, kp1cn_0_33or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_33orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_33or, kp1cn_0_33or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_33orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_33or, kp2_0_33or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_33orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_33or, kp2_0_33or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_34orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_34or, kp1cn_0_34or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_34orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_34or, kp1cn_0_34or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_34orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_34or, kp2_0_34or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_34orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_34or, kp2_0_34or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_35orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_35or, kp1cn_0_35or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_35orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_35or, kp1cn_0_35or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_35orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_35or, kp2_0_35or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_35orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_35or, kp2_0_35or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_36orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_36or, kp1cn_0_36or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_36orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_36or, kp1cn_0_36or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_36orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_36or, kp2_0_36or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_36orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_36or, kp2_0_36or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_37orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_37or, kp1cn_0_37or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_37orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_37or, kp1cn_0_37or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_37orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_37or, kp2_0_37or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_37orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_37or, kp2_0_37or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_38orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_38or, kp1cn_0_38or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_38orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_38or, kp1cn_0_38or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_38orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_38or, kp2_0_38or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_38orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_38or, kp2_0_38or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_39orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_39or, kp1cn_0_39or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_39orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_39or, kp1cn_0_39or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_39orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_39or, kp2_0_39or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_39orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_39or, kp2_0_39or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_40orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_40or, kp1cn_0_40or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_40orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_40or, kp1cn_0_40or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_40orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_40or, kp2_0_40or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_40orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_40or, kp2_0_40or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_41orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_41or, kp1cn_0_41or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_41orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_41or, kp1cn_0_41or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_41orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_41or, kp2_0_41or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_41orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_41or, kp2_0_41or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_42orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_42or, kp1cn_0_42or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_42orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_42or, kp1cn_0_42or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_42orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_42or, kp2_0_42or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_42orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_42or, kp2_0_42or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_43orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_43or, kp1cn_0_43or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_43orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_43or, kp1cn_0_43or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_43orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_43or, kp2_0_43or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_43orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_43or, kp2_0_43or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_44orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_44or, kp1cn_0_44or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_44orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_44or, kp1cn_0_44or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_44orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_44or, kp2_0_44or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_44orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_44or, kp2_0_44or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_45orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_45or, kp1cn_0_45or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_45orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_45or, kp1cn_0_45or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_45orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_45or, kp2_0_45or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_45orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_45or, kp2_0_45or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_46orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_46or, kp1cn_0_46or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_46orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_46or, kp1cn_0_46or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_46orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_46or, kp2_0_46or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_46orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_46or, kp2_0_46or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_47orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_47or, kp1cn_0_47or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_47orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_47or, kp1cn_0_47or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_47orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_47or, kp2_0_47or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_47orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_47or, kp2_0_47or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_48orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_48or, kp1cn_0_48or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_48orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_48or, kp1cn_0_48or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_48orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_48or, kp2_0_48or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_48orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_48or, kp2_0_48or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_49orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_49or, kp1cn_0_49or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_49orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_49or, kp1cn_0_49or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_49orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_49or, kp2_0_49or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_49orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_49or, kp2_0_49or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_50orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_50or, kp1cn_0_50or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_50orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_50or, kp1cn_0_50or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_50orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_50or, kp2_0_50or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_50orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_50or, kp2_0_50or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_51orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_51or, kp1cn_0_51or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_51orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_51or, kp1cn_0_51or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_51orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_51or, kp2_0_51or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_51orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_51or, kp2_0_51or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_52orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_52or, kp1cn_0_52or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_52orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_52or, kp1cn_0_52or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_52orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_52or, kp2_0_52or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_52orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_52or, kp2_0_52or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_53orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_53or, kp1cn_0_53or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_53orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_53or, kp1cn_0_53or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_53orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_53or, kp2_0_53or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_53orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_53or, kp2_0_53or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_54orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_54or, kp1cn_0_54or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_54orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_54or, kp1cn_0_54or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_54orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_54or, kp2_0_54or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_54orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_54or, kp2_0_54or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_55orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_55or, kp1cn_0_55or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_55orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_55or, kp1cn_0_55or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_55orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_55or, kp2_0_55or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_55orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_55or, kp2_0_55or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_56orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_56or, kp1cn_0_56or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_56orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_56or, kp1cn_0_56or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_56orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_56or, kp2_0_56or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_56orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_56or, kp2_0_56or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_57orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_57or, kp1cn_0_57or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_57orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_57or, kp1cn_0_57or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_57orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_57or, kp2_0_57or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_57orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_57or, kp2_0_57or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_58orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_58or, kp1cn_0_58or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_58orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_58or, kp1cn_0_58or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_58orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_58or, kp2_0_58or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_58orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_58or, kp2_0_58or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_59orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_59or, kp1cn_0_59or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_59orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_59or, kp1cn_0_59or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_59orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_59or, kp2_0_59or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_59orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_59or, kp2_0_59or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_60orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_60or, kp1cn_0_60or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_60orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_60or, kp1cn_0_60or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_60orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_60or, kp2_0_60or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_60orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_60or, kp2_0_60or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_61orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_61or, kp1cn_0_61or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_61orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_61or, kp1cn_0_61or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_61orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_61or, kp2_0_61or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_61orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_61or, kp2_0_61or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_62orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_62or, kp1cn_0_62or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_62orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_62or, kp1cn_0_62or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_62orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_62or, kp2_0_62or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_62orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_62or, kp2_0_62or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_63orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_63or, kp1cn_0_63or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_63orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_63or, kp1cn_0_63or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_63orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_63or, kp2_0_63or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_63orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_63or, kp2_0_63or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_64orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_64or, kp1cn_0_64or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_64orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_64or, kp1cn_0_64or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_64orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_64or, kp2_0_64or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_64orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_64or, kp2_0_64or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_65orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_65or, kp1cn_0_65or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_65orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_65or, kp1cn_0_65or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_65orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_65or, kp2_0_65or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_65orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_65or, kp2_0_65or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_66orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_66or, kp1cn_0_66or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_66orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_66or, kp1cn_0_66or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_66orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_66or, kp2_0_66or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_66orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_66or, kp2_0_66or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_67orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_67or, kp1cn_0_67or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_67orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_67or, kp1cn_0_67or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_67orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_67or, kp2_0_67or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_67orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_67or, kp2_0_67or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_68orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_68or, kp1cn_0_68or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_68orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_68or, kp1cn_0_68or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_68orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_68or, kp2_0_68or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_68orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_68or, kp2_0_68or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_69orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_69or, kp1cn_0_69or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_69orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_69or, kp1cn_0_69or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_69orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_69or, kp2_0_69or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_69orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_69or, kp2_0_69or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_70orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_70or, kp1cn_0_70or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_70orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_70or, kp1cn_0_70or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_70orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_70or, kp2_0_70or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_70orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_70or, kp2_0_70or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_71orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_71or, kp1cn_0_71or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_71orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_71or, kp1cn_0_71or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_71orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_71or, kp2_0_71or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_71orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_71or, kp2_0_71or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_72orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_72or, kp1cn_0_72or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_72orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_72or, kp1cn_0_72or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_72orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_72or, kp2_0_72or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_72orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_72or, kp2_0_72or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_73orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_73or, kp1cn_0_73or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_73orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_73or, kp1cn_0_73or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_73orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_73or, kp2_0_73or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_73orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_73or, kp2_0_73or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_74orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_74or, kp1cn_0_74or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_74orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_74or, kp1cn_0_74or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_74orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_74or, kp2_0_74or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_74orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_74or, kp2_0_74or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_75orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_75or, kp1cn_0_75or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_75orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_75or, kp1cn_0_75or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_75orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_75or, kp2_0_75or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_75orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_75or, kp2_0_75or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_76orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_76or, kp1cn_0_76or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_76orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_76or, kp1cn_0_76or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_76orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_76or, kp2_0_76or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_76orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_76or, kp2_0_76or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_77orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_77or, kp1cn_0_77or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_77orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_77or, kp1cn_0_77or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_77orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_77or, kp2_0_77or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_77orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_77or, kp2_0_77or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_78orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_78or, kp1cn_0_78or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_78orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_78or, kp1cn_0_78or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_78orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_78or, kp2_0_78or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_78orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_78or, kp2_0_78or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_79orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_79or, kp1cn_0_79or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_79orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_79or, kp1cn_0_79or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_79orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_79or, kp2_0_79or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_79orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_79or, kp2_0_79or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_80orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_80or, kp1cn_0_80or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_80orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_80or, kp1cn_0_80or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_80orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_80or, kp2_0_80or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_80orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_80or, kp2_0_80or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_81orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_81or, kp1cn_0_81or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_81orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_81or, kp1cn_0_81or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_81orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_81or, kp2_0_81or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_81orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_81or, kp2_0_81or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_82orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_82or, kp1cn_0_82or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_82orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_82or, kp1cn_0_82or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_82orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_82or, kp2_0_82or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_82orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_82or, kp2_0_82or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_83orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_83or, kp1cn_0_83or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_83orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_83or, kp1cn_0_83or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_83orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_83or, kp2_0_83or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_83orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_83or, kp2_0_83or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_84orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_84or, kp1cn_0_84or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_84orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_84or, kp1cn_0_84or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_84orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_84or, kp2_0_84or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_84orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_84or, kp2_0_84or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_85orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_85or, kp1cn_0_85or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_85orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_85or, kp1cn_0_85or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_85orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_85or, kp2_0_85or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_85orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_85or, kp2_0_85or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_86orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_86or, kp1cn_0_86or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_86orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_86or, kp1cn_0_86or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_86orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_86or, kp2_0_86or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_86orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_86or, kp2_0_86or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_87orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_87or, kp1cn_0_87or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_87orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_87or, kp1cn_0_87or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_87orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_87or, kp2_0_87or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_87orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_87or, kp2_0_87or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_88orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_88or, kp1cn_0_88or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_88orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_88or, kp1cn_0_88or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_88orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_88or, kp2_0_88or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_88orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_88or, kp2_0_88or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_89orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_89or, kp1cn_0_89or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_89orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_89or, kp1cn_0_89or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_89orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_89or, kp2_0_89or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_89orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_89or, kp2_0_89or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_90orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_90or, kp1cn_0_90or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_90orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_90or, kp1cn_0_90or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_90orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_90or, kp2_0_90or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_90orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_90or, kp2_0_90or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_91orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_91or, kp1cn_0_91or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_91orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_91or, kp1cn_0_91or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_91orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_91or, kp2_0_91or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_91orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_91or, kp2_0_91or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_92orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_92or, kp1cn_0_92or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_92orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_92or, kp1cn_0_92or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_92orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_92or, kp2_0_92or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_92orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_92or, kp2_0_92or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_93orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_93or, kp1cn_0_93or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_93orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_93or, kp1cn_0_93or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_93orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_93or, kp2_0_93or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_93orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_93or, kp2_0_93or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_94orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_94or, kp1cn_0_94or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_94orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_94or, kp1cn_0_94or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_94orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_94or, kp2_0_94or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_94orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_94or, kp2_0_94or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_95orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_95or, kp1cn_0_95or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_95orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_95or, kp1cn_0_95or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_95orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_95or, kp2_0_95or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_95orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_95or, kp2_0_95or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_96orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_96or, kp1cn_0_96or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_96orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_96or, kp1cn_0_96or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_96orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_96or, kp2_0_96or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_96orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_96or, kp2_0_96or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_97orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_97or, kp1cn_0_97or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_97orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_97or, kp1cn_0_97or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_97orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_97or, kp2_0_97or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_97orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_97or, kp2_0_97or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_98orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_98or, kp1cn_0_98or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_98orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_98or, kp1cn_0_98or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_98orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_98or, kp2_0_98or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_98orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_98or, kp2_0_98or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp1cn_0_99orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_99or, kp1cn_0_99or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp1cn_0_99orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp1cn_0_99or, kp1cn_0_99or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushkp2_0_99orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_99or, kp2_0_99or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
}

void pushCurrentkp2_0_99orToDevice(bool uninitialisedOnly) {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_kp2_0_99or, kp2_0_99or, 160 * sizeof(scalar), cudaMemcpyHostToDevice));
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
    pushkp1cn_0_0orToDevice(uninitialisedOnly);
    pushkp2_0_0orToDevice(uninitialisedOnly);
    pushkp1cn_0_1orToDevice(uninitialisedOnly);
    pushkp2_0_1orToDevice(uninitialisedOnly);
    pushkp1cn_0_2orToDevice(uninitialisedOnly);
    pushkp2_0_2orToDevice(uninitialisedOnly);
    pushkp1cn_0_3orToDevice(uninitialisedOnly);
    pushkp2_0_3orToDevice(uninitialisedOnly);
    pushkp1cn_0_4orToDevice(uninitialisedOnly);
    pushkp2_0_4orToDevice(uninitialisedOnly);
    pushkp1cn_0_5orToDevice(uninitialisedOnly);
    pushkp2_0_5orToDevice(uninitialisedOnly);
    pushkp1cn_0_6orToDevice(uninitialisedOnly);
    pushkp2_0_6orToDevice(uninitialisedOnly);
    pushkp1cn_0_7orToDevice(uninitialisedOnly);
    pushkp2_0_7orToDevice(uninitialisedOnly);
    pushkp1cn_0_8orToDevice(uninitialisedOnly);
    pushkp2_0_8orToDevice(uninitialisedOnly);
    pushkp1cn_0_9orToDevice(uninitialisedOnly);
    pushkp2_0_9orToDevice(uninitialisedOnly);
    pushkp1cn_0_10orToDevice(uninitialisedOnly);
    pushkp2_0_10orToDevice(uninitialisedOnly);
    pushkp1cn_0_11orToDevice(uninitialisedOnly);
    pushkp2_0_11orToDevice(uninitialisedOnly);
    pushkp1cn_0_12orToDevice(uninitialisedOnly);
    pushkp2_0_12orToDevice(uninitialisedOnly);
    pushkp1cn_0_13orToDevice(uninitialisedOnly);
    pushkp2_0_13orToDevice(uninitialisedOnly);
    pushkp1cn_0_14orToDevice(uninitialisedOnly);
    pushkp2_0_14orToDevice(uninitialisedOnly);
    pushkp1cn_0_15orToDevice(uninitialisedOnly);
    pushkp2_0_15orToDevice(uninitialisedOnly);
    pushkp1cn_0_16orToDevice(uninitialisedOnly);
    pushkp2_0_16orToDevice(uninitialisedOnly);
    pushkp1cn_0_17orToDevice(uninitialisedOnly);
    pushkp2_0_17orToDevice(uninitialisedOnly);
    pushkp1cn_0_18orToDevice(uninitialisedOnly);
    pushkp2_0_18orToDevice(uninitialisedOnly);
    pushkp1cn_0_19orToDevice(uninitialisedOnly);
    pushkp2_0_19orToDevice(uninitialisedOnly);
    pushkp1cn_0_20orToDevice(uninitialisedOnly);
    pushkp2_0_20orToDevice(uninitialisedOnly);
    pushkp1cn_0_21orToDevice(uninitialisedOnly);
    pushkp2_0_21orToDevice(uninitialisedOnly);
    pushkp1cn_0_22orToDevice(uninitialisedOnly);
    pushkp2_0_22orToDevice(uninitialisedOnly);
    pushkp1cn_0_23orToDevice(uninitialisedOnly);
    pushkp2_0_23orToDevice(uninitialisedOnly);
    pushkp1cn_0_24orToDevice(uninitialisedOnly);
    pushkp2_0_24orToDevice(uninitialisedOnly);
    pushkp1cn_0_25orToDevice(uninitialisedOnly);
    pushkp2_0_25orToDevice(uninitialisedOnly);
    pushkp1cn_0_26orToDevice(uninitialisedOnly);
    pushkp2_0_26orToDevice(uninitialisedOnly);
    pushkp1cn_0_27orToDevice(uninitialisedOnly);
    pushkp2_0_27orToDevice(uninitialisedOnly);
    pushkp1cn_0_28orToDevice(uninitialisedOnly);
    pushkp2_0_28orToDevice(uninitialisedOnly);
    pushkp1cn_0_29orToDevice(uninitialisedOnly);
    pushkp2_0_29orToDevice(uninitialisedOnly);
    pushkp1cn_0_30orToDevice(uninitialisedOnly);
    pushkp2_0_30orToDevice(uninitialisedOnly);
    pushkp1cn_0_31orToDevice(uninitialisedOnly);
    pushkp2_0_31orToDevice(uninitialisedOnly);
    pushkp1cn_0_32orToDevice(uninitialisedOnly);
    pushkp2_0_32orToDevice(uninitialisedOnly);
    pushkp1cn_0_33orToDevice(uninitialisedOnly);
    pushkp2_0_33orToDevice(uninitialisedOnly);
    pushkp1cn_0_34orToDevice(uninitialisedOnly);
    pushkp2_0_34orToDevice(uninitialisedOnly);
    pushkp1cn_0_35orToDevice(uninitialisedOnly);
    pushkp2_0_35orToDevice(uninitialisedOnly);
    pushkp1cn_0_36orToDevice(uninitialisedOnly);
    pushkp2_0_36orToDevice(uninitialisedOnly);
    pushkp1cn_0_37orToDevice(uninitialisedOnly);
    pushkp2_0_37orToDevice(uninitialisedOnly);
    pushkp1cn_0_38orToDevice(uninitialisedOnly);
    pushkp2_0_38orToDevice(uninitialisedOnly);
    pushkp1cn_0_39orToDevice(uninitialisedOnly);
    pushkp2_0_39orToDevice(uninitialisedOnly);
    pushkp1cn_0_40orToDevice(uninitialisedOnly);
    pushkp2_0_40orToDevice(uninitialisedOnly);
    pushkp1cn_0_41orToDevice(uninitialisedOnly);
    pushkp2_0_41orToDevice(uninitialisedOnly);
    pushkp1cn_0_42orToDevice(uninitialisedOnly);
    pushkp2_0_42orToDevice(uninitialisedOnly);
    pushkp1cn_0_43orToDevice(uninitialisedOnly);
    pushkp2_0_43orToDevice(uninitialisedOnly);
    pushkp1cn_0_44orToDevice(uninitialisedOnly);
    pushkp2_0_44orToDevice(uninitialisedOnly);
    pushkp1cn_0_45orToDevice(uninitialisedOnly);
    pushkp2_0_45orToDevice(uninitialisedOnly);
    pushkp1cn_0_46orToDevice(uninitialisedOnly);
    pushkp2_0_46orToDevice(uninitialisedOnly);
    pushkp1cn_0_47orToDevice(uninitialisedOnly);
    pushkp2_0_47orToDevice(uninitialisedOnly);
    pushkp1cn_0_48orToDevice(uninitialisedOnly);
    pushkp2_0_48orToDevice(uninitialisedOnly);
    pushkp1cn_0_49orToDevice(uninitialisedOnly);
    pushkp2_0_49orToDevice(uninitialisedOnly);
    pushkp1cn_0_50orToDevice(uninitialisedOnly);
    pushkp2_0_50orToDevice(uninitialisedOnly);
    pushkp1cn_0_51orToDevice(uninitialisedOnly);
    pushkp2_0_51orToDevice(uninitialisedOnly);
    pushkp1cn_0_52orToDevice(uninitialisedOnly);
    pushkp2_0_52orToDevice(uninitialisedOnly);
    pushkp1cn_0_53orToDevice(uninitialisedOnly);
    pushkp2_0_53orToDevice(uninitialisedOnly);
    pushkp1cn_0_54orToDevice(uninitialisedOnly);
    pushkp2_0_54orToDevice(uninitialisedOnly);
    pushkp1cn_0_55orToDevice(uninitialisedOnly);
    pushkp2_0_55orToDevice(uninitialisedOnly);
    pushkp1cn_0_56orToDevice(uninitialisedOnly);
    pushkp2_0_56orToDevice(uninitialisedOnly);
    pushkp1cn_0_57orToDevice(uninitialisedOnly);
    pushkp2_0_57orToDevice(uninitialisedOnly);
    pushkp1cn_0_58orToDevice(uninitialisedOnly);
    pushkp2_0_58orToDevice(uninitialisedOnly);
    pushkp1cn_0_59orToDevice(uninitialisedOnly);
    pushkp2_0_59orToDevice(uninitialisedOnly);
    pushkp1cn_0_60orToDevice(uninitialisedOnly);
    pushkp2_0_60orToDevice(uninitialisedOnly);
    pushkp1cn_0_61orToDevice(uninitialisedOnly);
    pushkp2_0_61orToDevice(uninitialisedOnly);
    pushkp1cn_0_62orToDevice(uninitialisedOnly);
    pushkp2_0_62orToDevice(uninitialisedOnly);
    pushkp1cn_0_63orToDevice(uninitialisedOnly);
    pushkp2_0_63orToDevice(uninitialisedOnly);
    pushkp1cn_0_64orToDevice(uninitialisedOnly);
    pushkp2_0_64orToDevice(uninitialisedOnly);
    pushkp1cn_0_65orToDevice(uninitialisedOnly);
    pushkp2_0_65orToDevice(uninitialisedOnly);
    pushkp1cn_0_66orToDevice(uninitialisedOnly);
    pushkp2_0_66orToDevice(uninitialisedOnly);
    pushkp1cn_0_67orToDevice(uninitialisedOnly);
    pushkp2_0_67orToDevice(uninitialisedOnly);
    pushkp1cn_0_68orToDevice(uninitialisedOnly);
    pushkp2_0_68orToDevice(uninitialisedOnly);
    pushkp1cn_0_69orToDevice(uninitialisedOnly);
    pushkp2_0_69orToDevice(uninitialisedOnly);
    pushkp1cn_0_70orToDevice(uninitialisedOnly);
    pushkp2_0_70orToDevice(uninitialisedOnly);
    pushkp1cn_0_71orToDevice(uninitialisedOnly);
    pushkp2_0_71orToDevice(uninitialisedOnly);
    pushkp1cn_0_72orToDevice(uninitialisedOnly);
    pushkp2_0_72orToDevice(uninitialisedOnly);
    pushkp1cn_0_73orToDevice(uninitialisedOnly);
    pushkp2_0_73orToDevice(uninitialisedOnly);
    pushkp1cn_0_74orToDevice(uninitialisedOnly);
    pushkp2_0_74orToDevice(uninitialisedOnly);
    pushkp1cn_0_75orToDevice(uninitialisedOnly);
    pushkp2_0_75orToDevice(uninitialisedOnly);
    pushkp1cn_0_76orToDevice(uninitialisedOnly);
    pushkp2_0_76orToDevice(uninitialisedOnly);
    pushkp1cn_0_77orToDevice(uninitialisedOnly);
    pushkp2_0_77orToDevice(uninitialisedOnly);
    pushkp1cn_0_78orToDevice(uninitialisedOnly);
    pushkp2_0_78orToDevice(uninitialisedOnly);
    pushkp1cn_0_79orToDevice(uninitialisedOnly);
    pushkp2_0_79orToDevice(uninitialisedOnly);
    pushkp1cn_0_80orToDevice(uninitialisedOnly);
    pushkp2_0_80orToDevice(uninitialisedOnly);
    pushkp1cn_0_81orToDevice(uninitialisedOnly);
    pushkp2_0_81orToDevice(uninitialisedOnly);
    pushkp1cn_0_82orToDevice(uninitialisedOnly);
    pushkp2_0_82orToDevice(uninitialisedOnly);
    pushkp1cn_0_83orToDevice(uninitialisedOnly);
    pushkp2_0_83orToDevice(uninitialisedOnly);
    pushkp1cn_0_84orToDevice(uninitialisedOnly);
    pushkp2_0_84orToDevice(uninitialisedOnly);
    pushkp1cn_0_85orToDevice(uninitialisedOnly);
    pushkp2_0_85orToDevice(uninitialisedOnly);
    pushkp1cn_0_86orToDevice(uninitialisedOnly);
    pushkp2_0_86orToDevice(uninitialisedOnly);
    pushkp1cn_0_87orToDevice(uninitialisedOnly);
    pushkp2_0_87orToDevice(uninitialisedOnly);
    pushkp1cn_0_88orToDevice(uninitialisedOnly);
    pushkp2_0_88orToDevice(uninitialisedOnly);
    pushkp1cn_0_89orToDevice(uninitialisedOnly);
    pushkp2_0_89orToDevice(uninitialisedOnly);
    pushkp1cn_0_90orToDevice(uninitialisedOnly);
    pushkp2_0_90orToDevice(uninitialisedOnly);
    pushkp1cn_0_91orToDevice(uninitialisedOnly);
    pushkp2_0_91orToDevice(uninitialisedOnly);
    pushkp1cn_0_92orToDevice(uninitialisedOnly);
    pushkp2_0_92orToDevice(uninitialisedOnly);
    pushkp1cn_0_93orToDevice(uninitialisedOnly);
    pushkp2_0_93orToDevice(uninitialisedOnly);
    pushkp1cn_0_94orToDevice(uninitialisedOnly);
    pushkp2_0_94orToDevice(uninitialisedOnly);
    pushkp1cn_0_95orToDevice(uninitialisedOnly);
    pushkp2_0_95orToDevice(uninitialisedOnly);
    pushkp1cn_0_96orToDevice(uninitialisedOnly);
    pushkp2_0_96orToDevice(uninitialisedOnly);
    pushkp1cn_0_97orToDevice(uninitialisedOnly);
    pushkp2_0_97orToDevice(uninitialisedOnly);
    pushkp1cn_0_98orToDevice(uninitialisedOnly);
    pushkp2_0_98orToDevice(uninitialisedOnly);
    pushkp1cn_0_99orToDevice(uninitialisedOnly);
    pushkp2_0_99orToDevice(uninitialisedOnly);
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
        CHECK_CUDA_ERRORS(cudaMemcpy(d_indorn_ln, indorn_ln, 7680000 * sizeof(uint32_t), cudaMemcpyHostToDevice));
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
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gln_ln, gln_ln, 16000000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
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
    if(!uninitialisedOnly) {
        CHECK_CUDA_ERRORS(cudaMemcpy(d_gln_pn, gln_pn, 3200000 * sizeof(scalar), cudaMemcpyHostToDevice));
    }
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
        CHECK_CUDA_ERRORS(cudaMemcpy(d_inSynorn_ln, inSynorn_ln, 800 * sizeof(double), cudaMemcpyHostToDevice));
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

void pullkp1cn_0_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_0or, d_kp1cn_0_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_0or, d_kp1cn_0_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_0or, d_kp2_0_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_0orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_0or, d_kp2_0_0or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_1or, d_kp1cn_0_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_1or, d_kp1cn_0_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_1or, d_kp2_0_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_1orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_1or, d_kp2_0_1or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_2or, d_kp1cn_0_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_2or, d_kp1cn_0_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_2or, d_kp2_0_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_2orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_2or, d_kp2_0_2or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_3orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_3or, d_kp1cn_0_3or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_3orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_3or, d_kp1cn_0_3or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_3orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_3or, d_kp2_0_3or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_3orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_3or, d_kp2_0_3or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_4orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_4or, d_kp1cn_0_4or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_4orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_4or, d_kp1cn_0_4or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_4orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_4or, d_kp2_0_4or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_4orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_4or, d_kp2_0_4or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_5orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_5or, d_kp1cn_0_5or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_5orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_5or, d_kp1cn_0_5or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_5orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_5or, d_kp2_0_5or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_5orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_5or, d_kp2_0_5or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_6orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_6or, d_kp1cn_0_6or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_6orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_6or, d_kp1cn_0_6or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_6orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_6or, d_kp2_0_6or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_6orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_6or, d_kp2_0_6or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_7orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_7or, d_kp1cn_0_7or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_7orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_7or, d_kp1cn_0_7or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_7orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_7or, d_kp2_0_7or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_7orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_7or, d_kp2_0_7or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_8orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_8or, d_kp1cn_0_8or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_8orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_8or, d_kp1cn_0_8or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_8orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_8or, d_kp2_0_8or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_8orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_8or, d_kp2_0_8or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_9orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_9or, d_kp1cn_0_9or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_9orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_9or, d_kp1cn_0_9or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_9orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_9or, d_kp2_0_9or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_9orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_9or, d_kp2_0_9or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_10orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_10or, d_kp1cn_0_10or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_10orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_10or, d_kp1cn_0_10or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_10orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_10or, d_kp2_0_10or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_10orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_10or, d_kp2_0_10or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_11orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_11or, d_kp1cn_0_11or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_11orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_11or, d_kp1cn_0_11or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_11orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_11or, d_kp2_0_11or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_11orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_11or, d_kp2_0_11or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_12orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_12or, d_kp1cn_0_12or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_12orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_12or, d_kp1cn_0_12or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_12orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_12or, d_kp2_0_12or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_12orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_12or, d_kp2_0_12or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_13orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_13or, d_kp1cn_0_13or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_13orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_13or, d_kp1cn_0_13or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_13orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_13or, d_kp2_0_13or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_13orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_13or, d_kp2_0_13or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_14orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_14or, d_kp1cn_0_14or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_14orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_14or, d_kp1cn_0_14or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_14orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_14or, d_kp2_0_14or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_14orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_14or, d_kp2_0_14or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_15orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_15or, d_kp1cn_0_15or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_15orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_15or, d_kp1cn_0_15or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_15orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_15or, d_kp2_0_15or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_15orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_15or, d_kp2_0_15or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_16orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_16or, d_kp1cn_0_16or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_16orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_16or, d_kp1cn_0_16or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_16orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_16or, d_kp2_0_16or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_16orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_16or, d_kp2_0_16or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_17orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_17or, d_kp1cn_0_17or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_17orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_17or, d_kp1cn_0_17or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_17orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_17or, d_kp2_0_17or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_17orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_17or, d_kp2_0_17or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_18orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_18or, d_kp1cn_0_18or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_18orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_18or, d_kp1cn_0_18or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_18orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_18or, d_kp2_0_18or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_18orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_18or, d_kp2_0_18or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_19orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_19or, d_kp1cn_0_19or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_19orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_19or, d_kp1cn_0_19or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_19orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_19or, d_kp2_0_19or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_19orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_19or, d_kp2_0_19or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_20orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_20or, d_kp1cn_0_20or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_20orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_20or, d_kp1cn_0_20or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_20orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_20or, d_kp2_0_20or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_20orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_20or, d_kp2_0_20or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_21orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_21or, d_kp1cn_0_21or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_21orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_21or, d_kp1cn_0_21or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_21orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_21or, d_kp2_0_21or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_21orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_21or, d_kp2_0_21or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_22orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_22or, d_kp1cn_0_22or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_22orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_22or, d_kp1cn_0_22or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_22orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_22or, d_kp2_0_22or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_22orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_22or, d_kp2_0_22or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_23orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_23or, d_kp1cn_0_23or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_23orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_23or, d_kp1cn_0_23or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_23orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_23or, d_kp2_0_23or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_23orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_23or, d_kp2_0_23or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_24orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_24or, d_kp1cn_0_24or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_24orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_24or, d_kp1cn_0_24or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_24orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_24or, d_kp2_0_24or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_24orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_24or, d_kp2_0_24or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_25orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_25or, d_kp1cn_0_25or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_25orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_25or, d_kp1cn_0_25or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_25orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_25or, d_kp2_0_25or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_25orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_25or, d_kp2_0_25or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_26orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_26or, d_kp1cn_0_26or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_26orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_26or, d_kp1cn_0_26or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_26orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_26or, d_kp2_0_26or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_26orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_26or, d_kp2_0_26or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_27orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_27or, d_kp1cn_0_27or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_27orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_27or, d_kp1cn_0_27or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_27orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_27or, d_kp2_0_27or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_27orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_27or, d_kp2_0_27or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_28orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_28or, d_kp1cn_0_28or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_28orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_28or, d_kp1cn_0_28or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_28orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_28or, d_kp2_0_28or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_28orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_28or, d_kp2_0_28or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_29orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_29or, d_kp1cn_0_29or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_29orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_29or, d_kp1cn_0_29or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_29orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_29or, d_kp2_0_29or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_29orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_29or, d_kp2_0_29or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_30orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_30or, d_kp1cn_0_30or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_30orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_30or, d_kp1cn_0_30or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_30orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_30or, d_kp2_0_30or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_30orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_30or, d_kp2_0_30or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_31orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_31or, d_kp1cn_0_31or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_31orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_31or, d_kp1cn_0_31or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_31orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_31or, d_kp2_0_31or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_31orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_31or, d_kp2_0_31or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_32orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_32or, d_kp1cn_0_32or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_32orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_32or, d_kp1cn_0_32or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_32orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_32or, d_kp2_0_32or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_32orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_32or, d_kp2_0_32or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_33orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_33or, d_kp1cn_0_33or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_33orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_33or, d_kp1cn_0_33or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_33orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_33or, d_kp2_0_33or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_33orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_33or, d_kp2_0_33or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_34orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_34or, d_kp1cn_0_34or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_34orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_34or, d_kp1cn_0_34or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_34orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_34or, d_kp2_0_34or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_34orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_34or, d_kp2_0_34or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_35orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_35or, d_kp1cn_0_35or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_35orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_35or, d_kp1cn_0_35or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_35orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_35or, d_kp2_0_35or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_35orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_35or, d_kp2_0_35or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_36orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_36or, d_kp1cn_0_36or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_36orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_36or, d_kp1cn_0_36or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_36orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_36or, d_kp2_0_36or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_36orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_36or, d_kp2_0_36or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_37orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_37or, d_kp1cn_0_37or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_37orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_37or, d_kp1cn_0_37or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_37orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_37or, d_kp2_0_37or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_37orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_37or, d_kp2_0_37or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_38orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_38or, d_kp1cn_0_38or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_38orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_38or, d_kp1cn_0_38or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_38orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_38or, d_kp2_0_38or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_38orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_38or, d_kp2_0_38or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_39orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_39or, d_kp1cn_0_39or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_39orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_39or, d_kp1cn_0_39or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_39orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_39or, d_kp2_0_39or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_39orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_39or, d_kp2_0_39or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_40orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_40or, d_kp1cn_0_40or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_40orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_40or, d_kp1cn_0_40or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_40orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_40or, d_kp2_0_40or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_40orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_40or, d_kp2_0_40or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_41orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_41or, d_kp1cn_0_41or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_41orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_41or, d_kp1cn_0_41or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_41orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_41or, d_kp2_0_41or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_41orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_41or, d_kp2_0_41or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_42orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_42or, d_kp1cn_0_42or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_42orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_42or, d_kp1cn_0_42or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_42orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_42or, d_kp2_0_42or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_42orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_42or, d_kp2_0_42or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_43orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_43or, d_kp1cn_0_43or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_43orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_43or, d_kp1cn_0_43or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_43orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_43or, d_kp2_0_43or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_43orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_43or, d_kp2_0_43or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_44orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_44or, d_kp1cn_0_44or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_44orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_44or, d_kp1cn_0_44or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_44orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_44or, d_kp2_0_44or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_44orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_44or, d_kp2_0_44or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_45orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_45or, d_kp1cn_0_45or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_45orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_45or, d_kp1cn_0_45or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_45orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_45or, d_kp2_0_45or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_45orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_45or, d_kp2_0_45or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_46orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_46or, d_kp1cn_0_46or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_46orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_46or, d_kp1cn_0_46or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_46orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_46or, d_kp2_0_46or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_46orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_46or, d_kp2_0_46or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_47orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_47or, d_kp1cn_0_47or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_47orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_47or, d_kp1cn_0_47or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_47orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_47or, d_kp2_0_47or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_47orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_47or, d_kp2_0_47or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_48orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_48or, d_kp1cn_0_48or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_48orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_48or, d_kp1cn_0_48or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_48orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_48or, d_kp2_0_48or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_48orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_48or, d_kp2_0_48or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_49orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_49or, d_kp1cn_0_49or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_49orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_49or, d_kp1cn_0_49or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_49orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_49or, d_kp2_0_49or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_49orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_49or, d_kp2_0_49or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_50orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_50or, d_kp1cn_0_50or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_50orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_50or, d_kp1cn_0_50or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_50orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_50or, d_kp2_0_50or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_50orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_50or, d_kp2_0_50or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_51orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_51or, d_kp1cn_0_51or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_51orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_51or, d_kp1cn_0_51or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_51orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_51or, d_kp2_0_51or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_51orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_51or, d_kp2_0_51or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_52orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_52or, d_kp1cn_0_52or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_52orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_52or, d_kp1cn_0_52or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_52orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_52or, d_kp2_0_52or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_52orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_52or, d_kp2_0_52or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_53orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_53or, d_kp1cn_0_53or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_53orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_53or, d_kp1cn_0_53or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_53orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_53or, d_kp2_0_53or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_53orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_53or, d_kp2_0_53or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_54orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_54or, d_kp1cn_0_54or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_54orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_54or, d_kp1cn_0_54or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_54orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_54or, d_kp2_0_54or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_54orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_54or, d_kp2_0_54or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_55orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_55or, d_kp1cn_0_55or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_55orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_55or, d_kp1cn_0_55or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_55orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_55or, d_kp2_0_55or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_55orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_55or, d_kp2_0_55or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_56orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_56or, d_kp1cn_0_56or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_56orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_56or, d_kp1cn_0_56or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_56orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_56or, d_kp2_0_56or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_56orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_56or, d_kp2_0_56or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_57orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_57or, d_kp1cn_0_57or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_57orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_57or, d_kp1cn_0_57or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_57orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_57or, d_kp2_0_57or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_57orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_57or, d_kp2_0_57or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_58orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_58or, d_kp1cn_0_58or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_58orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_58or, d_kp1cn_0_58or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_58orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_58or, d_kp2_0_58or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_58orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_58or, d_kp2_0_58or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_59orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_59or, d_kp1cn_0_59or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_59orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_59or, d_kp1cn_0_59or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_59orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_59or, d_kp2_0_59or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_59orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_59or, d_kp2_0_59or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_60orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_60or, d_kp1cn_0_60or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_60orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_60or, d_kp1cn_0_60or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_60orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_60or, d_kp2_0_60or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_60orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_60or, d_kp2_0_60or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_61orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_61or, d_kp1cn_0_61or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_61orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_61or, d_kp1cn_0_61or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_61orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_61or, d_kp2_0_61or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_61orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_61or, d_kp2_0_61or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_62orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_62or, d_kp1cn_0_62or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_62orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_62or, d_kp1cn_0_62or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_62orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_62or, d_kp2_0_62or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_62orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_62or, d_kp2_0_62or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_63orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_63or, d_kp1cn_0_63or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_63orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_63or, d_kp1cn_0_63or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_63orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_63or, d_kp2_0_63or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_63orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_63or, d_kp2_0_63or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_64orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_64or, d_kp1cn_0_64or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_64orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_64or, d_kp1cn_0_64or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_64orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_64or, d_kp2_0_64or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_64orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_64or, d_kp2_0_64or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_65orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_65or, d_kp1cn_0_65or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_65orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_65or, d_kp1cn_0_65or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_65orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_65or, d_kp2_0_65or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_65orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_65or, d_kp2_0_65or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_66orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_66or, d_kp1cn_0_66or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_66orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_66or, d_kp1cn_0_66or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_66orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_66or, d_kp2_0_66or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_66orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_66or, d_kp2_0_66or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_67orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_67or, d_kp1cn_0_67or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_67orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_67or, d_kp1cn_0_67or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_67orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_67or, d_kp2_0_67or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_67orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_67or, d_kp2_0_67or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_68orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_68or, d_kp1cn_0_68or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_68orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_68or, d_kp1cn_0_68or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_68orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_68or, d_kp2_0_68or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_68orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_68or, d_kp2_0_68or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_69orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_69or, d_kp1cn_0_69or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_69orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_69or, d_kp1cn_0_69or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_69orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_69or, d_kp2_0_69or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_69orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_69or, d_kp2_0_69or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_70orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_70or, d_kp1cn_0_70or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_70orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_70or, d_kp1cn_0_70or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_70orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_70or, d_kp2_0_70or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_70orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_70or, d_kp2_0_70or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_71orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_71or, d_kp1cn_0_71or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_71orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_71or, d_kp1cn_0_71or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_71orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_71or, d_kp2_0_71or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_71orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_71or, d_kp2_0_71or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_72orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_72or, d_kp1cn_0_72or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_72orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_72or, d_kp1cn_0_72or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_72orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_72or, d_kp2_0_72or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_72orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_72or, d_kp2_0_72or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_73orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_73or, d_kp1cn_0_73or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_73orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_73or, d_kp1cn_0_73or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_73orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_73or, d_kp2_0_73or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_73orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_73or, d_kp2_0_73or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_74orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_74or, d_kp1cn_0_74or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_74orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_74or, d_kp1cn_0_74or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_74orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_74or, d_kp2_0_74or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_74orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_74or, d_kp2_0_74or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_75orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_75or, d_kp1cn_0_75or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_75orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_75or, d_kp1cn_0_75or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_75orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_75or, d_kp2_0_75or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_75orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_75or, d_kp2_0_75or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_76orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_76or, d_kp1cn_0_76or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_76orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_76or, d_kp1cn_0_76or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_76orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_76or, d_kp2_0_76or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_76orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_76or, d_kp2_0_76or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_77orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_77or, d_kp1cn_0_77or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_77orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_77or, d_kp1cn_0_77or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_77orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_77or, d_kp2_0_77or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_77orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_77or, d_kp2_0_77or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_78orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_78or, d_kp1cn_0_78or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_78orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_78or, d_kp1cn_0_78or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_78orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_78or, d_kp2_0_78or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_78orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_78or, d_kp2_0_78or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_79orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_79or, d_kp1cn_0_79or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_79orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_79or, d_kp1cn_0_79or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_79orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_79or, d_kp2_0_79or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_79orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_79or, d_kp2_0_79or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_80orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_80or, d_kp1cn_0_80or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_80orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_80or, d_kp1cn_0_80or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_80orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_80or, d_kp2_0_80or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_80orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_80or, d_kp2_0_80or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_81orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_81or, d_kp1cn_0_81or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_81orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_81or, d_kp1cn_0_81or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_81orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_81or, d_kp2_0_81or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_81orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_81or, d_kp2_0_81or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_82orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_82or, d_kp1cn_0_82or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_82orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_82or, d_kp1cn_0_82or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_82orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_82or, d_kp2_0_82or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_82orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_82or, d_kp2_0_82or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_83orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_83or, d_kp1cn_0_83or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_83orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_83or, d_kp1cn_0_83or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_83orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_83or, d_kp2_0_83or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_83orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_83or, d_kp2_0_83or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_84orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_84or, d_kp1cn_0_84or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_84orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_84or, d_kp1cn_0_84or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_84orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_84or, d_kp2_0_84or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_84orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_84or, d_kp2_0_84or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_85orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_85or, d_kp1cn_0_85or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_85orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_85or, d_kp1cn_0_85or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_85orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_85or, d_kp2_0_85or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_85orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_85or, d_kp2_0_85or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_86orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_86or, d_kp1cn_0_86or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_86orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_86or, d_kp1cn_0_86or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_86orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_86or, d_kp2_0_86or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_86orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_86or, d_kp2_0_86or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_87orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_87or, d_kp1cn_0_87or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_87orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_87or, d_kp1cn_0_87or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_87orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_87or, d_kp2_0_87or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_87orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_87or, d_kp2_0_87or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_88orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_88or, d_kp1cn_0_88or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_88orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_88or, d_kp1cn_0_88or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_88orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_88or, d_kp2_0_88or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_88orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_88or, d_kp2_0_88or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_89orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_89or, d_kp1cn_0_89or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_89orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_89or, d_kp1cn_0_89or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_89orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_89or, d_kp2_0_89or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_89orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_89or, d_kp2_0_89or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_90orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_90or, d_kp1cn_0_90or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_90orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_90or, d_kp1cn_0_90or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_90orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_90or, d_kp2_0_90or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_90orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_90or, d_kp2_0_90or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_91orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_91or, d_kp1cn_0_91or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_91orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_91or, d_kp1cn_0_91or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_91orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_91or, d_kp2_0_91or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_91orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_91or, d_kp2_0_91or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_92orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_92or, d_kp1cn_0_92or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_92orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_92or, d_kp1cn_0_92or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_92orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_92or, d_kp2_0_92or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_92orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_92or, d_kp2_0_92or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_93orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_93or, d_kp1cn_0_93or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_93orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_93or, d_kp1cn_0_93or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_93orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_93or, d_kp2_0_93or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_93orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_93or, d_kp2_0_93or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_94orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_94or, d_kp1cn_0_94or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_94orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_94or, d_kp1cn_0_94or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_94orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_94or, d_kp2_0_94or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_94orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_94or, d_kp2_0_94or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_95orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_95or, d_kp1cn_0_95or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_95orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_95or, d_kp1cn_0_95or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_95orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_95or, d_kp2_0_95or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_95orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_95or, d_kp2_0_95or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_96orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_96or, d_kp1cn_0_96or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_96orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_96or, d_kp1cn_0_96or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_96orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_96or, d_kp2_0_96or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_96orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_96or, d_kp2_0_96or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_97orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_97or, d_kp1cn_0_97or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_97orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_97or, d_kp1cn_0_97or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_97orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_97or, d_kp2_0_97or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_97orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_97or, d_kp2_0_97or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_98orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_98or, d_kp1cn_0_98or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_98orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_98or, d_kp1cn_0_98or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_98orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_98or, d_kp2_0_98or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_98orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_98or, d_kp2_0_98or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp1cn_0_99orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_99or, d_kp1cn_0_99or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp1cn_0_99orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp1cn_0_99or, d_kp1cn_0_99or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullkp2_0_99orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_99or, d_kp2_0_99or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
}

void pullCurrentkp2_0_99orFromDevice() {
    CHECK_CUDA_ERRORS(cudaMemcpy(kp2_0_99or, d_kp2_0_99or, 160 * sizeof(scalar), cudaMemcpyDeviceToHost));
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
    pullkp1cn_0_0orFromDevice();
    pullkp2_0_0orFromDevice();
    pullkp1cn_0_1orFromDevice();
    pullkp2_0_1orFromDevice();
    pullkp1cn_0_2orFromDevice();
    pullkp2_0_2orFromDevice();
    pullkp1cn_0_3orFromDevice();
    pullkp2_0_3orFromDevice();
    pullkp1cn_0_4orFromDevice();
    pullkp2_0_4orFromDevice();
    pullkp1cn_0_5orFromDevice();
    pullkp2_0_5orFromDevice();
    pullkp1cn_0_6orFromDevice();
    pullkp2_0_6orFromDevice();
    pullkp1cn_0_7orFromDevice();
    pullkp2_0_7orFromDevice();
    pullkp1cn_0_8orFromDevice();
    pullkp2_0_8orFromDevice();
    pullkp1cn_0_9orFromDevice();
    pullkp2_0_9orFromDevice();
    pullkp1cn_0_10orFromDevice();
    pullkp2_0_10orFromDevice();
    pullkp1cn_0_11orFromDevice();
    pullkp2_0_11orFromDevice();
    pullkp1cn_0_12orFromDevice();
    pullkp2_0_12orFromDevice();
    pullkp1cn_0_13orFromDevice();
    pullkp2_0_13orFromDevice();
    pullkp1cn_0_14orFromDevice();
    pullkp2_0_14orFromDevice();
    pullkp1cn_0_15orFromDevice();
    pullkp2_0_15orFromDevice();
    pullkp1cn_0_16orFromDevice();
    pullkp2_0_16orFromDevice();
    pullkp1cn_0_17orFromDevice();
    pullkp2_0_17orFromDevice();
    pullkp1cn_0_18orFromDevice();
    pullkp2_0_18orFromDevice();
    pullkp1cn_0_19orFromDevice();
    pullkp2_0_19orFromDevice();
    pullkp1cn_0_20orFromDevice();
    pullkp2_0_20orFromDevice();
    pullkp1cn_0_21orFromDevice();
    pullkp2_0_21orFromDevice();
    pullkp1cn_0_22orFromDevice();
    pullkp2_0_22orFromDevice();
    pullkp1cn_0_23orFromDevice();
    pullkp2_0_23orFromDevice();
    pullkp1cn_0_24orFromDevice();
    pullkp2_0_24orFromDevice();
    pullkp1cn_0_25orFromDevice();
    pullkp2_0_25orFromDevice();
    pullkp1cn_0_26orFromDevice();
    pullkp2_0_26orFromDevice();
    pullkp1cn_0_27orFromDevice();
    pullkp2_0_27orFromDevice();
    pullkp1cn_0_28orFromDevice();
    pullkp2_0_28orFromDevice();
    pullkp1cn_0_29orFromDevice();
    pullkp2_0_29orFromDevice();
    pullkp1cn_0_30orFromDevice();
    pullkp2_0_30orFromDevice();
    pullkp1cn_0_31orFromDevice();
    pullkp2_0_31orFromDevice();
    pullkp1cn_0_32orFromDevice();
    pullkp2_0_32orFromDevice();
    pullkp1cn_0_33orFromDevice();
    pullkp2_0_33orFromDevice();
    pullkp1cn_0_34orFromDevice();
    pullkp2_0_34orFromDevice();
    pullkp1cn_0_35orFromDevice();
    pullkp2_0_35orFromDevice();
    pullkp1cn_0_36orFromDevice();
    pullkp2_0_36orFromDevice();
    pullkp1cn_0_37orFromDevice();
    pullkp2_0_37orFromDevice();
    pullkp1cn_0_38orFromDevice();
    pullkp2_0_38orFromDevice();
    pullkp1cn_0_39orFromDevice();
    pullkp2_0_39orFromDevice();
    pullkp1cn_0_40orFromDevice();
    pullkp2_0_40orFromDevice();
    pullkp1cn_0_41orFromDevice();
    pullkp2_0_41orFromDevice();
    pullkp1cn_0_42orFromDevice();
    pullkp2_0_42orFromDevice();
    pullkp1cn_0_43orFromDevice();
    pullkp2_0_43orFromDevice();
    pullkp1cn_0_44orFromDevice();
    pullkp2_0_44orFromDevice();
    pullkp1cn_0_45orFromDevice();
    pullkp2_0_45orFromDevice();
    pullkp1cn_0_46orFromDevice();
    pullkp2_0_46orFromDevice();
    pullkp1cn_0_47orFromDevice();
    pullkp2_0_47orFromDevice();
    pullkp1cn_0_48orFromDevice();
    pullkp2_0_48orFromDevice();
    pullkp1cn_0_49orFromDevice();
    pullkp2_0_49orFromDevice();
    pullkp1cn_0_50orFromDevice();
    pullkp2_0_50orFromDevice();
    pullkp1cn_0_51orFromDevice();
    pullkp2_0_51orFromDevice();
    pullkp1cn_0_52orFromDevice();
    pullkp2_0_52orFromDevice();
    pullkp1cn_0_53orFromDevice();
    pullkp2_0_53orFromDevice();
    pullkp1cn_0_54orFromDevice();
    pullkp2_0_54orFromDevice();
    pullkp1cn_0_55orFromDevice();
    pullkp2_0_55orFromDevice();
    pullkp1cn_0_56orFromDevice();
    pullkp2_0_56orFromDevice();
    pullkp1cn_0_57orFromDevice();
    pullkp2_0_57orFromDevice();
    pullkp1cn_0_58orFromDevice();
    pullkp2_0_58orFromDevice();
    pullkp1cn_0_59orFromDevice();
    pullkp2_0_59orFromDevice();
    pullkp1cn_0_60orFromDevice();
    pullkp2_0_60orFromDevice();
    pullkp1cn_0_61orFromDevice();
    pullkp2_0_61orFromDevice();
    pullkp1cn_0_62orFromDevice();
    pullkp2_0_62orFromDevice();
    pullkp1cn_0_63orFromDevice();
    pullkp2_0_63orFromDevice();
    pullkp1cn_0_64orFromDevice();
    pullkp2_0_64orFromDevice();
    pullkp1cn_0_65orFromDevice();
    pullkp2_0_65orFromDevice();
    pullkp1cn_0_66orFromDevice();
    pullkp2_0_66orFromDevice();
    pullkp1cn_0_67orFromDevice();
    pullkp2_0_67orFromDevice();
    pullkp1cn_0_68orFromDevice();
    pullkp2_0_68orFromDevice();
    pullkp1cn_0_69orFromDevice();
    pullkp2_0_69orFromDevice();
    pullkp1cn_0_70orFromDevice();
    pullkp2_0_70orFromDevice();
    pullkp1cn_0_71orFromDevice();
    pullkp2_0_71orFromDevice();
    pullkp1cn_0_72orFromDevice();
    pullkp2_0_72orFromDevice();
    pullkp1cn_0_73orFromDevice();
    pullkp2_0_73orFromDevice();
    pullkp1cn_0_74orFromDevice();
    pullkp2_0_74orFromDevice();
    pullkp1cn_0_75orFromDevice();
    pullkp2_0_75orFromDevice();
    pullkp1cn_0_76orFromDevice();
    pullkp2_0_76orFromDevice();
    pullkp1cn_0_77orFromDevice();
    pullkp2_0_77orFromDevice();
    pullkp1cn_0_78orFromDevice();
    pullkp2_0_78orFromDevice();
    pullkp1cn_0_79orFromDevice();
    pullkp2_0_79orFromDevice();
    pullkp1cn_0_80orFromDevice();
    pullkp2_0_80orFromDevice();
    pullkp1cn_0_81orFromDevice();
    pullkp2_0_81orFromDevice();
    pullkp1cn_0_82orFromDevice();
    pullkp2_0_82orFromDevice();
    pullkp1cn_0_83orFromDevice();
    pullkp2_0_83orFromDevice();
    pullkp1cn_0_84orFromDevice();
    pullkp2_0_84orFromDevice();
    pullkp1cn_0_85orFromDevice();
    pullkp2_0_85orFromDevice();
    pullkp1cn_0_86orFromDevice();
    pullkp2_0_86orFromDevice();
    pullkp1cn_0_87orFromDevice();
    pullkp2_0_87orFromDevice();
    pullkp1cn_0_88orFromDevice();
    pullkp2_0_88orFromDevice();
    pullkp1cn_0_89orFromDevice();
    pullkp2_0_89orFromDevice();
    pullkp1cn_0_90orFromDevice();
    pullkp2_0_90orFromDevice();
    pullkp1cn_0_91orFromDevice();
    pullkp2_0_91orFromDevice();
    pullkp1cn_0_92orFromDevice();
    pullkp2_0_92orFromDevice();
    pullkp1cn_0_93orFromDevice();
    pullkp2_0_93orFromDevice();
    pullkp1cn_0_94orFromDevice();
    pullkp2_0_94orFromDevice();
    pullkp1cn_0_95orFromDevice();
    pullkp2_0_95orFromDevice();
    pullkp1cn_0_96orFromDevice();
    pullkp2_0_96orFromDevice();
    pullkp1cn_0_97orFromDevice();
    pullkp2_0_97orFromDevice();
    pullkp1cn_0_98orFromDevice();
    pullkp2_0_98orFromDevice();
    pullkp1cn_0_99orFromDevice();
    pullkp2_0_99orFromDevice();
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
    CHECK_CUDA_ERRORS(cudaMemcpy(indorn_ln, d_indorn_ln, 7680000 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
    CHECK_CUDA_ERRORS(cudaMemcpy(inSynorn_ln, d_inSynorn_ln, 800 * sizeof(double), cudaMemcpyDeviceToHost));
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

scalar* getCurrentkp1cn_0_0or(unsigned int batch) {
    return kp1cn_0_0or;
}

scalar* getCurrentkp2_0_0or(unsigned int batch) {
    return kp2_0_0or;
}

scalar* getCurrentkp1cn_0_1or(unsigned int batch) {
    return kp1cn_0_1or;
}

scalar* getCurrentkp2_0_1or(unsigned int batch) {
    return kp2_0_1or;
}

scalar* getCurrentkp1cn_0_2or(unsigned int batch) {
    return kp1cn_0_2or;
}

scalar* getCurrentkp2_0_2or(unsigned int batch) {
    return kp2_0_2or;
}

scalar* getCurrentkp1cn_0_3or(unsigned int batch) {
    return kp1cn_0_3or;
}

scalar* getCurrentkp2_0_3or(unsigned int batch) {
    return kp2_0_3or;
}

scalar* getCurrentkp1cn_0_4or(unsigned int batch) {
    return kp1cn_0_4or;
}

scalar* getCurrentkp2_0_4or(unsigned int batch) {
    return kp2_0_4or;
}

scalar* getCurrentkp1cn_0_5or(unsigned int batch) {
    return kp1cn_0_5or;
}

scalar* getCurrentkp2_0_5or(unsigned int batch) {
    return kp2_0_5or;
}

scalar* getCurrentkp1cn_0_6or(unsigned int batch) {
    return kp1cn_0_6or;
}

scalar* getCurrentkp2_0_6or(unsigned int batch) {
    return kp2_0_6or;
}

scalar* getCurrentkp1cn_0_7or(unsigned int batch) {
    return kp1cn_0_7or;
}

scalar* getCurrentkp2_0_7or(unsigned int batch) {
    return kp2_0_7or;
}

scalar* getCurrentkp1cn_0_8or(unsigned int batch) {
    return kp1cn_0_8or;
}

scalar* getCurrentkp2_0_8or(unsigned int batch) {
    return kp2_0_8or;
}

scalar* getCurrentkp1cn_0_9or(unsigned int batch) {
    return kp1cn_0_9or;
}

scalar* getCurrentkp2_0_9or(unsigned int batch) {
    return kp2_0_9or;
}

scalar* getCurrentkp1cn_0_10or(unsigned int batch) {
    return kp1cn_0_10or;
}

scalar* getCurrentkp2_0_10or(unsigned int batch) {
    return kp2_0_10or;
}

scalar* getCurrentkp1cn_0_11or(unsigned int batch) {
    return kp1cn_0_11or;
}

scalar* getCurrentkp2_0_11or(unsigned int batch) {
    return kp2_0_11or;
}

scalar* getCurrentkp1cn_0_12or(unsigned int batch) {
    return kp1cn_0_12or;
}

scalar* getCurrentkp2_0_12or(unsigned int batch) {
    return kp2_0_12or;
}

scalar* getCurrentkp1cn_0_13or(unsigned int batch) {
    return kp1cn_0_13or;
}

scalar* getCurrentkp2_0_13or(unsigned int batch) {
    return kp2_0_13or;
}

scalar* getCurrentkp1cn_0_14or(unsigned int batch) {
    return kp1cn_0_14or;
}

scalar* getCurrentkp2_0_14or(unsigned int batch) {
    return kp2_0_14or;
}

scalar* getCurrentkp1cn_0_15or(unsigned int batch) {
    return kp1cn_0_15or;
}

scalar* getCurrentkp2_0_15or(unsigned int batch) {
    return kp2_0_15or;
}

scalar* getCurrentkp1cn_0_16or(unsigned int batch) {
    return kp1cn_0_16or;
}

scalar* getCurrentkp2_0_16or(unsigned int batch) {
    return kp2_0_16or;
}

scalar* getCurrentkp1cn_0_17or(unsigned int batch) {
    return kp1cn_0_17or;
}

scalar* getCurrentkp2_0_17or(unsigned int batch) {
    return kp2_0_17or;
}

scalar* getCurrentkp1cn_0_18or(unsigned int batch) {
    return kp1cn_0_18or;
}

scalar* getCurrentkp2_0_18or(unsigned int batch) {
    return kp2_0_18or;
}

scalar* getCurrentkp1cn_0_19or(unsigned int batch) {
    return kp1cn_0_19or;
}

scalar* getCurrentkp2_0_19or(unsigned int batch) {
    return kp2_0_19or;
}

scalar* getCurrentkp1cn_0_20or(unsigned int batch) {
    return kp1cn_0_20or;
}

scalar* getCurrentkp2_0_20or(unsigned int batch) {
    return kp2_0_20or;
}

scalar* getCurrentkp1cn_0_21or(unsigned int batch) {
    return kp1cn_0_21or;
}

scalar* getCurrentkp2_0_21or(unsigned int batch) {
    return kp2_0_21or;
}

scalar* getCurrentkp1cn_0_22or(unsigned int batch) {
    return kp1cn_0_22or;
}

scalar* getCurrentkp2_0_22or(unsigned int batch) {
    return kp2_0_22or;
}

scalar* getCurrentkp1cn_0_23or(unsigned int batch) {
    return kp1cn_0_23or;
}

scalar* getCurrentkp2_0_23or(unsigned int batch) {
    return kp2_0_23or;
}

scalar* getCurrentkp1cn_0_24or(unsigned int batch) {
    return kp1cn_0_24or;
}

scalar* getCurrentkp2_0_24or(unsigned int batch) {
    return kp2_0_24or;
}

scalar* getCurrentkp1cn_0_25or(unsigned int batch) {
    return kp1cn_0_25or;
}

scalar* getCurrentkp2_0_25or(unsigned int batch) {
    return kp2_0_25or;
}

scalar* getCurrentkp1cn_0_26or(unsigned int batch) {
    return kp1cn_0_26or;
}

scalar* getCurrentkp2_0_26or(unsigned int batch) {
    return kp2_0_26or;
}

scalar* getCurrentkp1cn_0_27or(unsigned int batch) {
    return kp1cn_0_27or;
}

scalar* getCurrentkp2_0_27or(unsigned int batch) {
    return kp2_0_27or;
}

scalar* getCurrentkp1cn_0_28or(unsigned int batch) {
    return kp1cn_0_28or;
}

scalar* getCurrentkp2_0_28or(unsigned int batch) {
    return kp2_0_28or;
}

scalar* getCurrentkp1cn_0_29or(unsigned int batch) {
    return kp1cn_0_29or;
}

scalar* getCurrentkp2_0_29or(unsigned int batch) {
    return kp2_0_29or;
}

scalar* getCurrentkp1cn_0_30or(unsigned int batch) {
    return kp1cn_0_30or;
}

scalar* getCurrentkp2_0_30or(unsigned int batch) {
    return kp2_0_30or;
}

scalar* getCurrentkp1cn_0_31or(unsigned int batch) {
    return kp1cn_0_31or;
}

scalar* getCurrentkp2_0_31or(unsigned int batch) {
    return kp2_0_31or;
}

scalar* getCurrentkp1cn_0_32or(unsigned int batch) {
    return kp1cn_0_32or;
}

scalar* getCurrentkp2_0_32or(unsigned int batch) {
    return kp2_0_32or;
}

scalar* getCurrentkp1cn_0_33or(unsigned int batch) {
    return kp1cn_0_33or;
}

scalar* getCurrentkp2_0_33or(unsigned int batch) {
    return kp2_0_33or;
}

scalar* getCurrentkp1cn_0_34or(unsigned int batch) {
    return kp1cn_0_34or;
}

scalar* getCurrentkp2_0_34or(unsigned int batch) {
    return kp2_0_34or;
}

scalar* getCurrentkp1cn_0_35or(unsigned int batch) {
    return kp1cn_0_35or;
}

scalar* getCurrentkp2_0_35or(unsigned int batch) {
    return kp2_0_35or;
}

scalar* getCurrentkp1cn_0_36or(unsigned int batch) {
    return kp1cn_0_36or;
}

scalar* getCurrentkp2_0_36or(unsigned int batch) {
    return kp2_0_36or;
}

scalar* getCurrentkp1cn_0_37or(unsigned int batch) {
    return kp1cn_0_37or;
}

scalar* getCurrentkp2_0_37or(unsigned int batch) {
    return kp2_0_37or;
}

scalar* getCurrentkp1cn_0_38or(unsigned int batch) {
    return kp1cn_0_38or;
}

scalar* getCurrentkp2_0_38or(unsigned int batch) {
    return kp2_0_38or;
}

scalar* getCurrentkp1cn_0_39or(unsigned int batch) {
    return kp1cn_0_39or;
}

scalar* getCurrentkp2_0_39or(unsigned int batch) {
    return kp2_0_39or;
}

scalar* getCurrentkp1cn_0_40or(unsigned int batch) {
    return kp1cn_0_40or;
}

scalar* getCurrentkp2_0_40or(unsigned int batch) {
    return kp2_0_40or;
}

scalar* getCurrentkp1cn_0_41or(unsigned int batch) {
    return kp1cn_0_41or;
}

scalar* getCurrentkp2_0_41or(unsigned int batch) {
    return kp2_0_41or;
}

scalar* getCurrentkp1cn_0_42or(unsigned int batch) {
    return kp1cn_0_42or;
}

scalar* getCurrentkp2_0_42or(unsigned int batch) {
    return kp2_0_42or;
}

scalar* getCurrentkp1cn_0_43or(unsigned int batch) {
    return kp1cn_0_43or;
}

scalar* getCurrentkp2_0_43or(unsigned int batch) {
    return kp2_0_43or;
}

scalar* getCurrentkp1cn_0_44or(unsigned int batch) {
    return kp1cn_0_44or;
}

scalar* getCurrentkp2_0_44or(unsigned int batch) {
    return kp2_0_44or;
}

scalar* getCurrentkp1cn_0_45or(unsigned int batch) {
    return kp1cn_0_45or;
}

scalar* getCurrentkp2_0_45or(unsigned int batch) {
    return kp2_0_45or;
}

scalar* getCurrentkp1cn_0_46or(unsigned int batch) {
    return kp1cn_0_46or;
}

scalar* getCurrentkp2_0_46or(unsigned int batch) {
    return kp2_0_46or;
}

scalar* getCurrentkp1cn_0_47or(unsigned int batch) {
    return kp1cn_0_47or;
}

scalar* getCurrentkp2_0_47or(unsigned int batch) {
    return kp2_0_47or;
}

scalar* getCurrentkp1cn_0_48or(unsigned int batch) {
    return kp1cn_0_48or;
}

scalar* getCurrentkp2_0_48or(unsigned int batch) {
    return kp2_0_48or;
}

scalar* getCurrentkp1cn_0_49or(unsigned int batch) {
    return kp1cn_0_49or;
}

scalar* getCurrentkp2_0_49or(unsigned int batch) {
    return kp2_0_49or;
}

scalar* getCurrentkp1cn_0_50or(unsigned int batch) {
    return kp1cn_0_50or;
}

scalar* getCurrentkp2_0_50or(unsigned int batch) {
    return kp2_0_50or;
}

scalar* getCurrentkp1cn_0_51or(unsigned int batch) {
    return kp1cn_0_51or;
}

scalar* getCurrentkp2_0_51or(unsigned int batch) {
    return kp2_0_51or;
}

scalar* getCurrentkp1cn_0_52or(unsigned int batch) {
    return kp1cn_0_52or;
}

scalar* getCurrentkp2_0_52or(unsigned int batch) {
    return kp2_0_52or;
}

scalar* getCurrentkp1cn_0_53or(unsigned int batch) {
    return kp1cn_0_53or;
}

scalar* getCurrentkp2_0_53or(unsigned int batch) {
    return kp2_0_53or;
}

scalar* getCurrentkp1cn_0_54or(unsigned int batch) {
    return kp1cn_0_54or;
}

scalar* getCurrentkp2_0_54or(unsigned int batch) {
    return kp2_0_54or;
}

scalar* getCurrentkp1cn_0_55or(unsigned int batch) {
    return kp1cn_0_55or;
}

scalar* getCurrentkp2_0_55or(unsigned int batch) {
    return kp2_0_55or;
}

scalar* getCurrentkp1cn_0_56or(unsigned int batch) {
    return kp1cn_0_56or;
}

scalar* getCurrentkp2_0_56or(unsigned int batch) {
    return kp2_0_56or;
}

scalar* getCurrentkp1cn_0_57or(unsigned int batch) {
    return kp1cn_0_57or;
}

scalar* getCurrentkp2_0_57or(unsigned int batch) {
    return kp2_0_57or;
}

scalar* getCurrentkp1cn_0_58or(unsigned int batch) {
    return kp1cn_0_58or;
}

scalar* getCurrentkp2_0_58or(unsigned int batch) {
    return kp2_0_58or;
}

scalar* getCurrentkp1cn_0_59or(unsigned int batch) {
    return kp1cn_0_59or;
}

scalar* getCurrentkp2_0_59or(unsigned int batch) {
    return kp2_0_59or;
}

scalar* getCurrentkp1cn_0_60or(unsigned int batch) {
    return kp1cn_0_60or;
}

scalar* getCurrentkp2_0_60or(unsigned int batch) {
    return kp2_0_60or;
}

scalar* getCurrentkp1cn_0_61or(unsigned int batch) {
    return kp1cn_0_61or;
}

scalar* getCurrentkp2_0_61or(unsigned int batch) {
    return kp2_0_61or;
}

scalar* getCurrentkp1cn_0_62or(unsigned int batch) {
    return kp1cn_0_62or;
}

scalar* getCurrentkp2_0_62or(unsigned int batch) {
    return kp2_0_62or;
}

scalar* getCurrentkp1cn_0_63or(unsigned int batch) {
    return kp1cn_0_63or;
}

scalar* getCurrentkp2_0_63or(unsigned int batch) {
    return kp2_0_63or;
}

scalar* getCurrentkp1cn_0_64or(unsigned int batch) {
    return kp1cn_0_64or;
}

scalar* getCurrentkp2_0_64or(unsigned int batch) {
    return kp2_0_64or;
}

scalar* getCurrentkp1cn_0_65or(unsigned int batch) {
    return kp1cn_0_65or;
}

scalar* getCurrentkp2_0_65or(unsigned int batch) {
    return kp2_0_65or;
}

scalar* getCurrentkp1cn_0_66or(unsigned int batch) {
    return kp1cn_0_66or;
}

scalar* getCurrentkp2_0_66or(unsigned int batch) {
    return kp2_0_66or;
}

scalar* getCurrentkp1cn_0_67or(unsigned int batch) {
    return kp1cn_0_67or;
}

scalar* getCurrentkp2_0_67or(unsigned int batch) {
    return kp2_0_67or;
}

scalar* getCurrentkp1cn_0_68or(unsigned int batch) {
    return kp1cn_0_68or;
}

scalar* getCurrentkp2_0_68or(unsigned int batch) {
    return kp2_0_68or;
}

scalar* getCurrentkp1cn_0_69or(unsigned int batch) {
    return kp1cn_0_69or;
}

scalar* getCurrentkp2_0_69or(unsigned int batch) {
    return kp2_0_69or;
}

scalar* getCurrentkp1cn_0_70or(unsigned int batch) {
    return kp1cn_0_70or;
}

scalar* getCurrentkp2_0_70or(unsigned int batch) {
    return kp2_0_70or;
}

scalar* getCurrentkp1cn_0_71or(unsigned int batch) {
    return kp1cn_0_71or;
}

scalar* getCurrentkp2_0_71or(unsigned int batch) {
    return kp2_0_71or;
}

scalar* getCurrentkp1cn_0_72or(unsigned int batch) {
    return kp1cn_0_72or;
}

scalar* getCurrentkp2_0_72or(unsigned int batch) {
    return kp2_0_72or;
}

scalar* getCurrentkp1cn_0_73or(unsigned int batch) {
    return kp1cn_0_73or;
}

scalar* getCurrentkp2_0_73or(unsigned int batch) {
    return kp2_0_73or;
}

scalar* getCurrentkp1cn_0_74or(unsigned int batch) {
    return kp1cn_0_74or;
}

scalar* getCurrentkp2_0_74or(unsigned int batch) {
    return kp2_0_74or;
}

scalar* getCurrentkp1cn_0_75or(unsigned int batch) {
    return kp1cn_0_75or;
}

scalar* getCurrentkp2_0_75or(unsigned int batch) {
    return kp2_0_75or;
}

scalar* getCurrentkp1cn_0_76or(unsigned int batch) {
    return kp1cn_0_76or;
}

scalar* getCurrentkp2_0_76or(unsigned int batch) {
    return kp2_0_76or;
}

scalar* getCurrentkp1cn_0_77or(unsigned int batch) {
    return kp1cn_0_77or;
}

scalar* getCurrentkp2_0_77or(unsigned int batch) {
    return kp2_0_77or;
}

scalar* getCurrentkp1cn_0_78or(unsigned int batch) {
    return kp1cn_0_78or;
}

scalar* getCurrentkp2_0_78or(unsigned int batch) {
    return kp2_0_78or;
}

scalar* getCurrentkp1cn_0_79or(unsigned int batch) {
    return kp1cn_0_79or;
}

scalar* getCurrentkp2_0_79or(unsigned int batch) {
    return kp2_0_79or;
}

scalar* getCurrentkp1cn_0_80or(unsigned int batch) {
    return kp1cn_0_80or;
}

scalar* getCurrentkp2_0_80or(unsigned int batch) {
    return kp2_0_80or;
}

scalar* getCurrentkp1cn_0_81or(unsigned int batch) {
    return kp1cn_0_81or;
}

scalar* getCurrentkp2_0_81or(unsigned int batch) {
    return kp2_0_81or;
}

scalar* getCurrentkp1cn_0_82or(unsigned int batch) {
    return kp1cn_0_82or;
}

scalar* getCurrentkp2_0_82or(unsigned int batch) {
    return kp2_0_82or;
}

scalar* getCurrentkp1cn_0_83or(unsigned int batch) {
    return kp1cn_0_83or;
}

scalar* getCurrentkp2_0_83or(unsigned int batch) {
    return kp2_0_83or;
}

scalar* getCurrentkp1cn_0_84or(unsigned int batch) {
    return kp1cn_0_84or;
}

scalar* getCurrentkp2_0_84or(unsigned int batch) {
    return kp2_0_84or;
}

scalar* getCurrentkp1cn_0_85or(unsigned int batch) {
    return kp1cn_0_85or;
}

scalar* getCurrentkp2_0_85or(unsigned int batch) {
    return kp2_0_85or;
}

scalar* getCurrentkp1cn_0_86or(unsigned int batch) {
    return kp1cn_0_86or;
}

scalar* getCurrentkp2_0_86or(unsigned int batch) {
    return kp2_0_86or;
}

scalar* getCurrentkp1cn_0_87or(unsigned int batch) {
    return kp1cn_0_87or;
}

scalar* getCurrentkp2_0_87or(unsigned int batch) {
    return kp2_0_87or;
}

scalar* getCurrentkp1cn_0_88or(unsigned int batch) {
    return kp1cn_0_88or;
}

scalar* getCurrentkp2_0_88or(unsigned int batch) {
    return kp2_0_88or;
}

scalar* getCurrentkp1cn_0_89or(unsigned int batch) {
    return kp1cn_0_89or;
}

scalar* getCurrentkp2_0_89or(unsigned int batch) {
    return kp2_0_89or;
}

scalar* getCurrentkp1cn_0_90or(unsigned int batch) {
    return kp1cn_0_90or;
}

scalar* getCurrentkp2_0_90or(unsigned int batch) {
    return kp2_0_90or;
}

scalar* getCurrentkp1cn_0_91or(unsigned int batch) {
    return kp1cn_0_91or;
}

scalar* getCurrentkp2_0_91or(unsigned int batch) {
    return kp2_0_91or;
}

scalar* getCurrentkp1cn_0_92or(unsigned int batch) {
    return kp1cn_0_92or;
}

scalar* getCurrentkp2_0_92or(unsigned int batch) {
    return kp2_0_92or;
}

scalar* getCurrentkp1cn_0_93or(unsigned int batch) {
    return kp1cn_0_93or;
}

scalar* getCurrentkp2_0_93or(unsigned int batch) {
    return kp2_0_93or;
}

scalar* getCurrentkp1cn_0_94or(unsigned int batch) {
    return kp1cn_0_94or;
}

scalar* getCurrentkp2_0_94or(unsigned int batch) {
    return kp2_0_94or;
}

scalar* getCurrentkp1cn_0_95or(unsigned int batch) {
    return kp1cn_0_95or;
}

scalar* getCurrentkp2_0_95or(unsigned int batch) {
    return kp2_0_95or;
}

scalar* getCurrentkp1cn_0_96or(unsigned int batch) {
    return kp1cn_0_96or;
}

scalar* getCurrentkp2_0_96or(unsigned int batch) {
    return kp2_0_96or;
}

scalar* getCurrentkp1cn_0_97or(unsigned int batch) {
    return kp1cn_0_97or;
}

scalar* getCurrentkp2_0_97or(unsigned int batch) {
    return kp2_0_97or;
}

scalar* getCurrentkp1cn_0_98or(unsigned int batch) {
    return kp1cn_0_98or;
}

scalar* getCurrentkp2_0_98or(unsigned int batch) {
    return kp2_0_98or;
}

scalar* getCurrentkp1cn_0_99or(unsigned int batch) {
    return kp1cn_0_99or;
}

scalar* getCurrentkp2_0_99or(unsigned int batch) {
    return kp2_0_99or;
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
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_0or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_0or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_1or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_1or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_2or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_2or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_3or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_3or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_3or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_3or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_4or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_4or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_4or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_4or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_5or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_5or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_5or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_5or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_6or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_6or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_6or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_6or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_7or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_7or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_7or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_7or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_8or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_8or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_8or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_8or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_9or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_9or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_9or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_9or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_10or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_10or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_10or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_10or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_11or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_11or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_11or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_11or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_12or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_12or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_12or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_12or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_13or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_13or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_13or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_13or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_14or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_14or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_14or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_14or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_15or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_15or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_15or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_15or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_16or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_16or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_16or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_16or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_17or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_17or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_17or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_17or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_18or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_18or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_18or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_18or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_19or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_19or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_19or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_19or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_20or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_20or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_20or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_20or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_21or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_21or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_21or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_21or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_22or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_22or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_22or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_22or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_23or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_23or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_23or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_23or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_24or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_24or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_24or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_24or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_25or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_25or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_25or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_25or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_26or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_26or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_26or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_26or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_27or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_27or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_27or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_27or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_28or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_28or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_28or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_28or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_29or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_29or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_29or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_29or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_30or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_30or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_30or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_30or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_31or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_31or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_31or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_31or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_32or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_32or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_32or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_32or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_33or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_33or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_33or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_33or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_34or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_34or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_34or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_34or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_35or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_35or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_35or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_35or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_36or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_36or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_36or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_36or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_37or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_37or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_37or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_37or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_38or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_38or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_38or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_38or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_39or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_39or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_39or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_39or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_40or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_40or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_40or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_40or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_41or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_41or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_41or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_41or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_42or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_42or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_42or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_42or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_43or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_43or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_43or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_43or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_44or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_44or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_44or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_44or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_45or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_45or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_45or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_45or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_46or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_46or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_46or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_46or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_47or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_47or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_47or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_47or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_48or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_48or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_48or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_48or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_49or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_49or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_49or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_49or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_50or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_50or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_50or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_50or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_51or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_51or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_51or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_51or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_52or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_52or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_52or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_52or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_53or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_53or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_53or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_53or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_54or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_54or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_54or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_54or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_55or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_55or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_55or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_55or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_56or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_56or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_56or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_56or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_57or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_57or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_57or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_57or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_58or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_58or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_58or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_58or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_59or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_59or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_59or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_59or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_60or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_60or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_60or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_60or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_61or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_61or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_61or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_61or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_62or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_62or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_62or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_62or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_63or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_63or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_63or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_63or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_64or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_64or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_64or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_64or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_65or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_65or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_65or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_65or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_66or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_66or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_66or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_66or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_67or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_67or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_67or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_67or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_68or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_68or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_68or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_68or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_69or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_69or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_69or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_69or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_70or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_70or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_70or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_70or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_71or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_71or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_71or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_71or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_72or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_72or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_72or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_72or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_73or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_73or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_73or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_73or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_74or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_74or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_74or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_74or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_75or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_75or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_75or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_75or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_76or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_76or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_76or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_76or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_77or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_77or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_77or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_77or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_78or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_78or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_78or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_78or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_79or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_79or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_79or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_79or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_80or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_80or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_80or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_80or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_81or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_81or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_81or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_81or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_82or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_82or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_82or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_82or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_83or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_83or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_83or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_83or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_84or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_84or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_84or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_84or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_85or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_85or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_85or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_85or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_86or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_86or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_86or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_86or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_87or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_87or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_87or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_87or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_88or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_88or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_88or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_88or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_89or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_89or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_89or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_89or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_90or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_90or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_90or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_90or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_91or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_91or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_91or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_91or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_92or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_92or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_92or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_92or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_93or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_93or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_93or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_93or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_94or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_94or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_94or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_94or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_95or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_95or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_95or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_95or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_96or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_96or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_96or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_96or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_97or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_97or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_97or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_97or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_98or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_98or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_98or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_98or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp1cn_0_99or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp1cn_0_99or, 160 * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&kp2_0_99or, 160 * sizeof(scalar), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_kp2_0_99or, 160 * sizeof(scalar)));
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
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynln_ln, 4000 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynln_ln, 4000 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynor_orn, 9600 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynor_orn, 9600 * sizeof(double)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&inSynorn_ln, 800 * sizeof(double), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_inSynorn_ln, 800 * sizeof(double)));
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
    CHECK_CUDA_ERRORS(cudaHostAlloc(&indorn_ln, 7680000 * sizeof(uint32_t), cudaHostAllocPortable));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_indorn_ln, 7680000 * sizeof(uint32_t)));
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
    
    pushMergedNeuronInitGroup0ToDevice(0, d_glbSpkCntpn, d_glbSpkpn, d_rngpn, d_inSynorn_ln, d_inSynln_pn, d_inSynorn_pn, 800);
    pushMergedNeuronInitGroup1ToDevice(0, d_glbSpkCntorn, d_glbSpkorn, d_rngorn, d_inSynor_orn, 9600);
    pushMergedNeuronInitGroup2ToDevice(0, d_glbSpkCntor, d_glbSpkor, 160);
    pushMergedNeuronInitGroup3ToDevice(0, d_glbSpkCntln, d_glbSpkln, d_rngln, d_inSynpn_ln, d_inSynln_ln, 4000);
    pushMergedSynapseDenseInitGroup0ToDevice(0, 2.00000000000000016e-05, d_gln_ln, 4000, 4000, 4000);
    pushMergedSynapseDenseInitGroup0ToDevice(1, 5.50000000000000020e-05, d_gln_pn, 800, 4000, 800);
    pushMergedSynapseConnectivityInitGroup0ToDevice(0, d_rowLengthpn_ln, d_indpn_ln, 25, 800, 4000);
    pushMergedSynapseConnectivityInitGroup1ToDevice(0, d_rowLengthorn_ln, d_indorn_ln, 2.50000000000000000e+01, 800, 9600, 800);
    pushMergedSynapseConnectivityInitGroup1ToDevice(1, d_rowLengthorn_pn, d_indorn_pn, 5.00000000000000000e+00, 800, 9600, 800);
    pushMergedSynapseConnectivityInitGroup2ToDevice(0, d_rowLengthor_orn, d_indor_orn, 60, 160, 9600);
    pushMergedNeuronUpdateGroup0ToDevice(0, d_glbSpkCntpn, d_glbSpkpn, d_rngpn, d_Vpn, d_apn, d_inSynorn_ln, d_inSynorn_pn, d_inSynln_pn, d_recordSpkpn, 800);
    pushMergedNeuronUpdateGroup1ToDevice(0, d_glbSpkCntorn, d_glbSpkorn, d_rngorn, d_Vorn, d_aorn, d_inSynor_orn, 9600);
    pushMergedNeuronUpdateGroup2ToDevice(0, d_kp1cn_0_65or, d_kp2_0_58or, d_kp1cn_0_59or, d_kp2_0_59or, d_kp1cn_0_60or, d_kp2_0_60or, d_kp1cn_0_61or, d_kp2_0_61or, d_kp1cn_0_62or, d_kp2_0_62or, d_kp1cn_0_63or, d_kp2_0_63or, d_kp1cn_0_64or, d_kp2_0_64or, d_kp1cn_0_58or, d_kp2_0_65or, d_kp1cn_0_66or, d_kp2_0_66or, d_kp1cn_0_67or, d_kp2_0_67or, d_kp1cn_0_68or, d_kp2_0_68or, d_kp1cn_0_69or, d_kp2_0_69or, d_kp1cn_0_70or, d_kp2_0_70or, d_kp1cn_0_71or, d_kp2_0_71or, d_kp1cn_0_51or, d_kp2_0_44or, d_kp1cn_0_45or, d_kp2_0_45or, d_kp1cn_0_46or, d_kp2_0_46or, d_kp1cn_0_47or, d_kp2_0_47or, d_kp1cn_0_48or, d_kp2_0_48or, d_kp1cn_0_49or, d_kp2_0_49or, d_kp1cn_0_50or, d_kp2_0_50or, d_kp1cn_0_72or, d_kp2_0_51or, d_kp1cn_0_52or, d_kp2_0_52or, d_kp1cn_0_53or, d_kp2_0_53or, d_kp1cn_0_54or, d_kp2_0_54or, d_kp1cn_0_55or, d_kp2_0_55or, d_kp1cn_0_56or, d_kp2_0_56or, d_kp1cn_0_57or, d_kp2_0_57or, d_kp1cn_0_93or, d_kp2_0_86or, d_kp1cn_0_87or, d_kp2_0_87or, d_kp1cn_0_88or, d_kp2_0_88or, d_kp1cn_0_89or, d_kp2_0_89or, d_kp1cn_0_90or, d_kp2_0_90or, d_kp1cn_0_91or, d_kp2_0_91or, d_kp1cn_0_92or, d_kp2_0_92or, d_kp1cn_0_86or, d_kp2_0_93or, d_kp1cn_0_94or, d_kp2_0_94or, d_kp1cn_0_95or, d_kp2_0_95or, d_kp1cn_0_96or, d_kp2_0_96or, d_kp1cn_0_97or, d_kp2_0_97or, d_kp1cn_0_98or, d_kp2_0_98or, d_kp1cn_0_99or, d_kp2_0_99or, d_kp1cn_0_79or, d_kp2_0_72or, d_kp1cn_0_73or, d_kp2_0_73or, d_kp1cn_0_74or, d_kp2_0_74or, d_kp1cn_0_75or, d_kp2_0_75or, d_kp1cn_0_76or, d_kp2_0_76or, d_kp1cn_0_77or, d_kp2_0_77or, d_kp1cn_0_78or, d_kp2_0_78or, d_kp1cn_0_44or, d_kp2_0_79or, d_kp1cn_0_80or, d_kp2_0_80or, d_kp1cn_0_81or, d_kp2_0_81or, d_kp1cn_0_82or, d_kp2_0_82or, d_kp1cn_0_83or, d_kp2_0_83or, d_kp1cn_0_84or, d_kp2_0_84or, d_kp1cn_0_85or, d_kp2_0_85or, d_kp2_0_9or, d_kp1cn_0_3or, d_kp2_0_3or, d_kp1cn_0_4or, d_kp2_0_4or, d_kp1cn_0_5or, d_kp2_0_5or, d_kp1cn_0_6or, d_kp2_0_6or, d_kp1cn_0_7or, d_kp2_0_7or, d_kp1cn_0_8or, d_kp2_0_8or, d_kp1cn_0_9or, d_kp2_0_2or, d_kp1cn_0_10or, d_kp2_0_10or, d_kp1cn_0_11or, d_kp2_0_11or, d_kp1cn_0_12or, d_kp2_0_12or, d_kp1cn_0_13or, d_kp2_0_13or, d_kp1cn_0_14or, d_kp2_0_14or, d_kp1cn_0_15or, d_kp2_0_15or, d_kp1cn_0_16or, d_km2_0or, d_glbSpkCntor, d_glbSpkor, d_r0or, d_rb_0or, d_ra_0or, d_rb_1or, d_ra_1or, d_rb_2or, d_ra_2or, d_raor, d_kp1cn_0or, d_km1_0or, d_kp2_0or, d_kp2_0_16or, d_kp1cn_1or, d_km1_1or, d_kp2_1or, d_km2_1or, d_kp1cn_2or, d_km1_2or, d_kp2_2or, d_km2_2or, d_kp1cn_0_0or, d_kp2_0_0or, d_kp1cn_0_1or, d_kp2_0_1or, d_kp1cn_0_2or, d_kp1cn_0_37or, d_kp2_0_30or, d_kp1cn_0_31or, d_kp2_0_31or, d_kp1cn_0_32or, d_kp2_0_32or, d_kp1cn_0_33or, d_kp2_0_33or, d_kp1cn_0_34or, d_kp2_0_34or, d_kp1cn_0_35or, d_kp2_0_35or, d_kp1cn_0_36or, d_kp2_0_36or, d_kp1cn_0_30or, d_kp2_0_37or, d_kp1cn_0_38or, d_kp2_0_38or, d_kp1cn_0_39or, d_kp2_0_39or, d_kp1cn_0_40or, d_kp2_0_40or, d_kp1cn_0_41or, d_kp2_0_41or, d_kp1cn_0_42or, d_kp2_0_42or, d_kp1cn_0_43or, d_kp2_0_43or, d_kp2_0_23or, d_kp1cn_0_17or, d_kp2_0_17or, d_kp1cn_0_18or, d_kp2_0_18or, d_kp1cn_0_19or, d_kp2_0_19or, d_kp1cn_0_20or, d_kp2_0_20or, d_kp1cn_0_21or, d_kp2_0_21or, d_kp1cn_0_22or, d_kp2_0_22or, d_kp1cn_0_23or, d_kp1cn_0_24or, d_kp2_0_24or, d_kp1cn_0_25or, d_kp2_0_25or, d_kp1cn_0_26or, d_kp2_0_26or, d_kp1cn_0_27or, d_kp2_0_27or, d_kp1cn_0_28or, d_kp2_0_28or, d_kp1cn_0_29or, d_kp2_0_29or, 160);
    pushMergedNeuronUpdateGroup3ToDevice(0, d_glbSpkCntln, d_glbSpkln, d_rngln, d_Vln, d_aln, d_inSynpn_ln, d_inSynln_ln, d_recordSpkln, 4000);
    pushMergedPresynapticUpdateGroup0ToDevice(0, d_inSynorn_ln, d_glbSpkCntorn, d_glbSpkorn, d_rowLengthorn_ln, d_indorn_ln, 8.00000000000000017e-03, 800, 9600, 800);
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
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_0or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_0or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_1or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_1or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_2or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_2or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_3or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_3or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_3or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_3or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_4or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_4or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_4or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_4or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_5or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_5or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_5or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_5or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_6or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_6or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_6or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_6or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_7or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_7or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_7or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_7or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_8or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_8or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_8or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_8or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_9or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_9or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_9or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_9or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_10or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_10or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_10or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_10or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_11or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_11or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_11or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_11or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_12or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_12or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_12or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_12or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_13or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_13or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_13or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_13or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_14or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_14or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_14or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_14or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_15or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_15or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_15or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_15or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_16or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_16or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_16or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_16or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_17or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_17or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_17or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_17or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_18or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_18or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_18or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_18or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_19or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_19or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_19or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_19or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_20or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_20or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_20or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_20or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_21or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_21or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_21or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_21or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_22or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_22or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_22or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_22or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_23or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_23or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_23or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_23or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_24or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_24or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_24or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_24or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_25or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_25or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_25or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_25or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_26or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_26or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_26or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_26or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_27or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_27or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_27or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_27or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_28or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_28or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_28or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_28or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_29or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_29or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_29or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_29or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_30or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_30or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_30or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_30or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_31or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_31or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_31or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_31or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_32or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_32or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_32or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_32or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_33or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_33or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_33or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_33or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_34or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_34or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_34or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_34or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_35or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_35or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_35or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_35or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_36or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_36or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_36or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_36or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_37or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_37or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_37or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_37or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_38or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_38or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_38or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_38or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_39or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_39or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_39or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_39or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_40or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_40or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_40or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_40or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_41or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_41or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_41or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_41or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_42or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_42or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_42or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_42or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_43or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_43or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_43or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_43or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_44or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_44or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_44or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_44or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_45or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_45or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_45or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_45or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_46or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_46or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_46or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_46or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_47or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_47or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_47or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_47or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_48or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_48or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_48or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_48or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_49or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_49or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_49or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_49or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_50or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_50or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_50or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_50or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_51or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_51or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_51or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_51or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_52or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_52or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_52or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_52or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_53or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_53or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_53or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_53or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_54or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_54or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_54or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_54or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_55or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_55or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_55or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_55or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_56or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_56or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_56or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_56or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_57or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_57or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_57or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_57or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_58or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_58or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_58or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_58or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_59or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_59or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_59or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_59or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_60or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_60or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_60or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_60or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_61or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_61or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_61or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_61or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_62or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_62or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_62or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_62or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_63or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_63or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_63or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_63or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_64or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_64or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_64or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_64or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_65or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_65or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_65or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_65or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_66or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_66or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_66or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_66or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_67or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_67or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_67or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_67or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_68or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_68or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_68or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_68or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_69or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_69or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_69or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_69or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_70or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_70or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_70or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_70or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_71or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_71or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_71or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_71or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_72or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_72or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_72or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_72or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_73or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_73or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_73or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_73or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_74or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_74or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_74or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_74or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_75or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_75or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_75or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_75or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_76or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_76or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_76or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_76or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_77or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_77or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_77or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_77or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_78or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_78or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_78or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_78or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_79or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_79or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_79or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_79or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_80or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_80or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_80or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_80or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_81or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_81or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_81or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_81or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_82or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_82or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_82or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_82or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_83or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_83or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_83or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_83or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_84or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_84or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_84or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_84or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_85or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_85or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_85or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_85or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_86or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_86or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_86or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_86or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_87or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_87or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_87or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_87or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_88or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_88or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_88or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_88or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_89or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_89or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_89or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_89or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_90or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_90or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_90or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_90or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_91or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_91or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_91or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_91or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_92or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_92or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_92or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_92or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_93or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_93or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_93or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_93or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_94or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_94or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_94or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_94or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_95or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_95or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_95or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_95or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_96or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_96or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_96or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_96or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_97or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_97or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_97or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_97or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_98or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_98or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_98or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_98or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp1cn_0_99or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp1cn_0_99or));
    CHECK_CUDA_ERRORS(cudaFreeHost(kp2_0_99or));
    CHECK_CUDA_ERRORS(cudaFree(d_kp2_0_99or));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkCntorn));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntorn));
    CHECK_CUDA_ERRORS(cudaFreeHost(glbSpkorn));
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkorn));
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
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynln_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynln_ln));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynor_orn));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynor_orn));
    CHECK_CUDA_ERRORS(cudaFreeHost(inSynorn_ln));
    CHECK_CUDA_ERRORS(cudaFree(d_inSynorn_ln));
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

