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
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup1
 {
    scalar* kp1cn_0_65;
    scalar* kp2_0_58;
    scalar* kp1cn_0_59;
    scalar* kp2_0_59;
    scalar* kp1cn_0_60;
    scalar* kp2_0_60;
    scalar* kp1cn_0_61;
    scalar* kp2_0_61;
    scalar* kp1cn_0_62;
    scalar* kp2_0_62;
    scalar* kp1cn_0_63;
    scalar* kp2_0_63;
    scalar* kp1cn_0_64;
    scalar* kp2_0_64;
    scalar* kp1cn_0_58;
    scalar* kp2_0_65;
    scalar* kp1cn_0_66;
    scalar* kp2_0_66;
    scalar* kp1cn_0_67;
    scalar* kp2_0_67;
    scalar* kp1cn_0_68;
    scalar* kp2_0_68;
    scalar* kp1cn_0_69;
    scalar* kp2_0_69;
    scalar* kp1cn_0_70;
    scalar* kp2_0_70;
    scalar* kp1cn_0_71;
    scalar* kp2_0_71;
    scalar* kp1cn_0_51;
    scalar* kp2_0_44;
    scalar* kp1cn_0_45;
    scalar* kp2_0_45;
    scalar* kp1cn_0_46;
    scalar* kp2_0_46;
    scalar* kp1cn_0_47;
    scalar* kp2_0_47;
    scalar* kp1cn_0_48;
    scalar* kp2_0_48;
    scalar* kp1cn_0_49;
    scalar* kp2_0_49;
    scalar* kp1cn_0_50;
    scalar* kp2_0_50;
    scalar* kp1cn_0_72;
    scalar* kp2_0_51;
    scalar* kp1cn_0_52;
    scalar* kp2_0_52;
    scalar* kp1cn_0_53;
    scalar* kp2_0_53;
    scalar* kp1cn_0_54;
    scalar* kp2_0_54;
    scalar* kp1cn_0_55;
    scalar* kp2_0_55;
    scalar* kp1cn_0_56;
    scalar* kp2_0_56;
    scalar* kp1cn_0_57;
    scalar* kp2_0_57;
    scalar* kp1cn_0_93;
    scalar* kp2_0_86;
    scalar* kp1cn_0_87;
    scalar* kp2_0_87;
    scalar* kp1cn_0_88;
    scalar* kp2_0_88;
    scalar* kp1cn_0_89;
    scalar* kp2_0_89;
    scalar* kp1cn_0_90;
    scalar* kp2_0_90;
    scalar* kp1cn_0_91;
    scalar* kp2_0_91;
    scalar* kp1cn_0_92;
    scalar* kp2_0_92;
    scalar* kp1cn_0_86;
    scalar* kp2_0_93;
    scalar* kp1cn_0_94;
    scalar* kp2_0_94;
    scalar* kp1cn_0_95;
    scalar* kp2_0_95;
    scalar* kp1cn_0_96;
    scalar* kp2_0_96;
    scalar* kp1cn_0_97;
    scalar* kp2_0_97;
    scalar* kp1cn_0_98;
    scalar* kp2_0_98;
    scalar* kp1cn_0_99;
    scalar* kp2_0_99;
    scalar* kp1cn_0_79;
    scalar* kp2_0_72;
    scalar* kp1cn_0_73;
    scalar* kp2_0_73;
    scalar* kp1cn_0_74;
    scalar* kp2_0_74;
    scalar* kp1cn_0_75;
    scalar* kp2_0_75;
    scalar* kp1cn_0_76;
    scalar* kp2_0_76;
    scalar* kp1cn_0_77;
    scalar* kp2_0_77;
    scalar* kp1cn_0_78;
    scalar* kp2_0_78;
    scalar* kp1cn_0_44;
    scalar* kp2_0_79;
    scalar* kp1cn_0_80;
    scalar* kp2_0_80;
    scalar* kp1cn_0_81;
    scalar* kp2_0_81;
    scalar* kp1cn_0_82;
    scalar* kp2_0_82;
    scalar* kp1cn_0_83;
    scalar* kp2_0_83;
    scalar* kp1cn_0_84;
    scalar* kp2_0_84;
    scalar* kp1cn_0_85;
    scalar* kp2_0_85;
    scalar* kp2_0_9;
    scalar* kp1cn_0_3;
    scalar* kp2_0_3;
    scalar* kp1cn_0_4;
    scalar* kp2_0_4;
    scalar* kp1cn_0_5;
    scalar* kp2_0_5;
    scalar* kp1cn_0_6;
    scalar* kp2_0_6;
    scalar* kp1cn_0_7;
    scalar* kp2_0_7;
    scalar* kp1cn_0_8;
    scalar* kp2_0_8;
    scalar* kp1cn_0_9;
    scalar* kp2_0_2;
    scalar* kp1cn_0_10;
    scalar* kp2_0_10;
    scalar* kp1cn_0_11;
    scalar* kp2_0_11;
    scalar* kp1cn_0_12;
    scalar* kp2_0_12;
    scalar* kp1cn_0_13;
    scalar* kp2_0_13;
    scalar* kp1cn_0_14;
    scalar* kp2_0_14;
    scalar* kp1cn_0_15;
    scalar* kp2_0_15;
    scalar* kp1cn_0_16;
    scalar* km2_0;
    unsigned int* spkCnt;
    unsigned int* spk;
    scalar* r0;
    scalar* rb_0;
    scalar* ra_0;
    scalar* rb_1;
    scalar* ra_1;
    scalar* rb_2;
    scalar* ra_2;
    scalar* ra;
    scalar* kp1cn_0;
    scalar* km1_0;
    scalar* kp2_0;
    scalar* kp2_0_16;
    scalar* kp1cn_1;
    scalar* km1_1;
    scalar* kp2_1;
    scalar* km2_1;
    scalar* kp1cn_2;
    scalar* km1_2;
    scalar* kp2_2;
    scalar* km2_2;
    scalar* kp1cn_0_0;
    scalar* kp2_0_0;
    scalar* kp1cn_0_1;
    scalar* kp2_0_1;
    scalar* kp1cn_0_2;
    scalar* kp1cn_0_37;
    scalar* kp2_0_30;
    scalar* kp1cn_0_31;
    scalar* kp2_0_31;
    scalar* kp1cn_0_32;
    scalar* kp2_0_32;
    scalar* kp1cn_0_33;
    scalar* kp2_0_33;
    scalar* kp1cn_0_34;
    scalar* kp2_0_34;
    scalar* kp1cn_0_35;
    scalar* kp2_0_35;
    scalar* kp1cn_0_36;
    scalar* kp2_0_36;
    scalar* kp1cn_0_30;
    scalar* kp2_0_37;
    scalar* kp1cn_0_38;
    scalar* kp2_0_38;
    scalar* kp1cn_0_39;
    scalar* kp2_0_39;
    scalar* kp1cn_0_40;
    scalar* kp2_0_40;
    scalar* kp1cn_0_41;
    scalar* kp2_0_41;
    scalar* kp1cn_0_42;
    scalar* kp2_0_42;
    scalar* kp1cn_0_43;
    scalar* kp2_0_43;
    scalar* kp2_0_23;
    scalar* kp1cn_0_17;
    scalar* kp2_0_17;
    scalar* kp1cn_0_18;
    scalar* kp2_0_18;
    scalar* kp1cn_0_19;
    scalar* kp2_0_19;
    scalar* kp1cn_0_20;
    scalar* kp2_0_20;
    scalar* kp1cn_0_21;
    scalar* kp2_0_21;
    scalar* kp1cn_0_22;
    scalar* kp2_0_22;
    scalar* kp1cn_0_23;
    scalar* kp1cn_0_24;
    scalar* kp2_0_24;
    scalar* kp1cn_0_25;
    scalar* kp2_0_25;
    scalar* kp1cn_0_26;
    scalar* kp2_0_26;
    scalar* kp1cn_0_27;
    scalar* kp2_0_27;
    scalar* kp1cn_0_28;
    scalar* kp2_0_28;
    scalar* kp1cn_0_29;
    scalar* kp2_0_29;
    unsigned int numNeurons;
    
}
;
struct MergedNeuronUpdateGroup2
 {
    unsigned int* spkCnt;
    unsigned int* spk;
    curandState* rng;
    scalar* V;
    scalar* a;
    double* inSynInSyn0;
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
void pushMergedNeuronUpdateGroup0ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, double* inSynInSyn1, unsigned int numNeurons) {
    MergedNeuronUpdateGroup0 group = {spkCnt, spk, rng, V, a, inSynInSyn0, inSynInSyn1, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup0, &group, sizeof(MergedNeuronUpdateGroup0), idx * sizeof(MergedNeuronUpdateGroup0)));
}
__device__ __constant__ MergedNeuronUpdateGroup1 d_mergedNeuronUpdateGroup1[1];
void pushMergedNeuronUpdateGroup1ToDevice(unsigned int idx, scalar* kp1cn_0_65, scalar* kp2_0_58, scalar* kp1cn_0_59, scalar* kp2_0_59, scalar* kp1cn_0_60, scalar* kp2_0_60, scalar* kp1cn_0_61, scalar* kp2_0_61, scalar* kp1cn_0_62, scalar* kp2_0_62, scalar* kp1cn_0_63, scalar* kp2_0_63, scalar* kp1cn_0_64, scalar* kp2_0_64, scalar* kp1cn_0_58, scalar* kp2_0_65, scalar* kp1cn_0_66, scalar* kp2_0_66, scalar* kp1cn_0_67, scalar* kp2_0_67, scalar* kp1cn_0_68, scalar* kp2_0_68, scalar* kp1cn_0_69, scalar* kp2_0_69, scalar* kp1cn_0_70, scalar* kp2_0_70, scalar* kp1cn_0_71, scalar* kp2_0_71, scalar* kp1cn_0_51, scalar* kp2_0_44, scalar* kp1cn_0_45, scalar* kp2_0_45, scalar* kp1cn_0_46, scalar* kp2_0_46, scalar* kp1cn_0_47, scalar* kp2_0_47, scalar* kp1cn_0_48, scalar* kp2_0_48, scalar* kp1cn_0_49, scalar* kp2_0_49, scalar* kp1cn_0_50, scalar* kp2_0_50, scalar* kp1cn_0_72, scalar* kp2_0_51, scalar* kp1cn_0_52, scalar* kp2_0_52, scalar* kp1cn_0_53, scalar* kp2_0_53, scalar* kp1cn_0_54, scalar* kp2_0_54, scalar* kp1cn_0_55, scalar* kp2_0_55, scalar* kp1cn_0_56, scalar* kp2_0_56, scalar* kp1cn_0_57, scalar* kp2_0_57, scalar* kp1cn_0_93, scalar* kp2_0_86, scalar* kp1cn_0_87, scalar* kp2_0_87, scalar* kp1cn_0_88, scalar* kp2_0_88, scalar* kp1cn_0_89, scalar* kp2_0_89, scalar* kp1cn_0_90, scalar* kp2_0_90, scalar* kp1cn_0_91, scalar* kp2_0_91, scalar* kp1cn_0_92, scalar* kp2_0_92, scalar* kp1cn_0_86, scalar* kp2_0_93, scalar* kp1cn_0_94, scalar* kp2_0_94, scalar* kp1cn_0_95, scalar* kp2_0_95, scalar* kp1cn_0_96, scalar* kp2_0_96, scalar* kp1cn_0_97, scalar* kp2_0_97, scalar* kp1cn_0_98, scalar* kp2_0_98, scalar* kp1cn_0_99, scalar* kp2_0_99, scalar* kp1cn_0_79, scalar* kp2_0_72, scalar* kp1cn_0_73, scalar* kp2_0_73, scalar* kp1cn_0_74, scalar* kp2_0_74, scalar* kp1cn_0_75, scalar* kp2_0_75, scalar* kp1cn_0_76, scalar* kp2_0_76, scalar* kp1cn_0_77, scalar* kp2_0_77, scalar* kp1cn_0_78, scalar* kp2_0_78, scalar* kp1cn_0_44, scalar* kp2_0_79, scalar* kp1cn_0_80, scalar* kp2_0_80, scalar* kp1cn_0_81, scalar* kp2_0_81, scalar* kp1cn_0_82, scalar* kp2_0_82, scalar* kp1cn_0_83, scalar* kp2_0_83, scalar* kp1cn_0_84, scalar* kp2_0_84, scalar* kp1cn_0_85, scalar* kp2_0_85, scalar* kp2_0_9, scalar* kp1cn_0_3, scalar* kp2_0_3, scalar* kp1cn_0_4, scalar* kp2_0_4, scalar* kp1cn_0_5, scalar* kp2_0_5, scalar* kp1cn_0_6, scalar* kp2_0_6, scalar* kp1cn_0_7, scalar* kp2_0_7, scalar* kp1cn_0_8, scalar* kp2_0_8, scalar* kp1cn_0_9, scalar* kp2_0_2, scalar* kp1cn_0_10, scalar* kp2_0_10, scalar* kp1cn_0_11, scalar* kp2_0_11, scalar* kp1cn_0_12, scalar* kp2_0_12, scalar* kp1cn_0_13, scalar* kp2_0_13, scalar* kp1cn_0_14, scalar* kp2_0_14, scalar* kp1cn_0_15, scalar* kp2_0_15, scalar* kp1cn_0_16, scalar* km2_0, unsigned int* spkCnt, unsigned int* spk, scalar* r0, scalar* rb_0, scalar* ra_0, scalar* rb_1, scalar* ra_1, scalar* rb_2, scalar* ra_2, scalar* ra, scalar* kp1cn_0, scalar* km1_0, scalar* kp2_0, scalar* kp2_0_16, scalar* kp1cn_1, scalar* km1_1, scalar* kp2_1, scalar* km2_1, scalar* kp1cn_2, scalar* km1_2, scalar* kp2_2, scalar* km2_2, scalar* kp1cn_0_0, scalar* kp2_0_0, scalar* kp1cn_0_1, scalar* kp2_0_1, scalar* kp1cn_0_2, scalar* kp1cn_0_37, scalar* kp2_0_30, scalar* kp1cn_0_31, scalar* kp2_0_31, scalar* kp1cn_0_32, scalar* kp2_0_32, scalar* kp1cn_0_33, scalar* kp2_0_33, scalar* kp1cn_0_34, scalar* kp2_0_34, scalar* kp1cn_0_35, scalar* kp2_0_35, scalar* kp1cn_0_36, scalar* kp2_0_36, scalar* kp1cn_0_30, scalar* kp2_0_37, scalar* kp1cn_0_38, scalar* kp2_0_38, scalar* kp1cn_0_39, scalar* kp2_0_39, scalar* kp1cn_0_40, scalar* kp2_0_40, scalar* kp1cn_0_41, scalar* kp2_0_41, scalar* kp1cn_0_42, scalar* kp2_0_42, scalar* kp1cn_0_43, scalar* kp2_0_43, scalar* kp2_0_23, scalar* kp1cn_0_17, scalar* kp2_0_17, scalar* kp1cn_0_18, scalar* kp2_0_18, scalar* kp1cn_0_19, scalar* kp2_0_19, scalar* kp1cn_0_20, scalar* kp2_0_20, scalar* kp1cn_0_21, scalar* kp2_0_21, scalar* kp1cn_0_22, scalar* kp2_0_22, scalar* kp1cn_0_23, scalar* kp1cn_0_24, scalar* kp2_0_24, scalar* kp1cn_0_25, scalar* kp2_0_25, scalar* kp1cn_0_26, scalar* kp2_0_26, scalar* kp1cn_0_27, scalar* kp2_0_27, scalar* kp1cn_0_28, scalar* kp2_0_28, scalar* kp1cn_0_29, scalar* kp2_0_29, unsigned int numNeurons) {
    MergedNeuronUpdateGroup1 group = {kp1cn_0_65, kp2_0_58, kp1cn_0_59, kp2_0_59, kp1cn_0_60, kp2_0_60, kp1cn_0_61, kp2_0_61, kp1cn_0_62, kp2_0_62, kp1cn_0_63, kp2_0_63, kp1cn_0_64, kp2_0_64, kp1cn_0_58, kp2_0_65, kp1cn_0_66, kp2_0_66, kp1cn_0_67, kp2_0_67, kp1cn_0_68, kp2_0_68, kp1cn_0_69, kp2_0_69, kp1cn_0_70, kp2_0_70, kp1cn_0_71, kp2_0_71, kp1cn_0_51, kp2_0_44, kp1cn_0_45, kp2_0_45, kp1cn_0_46, kp2_0_46, kp1cn_0_47, kp2_0_47, kp1cn_0_48, kp2_0_48, kp1cn_0_49, kp2_0_49, kp1cn_0_50, kp2_0_50, kp1cn_0_72, kp2_0_51, kp1cn_0_52, kp2_0_52, kp1cn_0_53, kp2_0_53, kp1cn_0_54, kp2_0_54, kp1cn_0_55, kp2_0_55, kp1cn_0_56, kp2_0_56, kp1cn_0_57, kp2_0_57, kp1cn_0_93, kp2_0_86, kp1cn_0_87, kp2_0_87, kp1cn_0_88, kp2_0_88, kp1cn_0_89, kp2_0_89, kp1cn_0_90, kp2_0_90, kp1cn_0_91, kp2_0_91, kp1cn_0_92, kp2_0_92, kp1cn_0_86, kp2_0_93, kp1cn_0_94, kp2_0_94, kp1cn_0_95, kp2_0_95, kp1cn_0_96, kp2_0_96, kp1cn_0_97, kp2_0_97, kp1cn_0_98, kp2_0_98, kp1cn_0_99, kp2_0_99, kp1cn_0_79, kp2_0_72, kp1cn_0_73, kp2_0_73, kp1cn_0_74, kp2_0_74, kp1cn_0_75, kp2_0_75, kp1cn_0_76, kp2_0_76, kp1cn_0_77, kp2_0_77, kp1cn_0_78, kp2_0_78, kp1cn_0_44, kp2_0_79, kp1cn_0_80, kp2_0_80, kp1cn_0_81, kp2_0_81, kp1cn_0_82, kp2_0_82, kp1cn_0_83, kp2_0_83, kp1cn_0_84, kp2_0_84, kp1cn_0_85, kp2_0_85, kp2_0_9, kp1cn_0_3, kp2_0_3, kp1cn_0_4, kp2_0_4, kp1cn_0_5, kp2_0_5, kp1cn_0_6, kp2_0_6, kp1cn_0_7, kp2_0_7, kp1cn_0_8, kp2_0_8, kp1cn_0_9, kp2_0_2, kp1cn_0_10, kp2_0_10, kp1cn_0_11, kp2_0_11, kp1cn_0_12, kp2_0_12, kp1cn_0_13, kp2_0_13, kp1cn_0_14, kp2_0_14, kp1cn_0_15, kp2_0_15, kp1cn_0_16, km2_0, spkCnt, spk, r0, rb_0, ra_0, rb_1, ra_1, rb_2, ra_2, ra, kp1cn_0, km1_0, kp2_0, kp2_0_16, kp1cn_1, km1_1, kp2_1, km2_1, kp1cn_2, km1_2, kp2_2, km2_2, kp1cn_0_0, kp2_0_0, kp1cn_0_1, kp2_0_1, kp1cn_0_2, kp1cn_0_37, kp2_0_30, kp1cn_0_31, kp2_0_31, kp1cn_0_32, kp2_0_32, kp1cn_0_33, kp2_0_33, kp1cn_0_34, kp2_0_34, kp1cn_0_35, kp2_0_35, kp1cn_0_36, kp2_0_36, kp1cn_0_30, kp2_0_37, kp1cn_0_38, kp2_0_38, kp1cn_0_39, kp2_0_39, kp1cn_0_40, kp2_0_40, kp1cn_0_41, kp2_0_41, kp1cn_0_42, kp2_0_42, kp1cn_0_43, kp2_0_43, kp2_0_23, kp1cn_0_17, kp2_0_17, kp1cn_0_18, kp2_0_18, kp1cn_0_19, kp2_0_19, kp1cn_0_20, kp2_0_20, kp1cn_0_21, kp2_0_21, kp1cn_0_22, kp2_0_22, kp1cn_0_23, kp1cn_0_24, kp2_0_24, kp1cn_0_25, kp2_0_25, kp1cn_0_26, kp2_0_26, kp1cn_0_27, kp2_0_27, kp1cn_0_28, kp2_0_28, kp1cn_0_29, kp2_0_29, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup1, &group, sizeof(MergedNeuronUpdateGroup1), idx * sizeof(MergedNeuronUpdateGroup1)));
}
__device__ __constant__ MergedNeuronUpdateGroup2 d_mergedNeuronUpdateGroup2[1];
void pushMergedNeuronUpdateGroup2ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, unsigned int numNeurons) {
    MergedNeuronUpdateGroup2 group = {spkCnt, spk, rng, V, a, inSynInSyn0, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup2, &group, sizeof(MergedNeuronUpdateGroup2), idx * sizeof(MergedNeuronUpdateGroup2)));
}
__device__ __constant__ MergedNeuronUpdateGroup3 d_mergedNeuronUpdateGroup3[1];
void pushMergedNeuronUpdateGroup3ToDevice(unsigned int idx, unsigned int* spkCnt, unsigned int* spk, curandState* rng, scalar* V, scalar* a, double* inSynInSyn0, double* inSynInSyn1, double* inSynInSyn2, unsigned int numNeurons) {
    MergedNeuronUpdateGroup3 group = {spkCnt, spk, rng, V, a, inSynInSyn0, inSynInSyn1, inSynInSyn2, numNeurons, };
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbolAsync(d_mergedNeuronUpdateGroup3, &group, sizeof(MergedNeuronUpdateGroup3), idx * sizeof(MergedNeuronUpdateGroup3)));
}
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// merged extra global parameter functions
// ------------------------------------------------------------------------
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID0[] = {0, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID1[] = {32, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID2[] = {192, };
__device__ __constant__ unsigned int d_mergedNeuronUpdateGroupStartID3[] = {224, };

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

extern "C" __global__ void updateNeuronsKernel(double t)
 {
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    __shared__ unsigned int shSpk[32];
    __shared__ unsigned int shPosSpk;
    __shared__ unsigned int shSpkCount;
    if (threadIdx.x == 0) {
        shSpkCount = 0;
    }
    
    __syncthreads();
    // merged0
    if(id < 32) {
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
                linSyn*=(9.90049833749168107e-01);
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn * ((0.00000000000000000e+00) - lV);
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
    }
    // merged1
    if(id >= 32 && id < 192) {
        struct MergedNeuronUpdateGroup1 *group = &d_mergedNeuronUpdateGroup1[0]; 
        const unsigned int lid = id - 32;
        
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
            scalar lkp1cn_0_0 = group->kp1cn_0_0[lid];
            scalar lkp2_0_0 = group->kp2_0_0[lid];
            scalar lkp1cn_0_1 = group->kp1cn_0_1[lid];
            scalar lkp2_0_1 = group->kp2_0_1[lid];
            scalar lkp1cn_0_2 = group->kp1cn_0_2[lid];
            scalar lkp2_0_2 = group->kp2_0_2[lid];
            scalar lkp1cn_0_3 = group->kp1cn_0_3[lid];
            scalar lkp2_0_3 = group->kp2_0_3[lid];
            scalar lkp1cn_0_4 = group->kp1cn_0_4[lid];
            scalar lkp2_0_4 = group->kp2_0_4[lid];
            scalar lkp1cn_0_5 = group->kp1cn_0_5[lid];
            scalar lkp2_0_5 = group->kp2_0_5[lid];
            scalar lkp1cn_0_6 = group->kp1cn_0_6[lid];
            scalar lkp2_0_6 = group->kp2_0_6[lid];
            scalar lkp1cn_0_7 = group->kp1cn_0_7[lid];
            scalar lkp2_0_7 = group->kp2_0_7[lid];
            scalar lkp1cn_0_8 = group->kp1cn_0_8[lid];
            scalar lkp2_0_8 = group->kp2_0_8[lid];
            scalar lkp1cn_0_9 = group->kp1cn_0_9[lid];
            scalar lkp2_0_9 = group->kp2_0_9[lid];
            scalar lkp1cn_0_10 = group->kp1cn_0_10[lid];
            scalar lkp2_0_10 = group->kp2_0_10[lid];
            scalar lkp1cn_0_11 = group->kp1cn_0_11[lid];
            scalar lkp2_0_11 = group->kp2_0_11[lid];
            scalar lkp1cn_0_12 = group->kp1cn_0_12[lid];
            scalar lkp2_0_12 = group->kp2_0_12[lid];
            scalar lkp1cn_0_13 = group->kp1cn_0_13[lid];
            scalar lkp2_0_13 = group->kp2_0_13[lid];
            scalar lkp1cn_0_14 = group->kp1cn_0_14[lid];
            scalar lkp2_0_14 = group->kp2_0_14[lid];
            scalar lkp1cn_0_15 = group->kp1cn_0_15[lid];
            scalar lkp2_0_15 = group->kp2_0_15[lid];
            scalar lkp1cn_0_16 = group->kp1cn_0_16[lid];
            scalar lkp2_0_16 = group->kp2_0_16[lid];
            scalar lkp1cn_0_17 = group->kp1cn_0_17[lid];
            scalar lkp2_0_17 = group->kp2_0_17[lid];
            scalar lkp1cn_0_18 = group->kp1cn_0_18[lid];
            scalar lkp2_0_18 = group->kp2_0_18[lid];
            scalar lkp1cn_0_19 = group->kp1cn_0_19[lid];
            scalar lkp2_0_19 = group->kp2_0_19[lid];
            scalar lkp1cn_0_20 = group->kp1cn_0_20[lid];
            scalar lkp2_0_20 = group->kp2_0_20[lid];
            scalar lkp1cn_0_21 = group->kp1cn_0_21[lid];
            scalar lkp2_0_21 = group->kp2_0_21[lid];
            scalar lkp1cn_0_22 = group->kp1cn_0_22[lid];
            scalar lkp2_0_22 = group->kp2_0_22[lid];
            scalar lkp1cn_0_23 = group->kp1cn_0_23[lid];
            scalar lkp2_0_23 = group->kp2_0_23[lid];
            scalar lkp1cn_0_24 = group->kp1cn_0_24[lid];
            scalar lkp2_0_24 = group->kp2_0_24[lid];
            scalar lkp1cn_0_25 = group->kp1cn_0_25[lid];
            scalar lkp2_0_25 = group->kp2_0_25[lid];
            scalar lkp1cn_0_26 = group->kp1cn_0_26[lid];
            scalar lkp2_0_26 = group->kp2_0_26[lid];
            scalar lkp1cn_0_27 = group->kp1cn_0_27[lid];
            scalar lkp2_0_27 = group->kp2_0_27[lid];
            scalar lkp1cn_0_28 = group->kp1cn_0_28[lid];
            scalar lkp2_0_28 = group->kp2_0_28[lid];
            scalar lkp1cn_0_29 = group->kp1cn_0_29[lid];
            scalar lkp2_0_29 = group->kp2_0_29[lid];
            scalar lkp1cn_0_30 = group->kp1cn_0_30[lid];
            scalar lkp2_0_30 = group->kp2_0_30[lid];
            scalar lkp1cn_0_31 = group->kp1cn_0_31[lid];
            scalar lkp2_0_31 = group->kp2_0_31[lid];
            scalar lkp1cn_0_32 = group->kp1cn_0_32[lid];
            scalar lkp2_0_32 = group->kp2_0_32[lid];
            scalar lkp1cn_0_33 = group->kp1cn_0_33[lid];
            scalar lkp2_0_33 = group->kp2_0_33[lid];
            scalar lkp1cn_0_34 = group->kp1cn_0_34[lid];
            scalar lkp2_0_34 = group->kp2_0_34[lid];
            scalar lkp1cn_0_35 = group->kp1cn_0_35[lid];
            scalar lkp2_0_35 = group->kp2_0_35[lid];
            scalar lkp1cn_0_36 = group->kp1cn_0_36[lid];
            scalar lkp2_0_36 = group->kp2_0_36[lid];
            scalar lkp1cn_0_37 = group->kp1cn_0_37[lid];
            scalar lkp2_0_37 = group->kp2_0_37[lid];
            scalar lkp1cn_0_38 = group->kp1cn_0_38[lid];
            scalar lkp2_0_38 = group->kp2_0_38[lid];
            scalar lkp1cn_0_39 = group->kp1cn_0_39[lid];
            scalar lkp2_0_39 = group->kp2_0_39[lid];
            scalar lkp1cn_0_40 = group->kp1cn_0_40[lid];
            scalar lkp2_0_40 = group->kp2_0_40[lid];
            scalar lkp1cn_0_41 = group->kp1cn_0_41[lid];
            scalar lkp2_0_41 = group->kp2_0_41[lid];
            scalar lkp1cn_0_42 = group->kp1cn_0_42[lid];
            scalar lkp2_0_42 = group->kp2_0_42[lid];
            scalar lkp1cn_0_43 = group->kp1cn_0_43[lid];
            scalar lkp2_0_43 = group->kp2_0_43[lid];
            scalar lkp1cn_0_44 = group->kp1cn_0_44[lid];
            scalar lkp2_0_44 = group->kp2_0_44[lid];
            scalar lkp1cn_0_45 = group->kp1cn_0_45[lid];
            scalar lkp2_0_45 = group->kp2_0_45[lid];
            scalar lkp1cn_0_46 = group->kp1cn_0_46[lid];
            scalar lkp2_0_46 = group->kp2_0_46[lid];
            scalar lkp1cn_0_47 = group->kp1cn_0_47[lid];
            scalar lkp2_0_47 = group->kp2_0_47[lid];
            scalar lkp1cn_0_48 = group->kp1cn_0_48[lid];
            scalar lkp2_0_48 = group->kp2_0_48[lid];
            scalar lkp1cn_0_49 = group->kp1cn_0_49[lid];
            scalar lkp2_0_49 = group->kp2_0_49[lid];
            scalar lkp1cn_0_50 = group->kp1cn_0_50[lid];
            scalar lkp2_0_50 = group->kp2_0_50[lid];
            scalar lkp1cn_0_51 = group->kp1cn_0_51[lid];
            scalar lkp2_0_51 = group->kp2_0_51[lid];
            scalar lkp1cn_0_52 = group->kp1cn_0_52[lid];
            scalar lkp2_0_52 = group->kp2_0_52[lid];
            scalar lkp1cn_0_53 = group->kp1cn_0_53[lid];
            scalar lkp2_0_53 = group->kp2_0_53[lid];
            scalar lkp1cn_0_54 = group->kp1cn_0_54[lid];
            scalar lkp2_0_54 = group->kp2_0_54[lid];
            scalar lkp1cn_0_55 = group->kp1cn_0_55[lid];
            scalar lkp2_0_55 = group->kp2_0_55[lid];
            scalar lkp1cn_0_56 = group->kp1cn_0_56[lid];
            scalar lkp2_0_56 = group->kp2_0_56[lid];
            scalar lkp1cn_0_57 = group->kp1cn_0_57[lid];
            scalar lkp2_0_57 = group->kp2_0_57[lid];
            scalar lkp1cn_0_58 = group->kp1cn_0_58[lid];
            scalar lkp2_0_58 = group->kp2_0_58[lid];
            scalar lkp1cn_0_59 = group->kp1cn_0_59[lid];
            scalar lkp2_0_59 = group->kp2_0_59[lid];
            scalar lkp1cn_0_60 = group->kp1cn_0_60[lid];
            scalar lkp2_0_60 = group->kp2_0_60[lid];
            scalar lkp1cn_0_61 = group->kp1cn_0_61[lid];
            scalar lkp2_0_61 = group->kp2_0_61[lid];
            scalar lkp1cn_0_62 = group->kp1cn_0_62[lid];
            scalar lkp2_0_62 = group->kp2_0_62[lid];
            scalar lkp1cn_0_63 = group->kp1cn_0_63[lid];
            scalar lkp2_0_63 = group->kp2_0_63[lid];
            scalar lkp1cn_0_64 = group->kp1cn_0_64[lid];
            scalar lkp2_0_64 = group->kp2_0_64[lid];
            scalar lkp1cn_0_65 = group->kp1cn_0_65[lid];
            scalar lkp2_0_65 = group->kp2_0_65[lid];
            scalar lkp1cn_0_66 = group->kp1cn_0_66[lid];
            scalar lkp2_0_66 = group->kp2_0_66[lid];
            scalar lkp1cn_0_67 = group->kp1cn_0_67[lid];
            scalar lkp2_0_67 = group->kp2_0_67[lid];
            scalar lkp1cn_0_68 = group->kp1cn_0_68[lid];
            scalar lkp2_0_68 = group->kp2_0_68[lid];
            scalar lkp1cn_0_69 = group->kp1cn_0_69[lid];
            scalar lkp2_0_69 = group->kp2_0_69[lid];
            scalar lkp1cn_0_70 = group->kp1cn_0_70[lid];
            scalar lkp2_0_70 = group->kp2_0_70[lid];
            scalar lkp1cn_0_71 = group->kp1cn_0_71[lid];
            scalar lkp2_0_71 = group->kp2_0_71[lid];
            scalar lkp1cn_0_72 = group->kp1cn_0_72[lid];
            scalar lkp2_0_72 = group->kp2_0_72[lid];
            scalar lkp1cn_0_73 = group->kp1cn_0_73[lid];
            scalar lkp2_0_73 = group->kp2_0_73[lid];
            scalar lkp1cn_0_74 = group->kp1cn_0_74[lid];
            scalar lkp2_0_74 = group->kp2_0_74[lid];
            scalar lkp1cn_0_75 = group->kp1cn_0_75[lid];
            scalar lkp2_0_75 = group->kp2_0_75[lid];
            scalar lkp1cn_0_76 = group->kp1cn_0_76[lid];
            scalar lkp2_0_76 = group->kp2_0_76[lid];
            scalar lkp1cn_0_77 = group->kp1cn_0_77[lid];
            scalar lkp2_0_77 = group->kp2_0_77[lid];
            scalar lkp1cn_0_78 = group->kp1cn_0_78[lid];
            scalar lkp2_0_78 = group->kp2_0_78[lid];
            scalar lkp1cn_0_79 = group->kp1cn_0_79[lid];
            scalar lkp2_0_79 = group->kp2_0_79[lid];
            scalar lkp1cn_0_80 = group->kp1cn_0_80[lid];
            scalar lkp2_0_80 = group->kp2_0_80[lid];
            scalar lkp1cn_0_81 = group->kp1cn_0_81[lid];
            scalar lkp2_0_81 = group->kp2_0_81[lid];
            scalar lkp1cn_0_82 = group->kp1cn_0_82[lid];
            scalar lkp2_0_82 = group->kp2_0_82[lid];
            scalar lkp1cn_0_83 = group->kp1cn_0_83[lid];
            scalar lkp2_0_83 = group->kp2_0_83[lid];
            scalar lkp1cn_0_84 = group->kp1cn_0_84[lid];
            scalar lkp2_0_84 = group->kp2_0_84[lid];
            scalar lkp1cn_0_85 = group->kp1cn_0_85[lid];
            scalar lkp2_0_85 = group->kp2_0_85[lid];
            scalar lkp1cn_0_86 = group->kp1cn_0_86[lid];
            scalar lkp2_0_86 = group->kp2_0_86[lid];
            scalar lkp1cn_0_87 = group->kp1cn_0_87[lid];
            scalar lkp2_0_87 = group->kp2_0_87[lid];
            scalar lkp1cn_0_88 = group->kp1cn_0_88[lid];
            scalar lkp2_0_88 = group->kp2_0_88[lid];
            scalar lkp1cn_0_89 = group->kp1cn_0_89[lid];
            scalar lkp2_0_89 = group->kp2_0_89[lid];
            scalar lkp1cn_0_90 = group->kp1cn_0_90[lid];
            scalar lkp2_0_90 = group->kp2_0_90[lid];
            scalar lkp1cn_0_91 = group->kp1cn_0_91[lid];
            scalar lkp2_0_91 = group->kp2_0_91[lid];
            scalar lkp1cn_0_92 = group->kp1cn_0_92[lid];
            scalar lkp2_0_92 = group->kp2_0_92[lid];
            scalar lkp1cn_0_93 = group->kp1cn_0_93[lid];
            scalar lkp2_0_93 = group->kp2_0_93[lid];
            scalar lkp1cn_0_94 = group->kp1cn_0_94[lid];
            scalar lkp2_0_94 = group->kp2_0_94[lid];
            scalar lkp1cn_0_95 = group->kp1cn_0_95[lid];
            scalar lkp2_0_95 = group->kp2_0_95[lid];
            scalar lkp1cn_0_96 = group->kp1cn_0_96[lid];
            scalar lkp2_0_96 = group->kp2_0_96[lid];
            scalar lkp1cn_0_97 = group->kp1cn_0_97[lid];
            scalar lkp2_0_97 = group->kp2_0_97[lid];
            scalar lkp1cn_0_98 = group->kp1cn_0_98[lid];
            scalar lkp2_0_98 = group->kp2_0_98[lid];
            scalar lkp1cn_0_99 = group->kp1cn_0_99[lid];
            scalar lkp2_0_99 = group->kp2_0_99[lid];
            
            // calculate membrane potential
            if (t >= 3000.0 && t <= 6000.0) {lkp1cn_0 = lkp1cn_0_0; lkp2_0 = lkp2_0_0; } else if (t >= 9000.0 && t <= 12000.0) {lkp1cn_0 = lkp1cn_0_1; lkp2_0 = lkp2_0_1; } else if (t >= 15000.0 && t <= 18000.0) {lkp1cn_0 = lkp1cn_0_2; lkp2_0 = lkp2_0_2; } else if (t >= 21000.0 && t <= 24000.0) {lkp1cn_0 = lkp1cn_0_3; lkp2_0 = lkp2_0_3; } else if (t >= 27000.0 && t <= 30000.0) {lkp1cn_0 = lkp1cn_0_4; lkp2_0 = lkp2_0_4; } else if (t >= 33000.0 && t <= 36000.0) {lkp1cn_0 = lkp1cn_0_5; lkp2_0 = lkp2_0_5; } else if (t >= 39000.0 && t <= 42000.0) {lkp1cn_0 = lkp1cn_0_6; lkp2_0 = lkp2_0_6; } else if (t >= 45000.0 && t <= 48000.0) {lkp1cn_0 = lkp1cn_0_7; lkp2_0 = lkp2_0_7; } else if (t >= 51000.0 && t <= 54000.0) {lkp1cn_0 = lkp1cn_0_8; lkp2_0 = lkp2_0_8; } else if (t >= 57000.0 && t <= 60000.0) {lkp1cn_0 = lkp1cn_0_9; lkp2_0 = lkp2_0_9; } else if (t >= 63000.0 && t <= 66000.0) {lkp1cn_0 = lkp1cn_0_10; lkp2_0 = lkp2_0_10; } else if (t >= 69000.0 && t <= 72000.0) {lkp1cn_0 = lkp1cn_0_11; lkp2_0 = lkp2_0_11; } else if (t >= 75000.0 && t <= 78000.0) {lkp1cn_0 = lkp1cn_0_12; lkp2_0 = lkp2_0_12; } else if (t >= 81000.0 && t <= 84000.0) {lkp1cn_0 = lkp1cn_0_13; lkp2_0 = lkp2_0_13; } else if (t >= 87000.0 && t <= 90000.0) {lkp1cn_0 = lkp1cn_0_14; lkp2_0 = lkp2_0_14; } else if (t >= 93000.0 && t <= 96000.0) {lkp1cn_0 = lkp1cn_0_15; lkp2_0 = lkp2_0_15; } else if (t >= 99000.0 && t <= 102000.0) {lkp1cn_0 = lkp1cn_0_16; lkp2_0 = lkp2_0_16; } else if (t >= 105000.0 && t <= 108000.0) {lkp1cn_0 = lkp1cn_0_17; lkp2_0 = lkp2_0_17; } else if (t >= 111000.0 && t <= 114000.0) {lkp1cn_0 = lkp1cn_0_18; lkp2_0 = lkp2_0_18; } else if (t >= 117000.0 && t <= 120000.0) {lkp1cn_0 = lkp1cn_0_19; lkp2_0 = lkp2_0_19; } else if (t >= 123000.0 && t <= 126000.0) {lkp1cn_0 = lkp1cn_0_20; lkp2_0 = lkp2_0_20; } else if (t >= 129000.0 && t <= 132000.0) {lkp1cn_0 = lkp1cn_0_21; lkp2_0 = lkp2_0_21; } else if (t >= 135000.0 && t <= 138000.0) {lkp1cn_0 = lkp1cn_0_22; lkp2_0 = lkp2_0_22; } else if (t >= 141000.0 && t <= 144000.0) {lkp1cn_0 = lkp1cn_0_23; lkp2_0 = lkp2_0_23; } else if (t >= 147000.0 && t <= 150000.0) {lkp1cn_0 = lkp1cn_0_24; lkp2_0 = lkp2_0_24; } else if (t >= 153000.0 && t <= 156000.0) {lkp1cn_0 = lkp1cn_0_25; lkp2_0 = lkp2_0_25; } else if (t >= 159000.0 && t <= 162000.0) {lkp1cn_0 = lkp1cn_0_26; lkp2_0 = lkp2_0_26; } else if (t >= 165000.0 && t <= 168000.0) {lkp1cn_0 = lkp1cn_0_27; lkp2_0 = lkp2_0_27; } else if (t >= 171000.0 && t <= 174000.0) {lkp1cn_0 = lkp1cn_0_28; lkp2_0 = lkp2_0_28; } else if (t >= 177000.0 && t <= 180000.0) {lkp1cn_0 = lkp1cn_0_29; lkp2_0 = lkp2_0_29; } else if (t >= 183000.0 && t <= 186000.0) {lkp1cn_0 = lkp1cn_0_30; lkp2_0 = lkp2_0_30; } else if (t >= 189000.0 && t <= 192000.0) {lkp1cn_0 = lkp1cn_0_31; lkp2_0 = lkp2_0_31; } else if (t >= 195000.0 && t <= 198000.0) {lkp1cn_0 = lkp1cn_0_32; lkp2_0 = lkp2_0_32; } else if (t >= 201000.0 && t <= 204000.0) {lkp1cn_0 = lkp1cn_0_33; lkp2_0 = lkp2_0_33; } else if (t >= 207000.0 && t <= 210000.0) {lkp1cn_0 = lkp1cn_0_34; lkp2_0 = lkp2_0_34; } else if (t >= 213000.0 && t <= 216000.0) {lkp1cn_0 = lkp1cn_0_35; lkp2_0 = lkp2_0_35; } else if (t >= 219000.0 && t <= 222000.0) {lkp1cn_0 = lkp1cn_0_36; lkp2_0 = lkp2_0_36; } else if (t >= 225000.0 && t <= 228000.0) {lkp1cn_0 = lkp1cn_0_37; lkp2_0 = lkp2_0_37; } else if (t >= 231000.0 && t <= 234000.0) {lkp1cn_0 = lkp1cn_0_38; lkp2_0 = lkp2_0_38; } else if (t >= 237000.0 && t <= 240000.0) {lkp1cn_0 = lkp1cn_0_39; lkp2_0 = lkp2_0_39; } else if (t >= 243000.0 && t <= 246000.0) {lkp1cn_0 = lkp1cn_0_40; lkp2_0 = lkp2_0_40; } else if (t >= 249000.0 && t <= 252000.0) {lkp1cn_0 = lkp1cn_0_41; lkp2_0 = lkp2_0_41; } else if (t >= 255000.0 && t <= 258000.0) {lkp1cn_0 = lkp1cn_0_42; lkp2_0 = lkp2_0_42; } else if (t >= 261000.0 && t <= 264000.0) {lkp1cn_0 = lkp1cn_0_43; lkp2_0 = lkp2_0_43; } else if (t >= 267000.0 && t <= 270000.0) {lkp1cn_0 = lkp1cn_0_44; lkp2_0 = lkp2_0_44; } else if (t >= 273000.0 && t <= 276000.0) {lkp1cn_0 = lkp1cn_0_45; lkp2_0 = lkp2_0_45; } else if (t >= 279000.0 && t <= 282000.0) {lkp1cn_0 = lkp1cn_0_46; lkp2_0 = lkp2_0_46; } else if (t >= 285000.0 && t <= 288000.0) {lkp1cn_0 = lkp1cn_0_47; lkp2_0 = lkp2_0_47; } else if (t >= 291000.0 && t <= 294000.0) {lkp1cn_0 = lkp1cn_0_48; lkp2_0 = lkp2_0_48; } else if (t >= 297000.0 && t <= 300000.0) {lkp1cn_0 = lkp1cn_0_49; lkp2_0 = lkp2_0_49; } else if (t >= 303000.0 && t <= 306000.0) {lkp1cn_0 = lkp1cn_0_50; lkp2_0 = lkp2_0_50; } else if (t >= 309000.0 && t <= 312000.0) {lkp1cn_0 = lkp1cn_0_51; lkp2_0 = lkp2_0_51; } else if (t >= 315000.0 && t <= 318000.0) {lkp1cn_0 = lkp1cn_0_52; lkp2_0 = lkp2_0_52; } else if (t >= 321000.0 && t <= 324000.0) {lkp1cn_0 = lkp1cn_0_53; lkp2_0 = lkp2_0_53; } else if (t >= 327000.0 && t <= 330000.0) {lkp1cn_0 = lkp1cn_0_54; lkp2_0 = lkp2_0_54; } else if (t >= 333000.0 && t <= 336000.0) {lkp1cn_0 = lkp1cn_0_55; lkp2_0 = lkp2_0_55; } else if (t >= 339000.0 && t <= 342000.0) {lkp1cn_0 = lkp1cn_0_56; lkp2_0 = lkp2_0_56; } else if (t >= 345000.0 && t <= 348000.0) {lkp1cn_0 = lkp1cn_0_57; lkp2_0 = lkp2_0_57; } else if (t >= 351000.0 && t <= 354000.0) {lkp1cn_0 = lkp1cn_0_58; lkp2_0 = lkp2_0_58; } else if (t >= 357000.0 && t <= 360000.0) {lkp1cn_0 = lkp1cn_0_59; lkp2_0 = lkp2_0_59; } else if (t >= 363000.0 && t <= 366000.0) {lkp1cn_0 = lkp1cn_0_60; lkp2_0 = lkp2_0_60; } else if (t >= 369000.0 && t <= 372000.0) {lkp1cn_0 = lkp1cn_0_61; lkp2_0 = lkp2_0_61; } else if (t >= 375000.0 && t <= 378000.0) {lkp1cn_0 = lkp1cn_0_62; lkp2_0 = lkp2_0_62; } else if (t >= 381000.0 && t <= 384000.0) {lkp1cn_0 = lkp1cn_0_63; lkp2_0 = lkp2_0_63; } else if (t >= 387000.0 && t <= 390000.0) {lkp1cn_0 = lkp1cn_0_64; lkp2_0 = lkp2_0_64; } else if (t >= 393000.0 && t <= 396000.0) {lkp1cn_0 = lkp1cn_0_65; lkp2_0 = lkp2_0_65; } else if (t >= 399000.0 && t <= 402000.0) {lkp1cn_0 = lkp1cn_0_66; lkp2_0 = lkp2_0_66; } else if (t >= 405000.0 && t <= 408000.0) {lkp1cn_0 = lkp1cn_0_67; lkp2_0 = lkp2_0_67; } else if (t >= 411000.0 && t <= 414000.0) {lkp1cn_0 = lkp1cn_0_68; lkp2_0 = lkp2_0_68; } else if (t >= 417000.0 && t <= 420000.0) {lkp1cn_0 = lkp1cn_0_69; lkp2_0 = lkp2_0_69; } else if (t >= 423000.0 && t <= 426000.0) {lkp1cn_0 = lkp1cn_0_70; lkp2_0 = lkp2_0_70; } else if (t >= 429000.0 && t <= 432000.0) {lkp1cn_0 = lkp1cn_0_71; lkp2_0 = lkp2_0_71; } else if (t >= 435000.0 && t <= 438000.0) {lkp1cn_0 = lkp1cn_0_72; lkp2_0 = lkp2_0_72; } else if (t >= 441000.0 && t <= 444000.0) {lkp1cn_0 = lkp1cn_0_73; lkp2_0 = lkp2_0_73; } else if (t >= 447000.0 && t <= 450000.0) {lkp1cn_0 = lkp1cn_0_74; lkp2_0 = lkp2_0_74; } else if (t >= 453000.0 && t <= 456000.0) {lkp1cn_0 = lkp1cn_0_75; lkp2_0 = lkp2_0_75; } else if (t >= 459000.0 && t <= 462000.0) {lkp1cn_0 = lkp1cn_0_76; lkp2_0 = lkp2_0_76; } else if (t >= 465000.0 && t <= 468000.0) {lkp1cn_0 = lkp1cn_0_77; lkp2_0 = lkp2_0_77; } else if (t >= 471000.0 && t <= 474000.0) {lkp1cn_0 = lkp1cn_0_78; lkp2_0 = lkp2_0_78; } else if (t >= 477000.0 && t <= 480000.0) {lkp1cn_0 = lkp1cn_0_79; lkp2_0 = lkp2_0_79; } else if (t >= 483000.0 && t <= 486000.0) {lkp1cn_0 = lkp1cn_0_80; lkp2_0 = lkp2_0_80; } else if (t >= 489000.0 && t <= 492000.0) {lkp1cn_0 = lkp1cn_0_81; lkp2_0 = lkp2_0_81; } else if (t >= 495000.0 && t <= 498000.0) {lkp1cn_0 = lkp1cn_0_82; lkp2_0 = lkp2_0_82; } else if (t >= 501000.0 && t <= 504000.0) {lkp1cn_0 = lkp1cn_0_83; lkp2_0 = lkp2_0_83; } else if (t >= 507000.0 && t <= 510000.0) {lkp1cn_0 = lkp1cn_0_84; lkp2_0 = lkp2_0_84; } else if (t >= 513000.0 && t <= 516000.0) {lkp1cn_0 = lkp1cn_0_85; lkp2_0 = lkp2_0_85; } else if (t >= 519000.0 && t <= 522000.0) {lkp1cn_0 = lkp1cn_0_86; lkp2_0 = lkp2_0_86; } else if (t >= 525000.0 && t <= 528000.0) {lkp1cn_0 = lkp1cn_0_87; lkp2_0 = lkp2_0_87; } else if (t >= 531000.0 && t <= 534000.0) {lkp1cn_0 = lkp1cn_0_88; lkp2_0 = lkp2_0_88; } else if (t >= 537000.0 && t <= 540000.0) {lkp1cn_0 = lkp1cn_0_89; lkp2_0 = lkp2_0_89; } else if (t >= 543000.0 && t <= 546000.0) {lkp1cn_0 = lkp1cn_0_90; lkp2_0 = lkp2_0_90; } else if (t >= 549000.0 && t <= 552000.0) {lkp1cn_0 = lkp1cn_0_91; lkp2_0 = lkp2_0_91; } else if (t >= 555000.0 && t <= 558000.0) {lkp1cn_0 = lkp1cn_0_92; lkp2_0 = lkp2_0_92; } else if (t >= 561000.0 && t <= 564000.0) {lkp1cn_0 = lkp1cn_0_93; lkp2_0 = lkp2_0_93; } else if (t >= 567000.0 && t <= 570000.0) {lkp1cn_0 = lkp1cn_0_94; lkp2_0 = lkp2_0_94; } else if (t >= 573000.0 && t <= 576000.0) {lkp1cn_0 = lkp1cn_0_95; lkp2_0 = lkp2_0_95; } else if (t >= 579000.0 && t <= 582000.0) {lkp1cn_0 = lkp1cn_0_96; lkp2_0 = lkp2_0_96; } else if (t >= 585000.0 && t <= 588000.0) {lkp1cn_0 = lkp1cn_0_97; lkp2_0 = lkp2_0_97; } else if (t >= 591000.0 && t <= 594000.0) {lkp1cn_0 = lkp1cn_0_98; lkp2_0 = lkp2_0_98; } else if (t >= 597000.0 && t <= 600000.0) {lkp1cn_0 = lkp1cn_0_99; lkp2_0 = lkp2_0_99; } else {lkp1cn_0 = 0;}
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
            group->kp1cn_0_0[lid] = lkp1cn_0_0;
            group->kp2_0_0[lid] = lkp2_0_0;
            group->kp1cn_0_1[lid] = lkp1cn_0_1;
            group->kp2_0_1[lid] = lkp2_0_1;
            group->kp1cn_0_2[lid] = lkp1cn_0_2;
            group->kp2_0_2[lid] = lkp2_0_2;
            group->kp1cn_0_3[lid] = lkp1cn_0_3;
            group->kp2_0_3[lid] = lkp2_0_3;
            group->kp1cn_0_4[lid] = lkp1cn_0_4;
            group->kp2_0_4[lid] = lkp2_0_4;
            group->kp1cn_0_5[lid] = lkp1cn_0_5;
            group->kp2_0_5[lid] = lkp2_0_5;
            group->kp1cn_0_6[lid] = lkp1cn_0_6;
            group->kp2_0_6[lid] = lkp2_0_6;
            group->kp1cn_0_7[lid] = lkp1cn_0_7;
            group->kp2_0_7[lid] = lkp2_0_7;
            group->kp1cn_0_8[lid] = lkp1cn_0_8;
            group->kp2_0_8[lid] = lkp2_0_8;
            group->kp1cn_0_9[lid] = lkp1cn_0_9;
            group->kp2_0_9[lid] = lkp2_0_9;
            group->kp1cn_0_10[lid] = lkp1cn_0_10;
            group->kp2_0_10[lid] = lkp2_0_10;
            group->kp1cn_0_11[lid] = lkp1cn_0_11;
            group->kp2_0_11[lid] = lkp2_0_11;
            group->kp1cn_0_12[lid] = lkp1cn_0_12;
            group->kp2_0_12[lid] = lkp2_0_12;
            group->kp1cn_0_13[lid] = lkp1cn_0_13;
            group->kp2_0_13[lid] = lkp2_0_13;
            group->kp1cn_0_14[lid] = lkp1cn_0_14;
            group->kp2_0_14[lid] = lkp2_0_14;
            group->kp1cn_0_15[lid] = lkp1cn_0_15;
            group->kp2_0_15[lid] = lkp2_0_15;
            group->kp1cn_0_16[lid] = lkp1cn_0_16;
            group->kp2_0_16[lid] = lkp2_0_16;
            group->kp1cn_0_17[lid] = lkp1cn_0_17;
            group->kp2_0_17[lid] = lkp2_0_17;
            group->kp1cn_0_18[lid] = lkp1cn_0_18;
            group->kp2_0_18[lid] = lkp2_0_18;
            group->kp1cn_0_19[lid] = lkp1cn_0_19;
            group->kp2_0_19[lid] = lkp2_0_19;
            group->kp1cn_0_20[lid] = lkp1cn_0_20;
            group->kp2_0_20[lid] = lkp2_0_20;
            group->kp1cn_0_21[lid] = lkp1cn_0_21;
            group->kp2_0_21[lid] = lkp2_0_21;
            group->kp1cn_0_22[lid] = lkp1cn_0_22;
            group->kp2_0_22[lid] = lkp2_0_22;
            group->kp1cn_0_23[lid] = lkp1cn_0_23;
            group->kp2_0_23[lid] = lkp2_0_23;
            group->kp1cn_0_24[lid] = lkp1cn_0_24;
            group->kp2_0_24[lid] = lkp2_0_24;
            group->kp1cn_0_25[lid] = lkp1cn_0_25;
            group->kp2_0_25[lid] = lkp2_0_25;
            group->kp1cn_0_26[lid] = lkp1cn_0_26;
            group->kp2_0_26[lid] = lkp2_0_26;
            group->kp1cn_0_27[lid] = lkp1cn_0_27;
            group->kp2_0_27[lid] = lkp2_0_27;
            group->kp1cn_0_28[lid] = lkp1cn_0_28;
            group->kp2_0_28[lid] = lkp2_0_28;
            group->kp1cn_0_29[lid] = lkp1cn_0_29;
            group->kp2_0_29[lid] = lkp2_0_29;
            group->kp1cn_0_30[lid] = lkp1cn_0_30;
            group->kp2_0_30[lid] = lkp2_0_30;
            group->kp1cn_0_31[lid] = lkp1cn_0_31;
            group->kp2_0_31[lid] = lkp2_0_31;
            group->kp1cn_0_32[lid] = lkp1cn_0_32;
            group->kp2_0_32[lid] = lkp2_0_32;
            group->kp1cn_0_33[lid] = lkp1cn_0_33;
            group->kp2_0_33[lid] = lkp2_0_33;
            group->kp1cn_0_34[lid] = lkp1cn_0_34;
            group->kp2_0_34[lid] = lkp2_0_34;
            group->kp1cn_0_35[lid] = lkp1cn_0_35;
            group->kp2_0_35[lid] = lkp2_0_35;
            group->kp1cn_0_36[lid] = lkp1cn_0_36;
            group->kp2_0_36[lid] = lkp2_0_36;
            group->kp1cn_0_37[lid] = lkp1cn_0_37;
            group->kp2_0_37[lid] = lkp2_0_37;
            group->kp1cn_0_38[lid] = lkp1cn_0_38;
            group->kp2_0_38[lid] = lkp2_0_38;
            group->kp1cn_0_39[lid] = lkp1cn_0_39;
            group->kp2_0_39[lid] = lkp2_0_39;
            group->kp1cn_0_40[lid] = lkp1cn_0_40;
            group->kp2_0_40[lid] = lkp2_0_40;
            group->kp1cn_0_41[lid] = lkp1cn_0_41;
            group->kp2_0_41[lid] = lkp2_0_41;
            group->kp1cn_0_42[lid] = lkp1cn_0_42;
            group->kp2_0_42[lid] = lkp2_0_42;
            group->kp1cn_0_43[lid] = lkp1cn_0_43;
            group->kp2_0_43[lid] = lkp2_0_43;
            group->kp1cn_0_44[lid] = lkp1cn_0_44;
            group->kp2_0_44[lid] = lkp2_0_44;
            group->kp1cn_0_45[lid] = lkp1cn_0_45;
            group->kp2_0_45[lid] = lkp2_0_45;
            group->kp1cn_0_46[lid] = lkp1cn_0_46;
            group->kp2_0_46[lid] = lkp2_0_46;
            group->kp1cn_0_47[lid] = lkp1cn_0_47;
            group->kp2_0_47[lid] = lkp2_0_47;
            group->kp1cn_0_48[lid] = lkp1cn_0_48;
            group->kp2_0_48[lid] = lkp2_0_48;
            group->kp1cn_0_49[lid] = lkp1cn_0_49;
            group->kp2_0_49[lid] = lkp2_0_49;
            group->kp1cn_0_50[lid] = lkp1cn_0_50;
            group->kp2_0_50[lid] = lkp2_0_50;
            group->kp1cn_0_51[lid] = lkp1cn_0_51;
            group->kp2_0_51[lid] = lkp2_0_51;
            group->kp1cn_0_52[lid] = lkp1cn_0_52;
            group->kp2_0_52[lid] = lkp2_0_52;
            group->kp1cn_0_53[lid] = lkp1cn_0_53;
            group->kp2_0_53[lid] = lkp2_0_53;
            group->kp1cn_0_54[lid] = lkp1cn_0_54;
            group->kp2_0_54[lid] = lkp2_0_54;
            group->kp1cn_0_55[lid] = lkp1cn_0_55;
            group->kp2_0_55[lid] = lkp2_0_55;
            group->kp1cn_0_56[lid] = lkp1cn_0_56;
            group->kp2_0_56[lid] = lkp2_0_56;
            group->kp1cn_0_57[lid] = lkp1cn_0_57;
            group->kp2_0_57[lid] = lkp2_0_57;
            group->kp1cn_0_58[lid] = lkp1cn_0_58;
            group->kp2_0_58[lid] = lkp2_0_58;
            group->kp1cn_0_59[lid] = lkp1cn_0_59;
            group->kp2_0_59[lid] = lkp2_0_59;
            group->kp1cn_0_60[lid] = lkp1cn_0_60;
            group->kp2_0_60[lid] = lkp2_0_60;
            group->kp1cn_0_61[lid] = lkp1cn_0_61;
            group->kp2_0_61[lid] = lkp2_0_61;
            group->kp1cn_0_62[lid] = lkp1cn_0_62;
            group->kp2_0_62[lid] = lkp2_0_62;
            group->kp1cn_0_63[lid] = lkp1cn_0_63;
            group->kp2_0_63[lid] = lkp2_0_63;
            group->kp1cn_0_64[lid] = lkp1cn_0_64;
            group->kp2_0_64[lid] = lkp2_0_64;
            group->kp1cn_0_65[lid] = lkp1cn_0_65;
            group->kp2_0_65[lid] = lkp2_0_65;
            group->kp1cn_0_66[lid] = lkp1cn_0_66;
            group->kp2_0_66[lid] = lkp2_0_66;
            group->kp1cn_0_67[lid] = lkp1cn_0_67;
            group->kp2_0_67[lid] = lkp2_0_67;
            group->kp1cn_0_68[lid] = lkp1cn_0_68;
            group->kp2_0_68[lid] = lkp2_0_68;
            group->kp1cn_0_69[lid] = lkp1cn_0_69;
            group->kp2_0_69[lid] = lkp2_0_69;
            group->kp1cn_0_70[lid] = lkp1cn_0_70;
            group->kp2_0_70[lid] = lkp2_0_70;
            group->kp1cn_0_71[lid] = lkp1cn_0_71;
            group->kp2_0_71[lid] = lkp2_0_71;
            group->kp1cn_0_72[lid] = lkp1cn_0_72;
            group->kp2_0_72[lid] = lkp2_0_72;
            group->kp1cn_0_73[lid] = lkp1cn_0_73;
            group->kp2_0_73[lid] = lkp2_0_73;
            group->kp1cn_0_74[lid] = lkp1cn_0_74;
            group->kp2_0_74[lid] = lkp2_0_74;
            group->kp1cn_0_75[lid] = lkp1cn_0_75;
            group->kp2_0_75[lid] = lkp2_0_75;
            group->kp1cn_0_76[lid] = lkp1cn_0_76;
            group->kp2_0_76[lid] = lkp2_0_76;
            group->kp1cn_0_77[lid] = lkp1cn_0_77;
            group->kp2_0_77[lid] = lkp2_0_77;
            group->kp1cn_0_78[lid] = lkp1cn_0_78;
            group->kp2_0_78[lid] = lkp2_0_78;
            group->kp1cn_0_79[lid] = lkp1cn_0_79;
            group->kp2_0_79[lid] = lkp2_0_79;
            group->kp1cn_0_80[lid] = lkp1cn_0_80;
            group->kp2_0_80[lid] = lkp2_0_80;
            group->kp1cn_0_81[lid] = lkp1cn_0_81;
            group->kp2_0_81[lid] = lkp2_0_81;
            group->kp1cn_0_82[lid] = lkp1cn_0_82;
            group->kp2_0_82[lid] = lkp2_0_82;
            group->kp1cn_0_83[lid] = lkp1cn_0_83;
            group->kp2_0_83[lid] = lkp2_0_83;
            group->kp1cn_0_84[lid] = lkp1cn_0_84;
            group->kp2_0_84[lid] = lkp2_0_84;
            group->kp1cn_0_85[lid] = lkp1cn_0_85;
            group->kp2_0_85[lid] = lkp2_0_85;
            group->kp1cn_0_86[lid] = lkp1cn_0_86;
            group->kp2_0_86[lid] = lkp2_0_86;
            group->kp1cn_0_87[lid] = lkp1cn_0_87;
            group->kp2_0_87[lid] = lkp2_0_87;
            group->kp1cn_0_88[lid] = lkp1cn_0_88;
            group->kp2_0_88[lid] = lkp2_0_88;
            group->kp1cn_0_89[lid] = lkp1cn_0_89;
            group->kp2_0_89[lid] = lkp2_0_89;
            group->kp1cn_0_90[lid] = lkp1cn_0_90;
            group->kp2_0_90[lid] = lkp2_0_90;
            group->kp1cn_0_91[lid] = lkp1cn_0_91;
            group->kp2_0_91[lid] = lkp2_0_91;
            group->kp1cn_0_92[lid] = lkp1cn_0_92;
            group->kp2_0_92[lid] = lkp2_0_92;
            group->kp1cn_0_93[lid] = lkp1cn_0_93;
            group->kp2_0_93[lid] = lkp2_0_93;
            group->kp1cn_0_94[lid] = lkp1cn_0_94;
            group->kp2_0_94[lid] = lkp2_0_94;
            group->kp1cn_0_95[lid] = lkp1cn_0_95;
            group->kp2_0_95[lid] = lkp2_0_95;
            group->kp1cn_0_96[lid] = lkp1cn_0_96;
            group->kp2_0_96[lid] = lkp2_0_96;
            group->kp1cn_0_97[lid] = lkp1cn_0_97;
            group->kp2_0_97[lid] = lkp2_0_97;
            group->kp1cn_0_98[lid] = lkp1cn_0_98;
            group->kp2_0_98[lid] = lkp2_0_98;
            group->kp1cn_0_99[lid] = lkp1cn_0_99;
            group->kp2_0_99[lid] = lkp2_0_99;
        }
        __syncthreads();
    }
    // merged2
    if(id >= 192 && id < 224) {
        struct MergedNeuronUpdateGroup2 *group = &d_mergedNeuronUpdateGroup2[0]; 
        const unsigned int lid = id - 192;
        
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
    }
    // merged3
    if(id >= 224 && id < 256) {
        struct MergedNeuronUpdateGroup3 *group = &d_mergedNeuronUpdateGroup3[0]; 
        const unsigned int lid = id - 224;
        
        if(lid < group->numNeurons) {
            scalar lV = group->V[lid];
            scalar la = group->a[lid];
            
            double Isyn = 0;
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn0[lid];
                Isyn += linSyn * ((0.00000000000000000e+00) - lV);
                linSyn*=(9.90049833749168107e-01);
                group->inSynInSyn0[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn1[lid];
                Isyn += linSyn * ((0.00000000000000000e+00) - lV);
                linSyn*=(9.90049833749168107e-01);
                group->inSynInSyn1[lid] = linSyn;
            }
             {
                // pull inSyn values in a coalesced access
                double linSyn = group->inSynInSyn2[lid];
                Isyn += linSyn * ((-8.00000000000000000e+01) - lV);
                linSyn*=(9.95012479192682320e-01);
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
    }
}
void updateNeurons(double t) {
     {
        const dim3 threads(32, 1);
        const dim3 grid(1, 1);
        neuronSpikeQueueUpdateKernel<<<grid, threads>>>();
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
     {
        const dim3 threads(32, 1);
        const dim3 grid(8, 1);
        updateNeuronsKernel<<<grid, threads>>>(t);
        CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    }
}
