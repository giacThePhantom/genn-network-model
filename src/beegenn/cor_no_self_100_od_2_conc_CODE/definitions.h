#pragma once
#define EXPORT_VAR extern
#define EXPORT_FUNC
// Standard C++ includes
#include <random>
#include <string>
#include <stdexcept>

// Standard C includes
#include <cassert>
#include <cstdint>
#define DT 2.00000000000000011e-01
typedef double scalar;
#define SCALAR_MIN 2.22507385850720138e-308
#define SCALAR_MAX 1.79769313486231571e+308

#define TIME_MIN 2.22507385850720138e-308
#define TIME_MAX 1.79769313486231571e+308

// ------------------------------------------------------------------------
// bit tool macros
#define B(x,i) ((x) & (0x80000000 >> (i))) //!< Extract the bit at the specified position i from x
#define setB(x,i) x= ((x) | (0x80000000 >> (i))) //!< Set the bit at the specified position i in x to 1
#define delB(x,i) x= ((x) & (~(0x80000000 >> (i)))) //!< Set the bit at the specified position i in x to 0

extern "C" {
// ------------------------------------------------------------------------
// global variables
// ------------------------------------------------------------------------
EXPORT_VAR unsigned long long iT;
EXPORT_VAR double t;

// ------------------------------------------------------------------------
// timers
// ------------------------------------------------------------------------
EXPORT_VAR double initTime;
EXPORT_VAR double initSparseTime;
EXPORT_VAR double neuronUpdateTime;
EXPORT_VAR double presynapticUpdateTime;
EXPORT_VAR double postsynapticUpdateTime;
EXPORT_VAR double synapseDynamicsTime;
// ------------------------------------------------------------------------
// local neuron groups
// ------------------------------------------------------------------------
#define spikeCount_ln glbSpkCntln[0]
#define spike_ln glbSpkln
#define glbSpkShiftln 0

EXPORT_VAR unsigned int* glbSpkCntln;
EXPORT_VAR unsigned int* d_glbSpkCntln;
EXPORT_VAR unsigned int* glbSpkln;
EXPORT_VAR unsigned int* d_glbSpkln;
EXPORT_VAR uint32_t* recordSpkln;
EXPORT_VAR uint32_t* d_recordSpkln;
EXPORT_VAR scalar* Vln;
EXPORT_VAR scalar* d_Vln;
EXPORT_VAR scalar* aln;
EXPORT_VAR scalar* d_aln;
#define spikeCount_or glbSpkCntor[0]
#define spike_or glbSpkor
#define glbSpkShiftor 0

EXPORT_VAR unsigned int* glbSpkCntor;
EXPORT_VAR unsigned int* d_glbSpkCntor;
EXPORT_VAR unsigned int* glbSpkor;
EXPORT_VAR unsigned int* d_glbSpkor;
EXPORT_VAR scalar* r0or;
EXPORT_VAR scalar* d_r0or;
EXPORT_VAR scalar* rb_0or;
EXPORT_VAR scalar* d_rb_0or;
EXPORT_VAR scalar* ra_0or;
EXPORT_VAR scalar* d_ra_0or;
EXPORT_VAR scalar* rb_1or;
EXPORT_VAR scalar* d_rb_1or;
EXPORT_VAR scalar* ra_1or;
EXPORT_VAR scalar* d_ra_1or;
EXPORT_VAR scalar* rb_2or;
EXPORT_VAR scalar* d_rb_2or;
EXPORT_VAR scalar* ra_2or;
EXPORT_VAR scalar* d_ra_2or;
EXPORT_VAR scalar* raor;
EXPORT_VAR scalar* d_raor;
EXPORT_VAR scalar* kp1cn_0or;
EXPORT_VAR scalar* d_kp1cn_0or;
EXPORT_VAR scalar* km1_0or;
EXPORT_VAR scalar* d_km1_0or;
EXPORT_VAR scalar* kp2_0or;
EXPORT_VAR scalar* d_kp2_0or;
EXPORT_VAR scalar* km2_0or;
EXPORT_VAR scalar* d_km2_0or;
EXPORT_VAR scalar* kp1cn_1or;
EXPORT_VAR scalar* d_kp1cn_1or;
EXPORT_VAR scalar* km1_1or;
EXPORT_VAR scalar* d_km1_1or;
EXPORT_VAR scalar* kp2_1or;
EXPORT_VAR scalar* d_kp2_1or;
EXPORT_VAR scalar* km2_1or;
EXPORT_VAR scalar* d_km2_1or;
EXPORT_VAR scalar* kp1cn_2or;
EXPORT_VAR scalar* d_kp1cn_2or;
EXPORT_VAR scalar* km1_2or;
EXPORT_VAR scalar* d_km1_2or;
EXPORT_VAR scalar* kp2_2or;
EXPORT_VAR scalar* d_kp2_2or;
EXPORT_VAR scalar* km2_2or;
EXPORT_VAR scalar* d_km2_2or;
#define spikeCount_orn glbSpkCntorn[0]
#define spike_orn glbSpkorn
#define glbSpkShiftorn 0

EXPORT_VAR unsigned int* glbSpkCntorn;
EXPORT_VAR unsigned int* d_glbSpkCntorn;
EXPORT_VAR unsigned int* glbSpkorn;
EXPORT_VAR unsigned int* d_glbSpkorn;
EXPORT_VAR uint32_t* recordSpkorn;
EXPORT_VAR uint32_t* d_recordSpkorn;
EXPORT_VAR scalar* Vorn;
EXPORT_VAR scalar* d_Vorn;
EXPORT_VAR scalar* aorn;
EXPORT_VAR scalar* d_aorn;
#define spikeCount_pn glbSpkCntpn[0]
#define spike_pn glbSpkpn
#define glbSpkShiftpn 0

EXPORT_VAR unsigned int* glbSpkCntpn;
EXPORT_VAR unsigned int* d_glbSpkCntpn;
EXPORT_VAR unsigned int* glbSpkpn;
EXPORT_VAR unsigned int* d_glbSpkpn;
EXPORT_VAR uint32_t* recordSpkpn;
EXPORT_VAR uint32_t* d_recordSpkpn;
EXPORT_VAR scalar* Vpn;
EXPORT_VAR scalar* d_Vpn;
EXPORT_VAR scalar* apn;
EXPORT_VAR scalar* d_apn;

// ------------------------------------------------------------------------
// custom update variables
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// pre and postsynaptic variables
// ------------------------------------------------------------------------
EXPORT_VAR double* inSynpn_ln;
EXPORT_VAR double* d_inSynpn_ln;
EXPORT_VAR double* inSynorn_ln;
EXPORT_VAR double* d_inSynorn_ln;
EXPORT_VAR double* inSynln_ln;
EXPORT_VAR double* d_inSynln_ln;
EXPORT_VAR double* inSynor_orn;
EXPORT_VAR double* d_inSynor_orn;
EXPORT_VAR double* inSynln_pn;
EXPORT_VAR double* d_inSynln_pn;
EXPORT_VAR double* inSynorn_pn;
EXPORT_VAR double* d_inSynorn_pn;

// ------------------------------------------------------------------------
// synapse connectivity
// ------------------------------------------------------------------------
EXPORT_VAR const unsigned int maxRowLengthor_orn;
EXPORT_VAR unsigned int* rowLengthor_orn;
EXPORT_VAR unsigned int* d_rowLengthor_orn;
EXPORT_VAR uint32_t* indor_orn;
EXPORT_VAR uint32_t* d_indor_orn;
EXPORT_VAR const unsigned int maxRowLengthorn_ln;
EXPORT_VAR unsigned int* rowLengthorn_ln;
EXPORT_VAR unsigned int* d_rowLengthorn_ln;
EXPORT_VAR uint32_t* indorn_ln;
EXPORT_VAR uint32_t* d_indorn_ln;
EXPORT_VAR const unsigned int maxRowLengthorn_pn;
EXPORT_VAR unsigned int* rowLengthorn_pn;
EXPORT_VAR unsigned int* d_rowLengthorn_pn;
EXPORT_VAR uint32_t* indorn_pn;
EXPORT_VAR uint32_t* d_indorn_pn;
EXPORT_VAR const unsigned int maxRowLengthpn_ln;
EXPORT_VAR unsigned int* rowLengthpn_ln;
EXPORT_VAR unsigned int* d_rowLengthpn_ln;
EXPORT_VAR uint32_t* indpn_ln;
EXPORT_VAR uint32_t* d_indpn_ln;

// ------------------------------------------------------------------------
// synapse variables
// ------------------------------------------------------------------------
EXPORT_VAR scalar* gln_ln;
EXPORT_VAR scalar* d_gln_ln;
EXPORT_VAR scalar* gln_pn;
EXPORT_VAR scalar* d_gln_pn;

EXPORT_FUNC void pushlnSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllnSpikesFromDevice();
EXPORT_FUNC void pushlnCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllnCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getlnCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getlnCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVlnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVlnFromDevice();
EXPORT_FUNC void pushCurrentVlnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVlnFromDevice();
EXPORT_FUNC scalar* getCurrentVln(unsigned int batch = 0); 
EXPORT_FUNC void pushalnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullalnFromDevice();
EXPORT_FUNC void pushCurrentalnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentalnFromDevice();
EXPORT_FUNC scalar* getCurrentaln(unsigned int batch = 0); 
EXPORT_FUNC void pushlnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pulllnStateFromDevice();
EXPORT_FUNC void pushorSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorSpikesFromDevice();
EXPORT_FUNC void pushorCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getorCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getorCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushr0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullr0orFromDevice();
EXPORT_FUNC void pushCurrentr0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentr0orFromDevice();
EXPORT_FUNC scalar* getCurrentr0or(unsigned int batch = 0); 
EXPORT_FUNC void pushrb_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrb_0orFromDevice();
EXPORT_FUNC void pushCurrentrb_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrb_0orFromDevice();
EXPORT_FUNC scalar* getCurrentrb_0or(unsigned int batch = 0); 
EXPORT_FUNC void pushra_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullra_0orFromDevice();
EXPORT_FUNC void pushCurrentra_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentra_0orFromDevice();
EXPORT_FUNC scalar* getCurrentra_0or(unsigned int batch = 0); 
EXPORT_FUNC void pushrb_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrb_1orFromDevice();
EXPORT_FUNC void pushCurrentrb_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrb_1orFromDevice();
EXPORT_FUNC scalar* getCurrentrb_1or(unsigned int batch = 0); 
EXPORT_FUNC void pushra_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullra_1orFromDevice();
EXPORT_FUNC void pushCurrentra_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentra_1orFromDevice();
EXPORT_FUNC scalar* getCurrentra_1or(unsigned int batch = 0); 
EXPORT_FUNC void pushrb_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullrb_2orFromDevice();
EXPORT_FUNC void pushCurrentrb_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentrb_2orFromDevice();
EXPORT_FUNC scalar* getCurrentrb_2or(unsigned int batch = 0); 
EXPORT_FUNC void pushra_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullra_2orFromDevice();
EXPORT_FUNC void pushCurrentra_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentra_2orFromDevice();
EXPORT_FUNC scalar* getCurrentra_2or(unsigned int batch = 0); 
EXPORT_FUNC void pushraorToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullraorFromDevice();
EXPORT_FUNC void pushCurrentraorToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentraorFromDevice();
EXPORT_FUNC scalar* getCurrentraor(unsigned int batch = 0); 
EXPORT_FUNC void pushkp1cn_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkp1cn_0orFromDevice();
EXPORT_FUNC void pushCurrentkp1cn_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkp1cn_0orFromDevice();
EXPORT_FUNC scalar* getCurrentkp1cn_0or(unsigned int batch = 0); 
EXPORT_FUNC void pushkm1_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkm1_0orFromDevice();
EXPORT_FUNC void pushCurrentkm1_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkm1_0orFromDevice();
EXPORT_FUNC scalar* getCurrentkm1_0or(unsigned int batch = 0); 
EXPORT_FUNC void pushkp2_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkp2_0orFromDevice();
EXPORT_FUNC void pushCurrentkp2_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkp2_0orFromDevice();
EXPORT_FUNC scalar* getCurrentkp2_0or(unsigned int batch = 0); 
EXPORT_FUNC void pushkm2_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkm2_0orFromDevice();
EXPORT_FUNC void pushCurrentkm2_0orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkm2_0orFromDevice();
EXPORT_FUNC scalar* getCurrentkm2_0or(unsigned int batch = 0); 
EXPORT_FUNC void pushkp1cn_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkp1cn_1orFromDevice();
EXPORT_FUNC void pushCurrentkp1cn_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkp1cn_1orFromDevice();
EXPORT_FUNC scalar* getCurrentkp1cn_1or(unsigned int batch = 0); 
EXPORT_FUNC void pushkm1_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkm1_1orFromDevice();
EXPORT_FUNC void pushCurrentkm1_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkm1_1orFromDevice();
EXPORT_FUNC scalar* getCurrentkm1_1or(unsigned int batch = 0); 
EXPORT_FUNC void pushkp2_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkp2_1orFromDevice();
EXPORT_FUNC void pushCurrentkp2_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkp2_1orFromDevice();
EXPORT_FUNC scalar* getCurrentkp2_1or(unsigned int batch = 0); 
EXPORT_FUNC void pushkm2_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkm2_1orFromDevice();
EXPORT_FUNC void pushCurrentkm2_1orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkm2_1orFromDevice();
EXPORT_FUNC scalar* getCurrentkm2_1or(unsigned int batch = 0); 
EXPORT_FUNC void pushkp1cn_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkp1cn_2orFromDevice();
EXPORT_FUNC void pushCurrentkp1cn_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkp1cn_2orFromDevice();
EXPORT_FUNC scalar* getCurrentkp1cn_2or(unsigned int batch = 0); 
EXPORT_FUNC void pushkm1_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkm1_2orFromDevice();
EXPORT_FUNC void pushCurrentkm1_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkm1_2orFromDevice();
EXPORT_FUNC scalar* getCurrentkm1_2or(unsigned int batch = 0); 
EXPORT_FUNC void pushkp2_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkp2_2orFromDevice();
EXPORT_FUNC void pushCurrentkp2_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkp2_2orFromDevice();
EXPORT_FUNC scalar* getCurrentkp2_2or(unsigned int batch = 0); 
EXPORT_FUNC void pushkm2_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullkm2_2orFromDevice();
EXPORT_FUNC void pushCurrentkm2_2orToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentkm2_2orFromDevice();
EXPORT_FUNC scalar* getCurrentkm2_2or(unsigned int batch = 0); 
EXPORT_FUNC void pushorStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorStateFromDevice();
EXPORT_FUNC void pushornSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullornSpikesFromDevice();
EXPORT_FUNC void pushornCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullornCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getornCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getornCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVornToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVornFromDevice();
EXPORT_FUNC void pushCurrentVornToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVornFromDevice();
EXPORT_FUNC scalar* getCurrentVorn(unsigned int batch = 0); 
EXPORT_FUNC void pushaornToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullaornFromDevice();
EXPORT_FUNC void pushCurrentaornToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentaornFromDevice();
EXPORT_FUNC scalar* getCurrentaorn(unsigned int batch = 0); 
EXPORT_FUNC void pushornStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullornStateFromDevice();
EXPORT_FUNC void pushpnSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpnSpikesFromDevice();
EXPORT_FUNC void pushpnCurrentSpikesToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpnCurrentSpikesFromDevice();
EXPORT_FUNC unsigned int* getpnCurrentSpikes(unsigned int batch = 0); 
EXPORT_FUNC unsigned int& getpnCurrentSpikeCount(unsigned int batch = 0); 
EXPORT_FUNC void pushVpnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullVpnFromDevice();
EXPORT_FUNC void pushCurrentVpnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentVpnFromDevice();
EXPORT_FUNC scalar* getCurrentVpn(unsigned int batch = 0); 
EXPORT_FUNC void pushapnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullapnFromDevice();
EXPORT_FUNC void pushCurrentapnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullCurrentapnFromDevice();
EXPORT_FUNC scalar* getCurrentapn(unsigned int batch = 0); 
EXPORT_FUNC void pushpnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpnStateFromDevice();
EXPORT_FUNC void pushor_ornConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullor_ornConnectivityFromDevice();
EXPORT_FUNC void pushorn_lnConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorn_lnConnectivityFromDevice();
EXPORT_FUNC void pushorn_pnConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorn_pnConnectivityFromDevice();
EXPORT_FUNC void pushpn_lnConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpn_lnConnectivityFromDevice();
EXPORT_FUNC void pushgln_lnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgln_lnFromDevice();
EXPORT_FUNC void pushinSynln_lnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynln_lnFromDevice();
EXPORT_FUNC void pushln_lnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullln_lnStateFromDevice();
EXPORT_FUNC void pushgln_pnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullgln_pnFromDevice();
EXPORT_FUNC void pushinSynln_pnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynln_pnFromDevice();
EXPORT_FUNC void pushln_pnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullln_pnStateFromDevice();
EXPORT_FUNC void pushinSynor_ornToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynor_ornFromDevice();
EXPORT_FUNC void pushor_ornStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullor_ornStateFromDevice();
EXPORT_FUNC void pushinSynorn_lnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynorn_lnFromDevice();
EXPORT_FUNC void pushorn_lnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorn_lnStateFromDevice();
EXPORT_FUNC void pushinSynorn_pnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynorn_pnFromDevice();
EXPORT_FUNC void pushorn_pnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullorn_pnStateFromDevice();
EXPORT_FUNC void pushinSynpn_lnToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullinSynpn_lnFromDevice();
EXPORT_FUNC void pushpn_lnStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void pullpn_lnStateFromDevice();
// Runner functions
EXPORT_FUNC void copyStateToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyConnectivityToDevice(bool uninitialisedOnly = false);
EXPORT_FUNC void copyStateFromDevice();
EXPORT_FUNC void copyCurrentSpikesFromDevice();
EXPORT_FUNC void copyCurrentSpikeEventsFromDevice();
EXPORT_FUNC void allocateRecordingBuffers(unsigned int timesteps);
EXPORT_FUNC void pullRecordingBuffersFromDevice();
EXPORT_FUNC void allocateMem();
EXPORT_FUNC void freeMem();
EXPORT_FUNC size_t getFreeDeviceMemBytes();
EXPORT_FUNC void stepTime();

// Functions generated by backend
EXPORT_FUNC void updateNeurons(double t, unsigned int recordingTimestep); 
EXPORT_FUNC void updateSynapses(double t);
EXPORT_FUNC void initialize();
EXPORT_FUNC void initializeSparse();
}  // extern "C"
