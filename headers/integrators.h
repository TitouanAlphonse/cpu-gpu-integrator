#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"



// CPU :

void kick_mb(massive_body* mb, int N_mb, double tau);
void kick_tp_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
void kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);

void drift_mb(massive_body* mb, int N_mb, double tau);
void drift_tp_CPU(test_particle* tp, int N_tp, double tau);
void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);

__host__ void writing(massive_body* mb, test_particle* tp, int N_mb, int N_tp, ofstream& fich);

void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, string suffix);
void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step);

__host__ void kick1_mb_multi_t(massive_body* mb_multi_t, int N_mb, double tau, int substep);
__host__ void drift_mb_multi_t(massive_body* mb_multi_t, int N_mb, double tau, int substep);
__host__ void kick2_mb_multi_t(massive_body* mb_multi_t, int N_mb, double tau, int substep);

__host__ void writing_multi_t(massive_body* mb_multi_t, test_particle* tp_multi_t, int N_mb, int N_tp, int N_substep, ofstream& fich);


// GPU :

__device__ void kick_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
__device__ void drift_tp_GPU(test_particle* tp, int N_tp, double tau);
__global__ void step_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift);

__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int nb_block, int nb_threads, string suffix);
__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int nb_block, int nb_threads);
__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, double integration_duration, int nb_block, int nb_threads, string suffix);

__global__ void step_tp_GPU_multi_t(massive_body* mb_multi_t, test_particle* tp_multi_t, int N_mb, int N_tp, double tau_kick, double tau_drift, int N_substep);

__host__ void leapfrog_GPU_multi_t(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int N_substep, int nb_block, int nb_thread, string suffix);
__host__ void leapfrog_GPU_multi_t(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int N_substep, int nb_block, int nb_thread);