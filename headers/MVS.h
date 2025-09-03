#ifndef MVS_H
#define MVS_H

#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"
#include"tools.h"


namespace MVS {

// This function returns null vectors since the orbital parameters are already
void pos_vel_sub(Vec3d& q_sub, Vec3d& v_sub, massive_body* mb, int N_mb, double M_tot);

__host__ __device__ void kepler_solver(double& delta_E, double tau, double n, double EC, double ES, double e);

void kick_mb(massive_body* mb_helio, massive_body* mb_jacobi, int N_mb, double tau);
void kick_tp(massive_body* mb_helio, test_particle* tp, int N_mb, int N_tp, double tau);
void kick_CPU(massive_body* mb, massive_body* aux_mb, test_particle* tp, int N_mb, int N_tp, double tau);

void drift_mb(massive_body* mb_helio, massive_body* mb_jacobi, int N_mb, double tau);
void drift_tp_CPU(test_particle* tp, double m0, int N_mb, double tau);
void drift_CPU(massive_body* mb, massive_body* aux_mb, test_particle* tp, int N_mb, int N_tp, double tau);

void step_MVS_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, massive_body* aux_mb);

__device__ void kick_tp_GPU(massive_body* mb, test_particle& tp_i, int N_mb, int N_tp, double tau);
__device__ void drift_tp_GPU(test_particle& tp_i, double m0, int N_tp, double tau);
__global__ void step_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift);
__host__ void step_MVS_GPU(massive_body* mb, test_particle* tp, massive_body* access_mb, test_particle* access_tp, int N_mb, int N_tp, double tau, int nb_block, int nb_thread, massive_body* aux_mb);

}


#endif