#ifndef LEAPFROG_H
#define LEAPFROG_H

#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"
#include"tools.h"

namespace leapfrog {

// This function returns the center of mass to allow to well compute the orbital elements
void pos_vel_sub(Vec3d& q_sub, Vec3d& v_sub, massive_body* mb, int N_mb, double M_tot);
void pos_vel_sub_multi_step(Vec3d& q_sub, Vec3d& v_sub, massive_body_qv* mb_qv_multi_step, massive_body_mR* m_mR, int N_mb, double M_tot, int substep);

// CPU :

void kick_mb(massive_body* mb, int N_mb, double tau);
void kick_tp_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
void kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);

void drift_mb(massive_body* mb, int N_mb, double tau);
void drift_tp_CPU(test_particle* tp, int N_tp, double tau);
void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);

void step_leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, massive_body* aux_mb);


// GPU :

__device__ void kick_tp_GPU(massive_body* mb, test_particle& tp_i, int N_mb, int N_tp, double tau);
__device__ void drift_tp_GPU(test_particle& tp_i, int N_tp, double tau);
__global__ void step_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift);

__host__ void step_leapfrog_GPU(massive_body* mb, test_particle* tp, massive_body* access_mb, test_particle* access_tp, int N_mb, int N_tp, double tau, int nb_block, int nb_thread, massive_body* aux_mb);


// GPU multi-step :

__host__ void kick_mb_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, int N_mb, double tau, int substep);
__host__ void drift_mb_multi_step(massive_body_qv* mb_qv_multi_step, int N_mb, double tau, int substep);
__host__ void substep_leapfrog_mb_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, int N_mb,  double tau, int substep, massive_body_qv* mb_qv_half_multi_step);

__device__ void kick_tp_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle& tp_i_ss, int N_mb, int N_tp, double tau, int substep);
__global__ void step_update_tp_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle* tp_multi_step, int N_mb, int N_tp, double tau_kick, double tau_drift, int N_substep);
__host__ void step_leapfrog_tp_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle* tp_multi_step, massive_body_qv* access_mb_qv_multi_step, massive_body_mR* access_mb_mR, test_particle* access_tp_multi_step, int N_mb, int N_tp, int N_substep, double tau, int nb_block, int nb_thread, massive_body_qv* aux_mb_qv_multi_step);

}


#endif