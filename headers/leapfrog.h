#ifndef LEAPFROG_H
#define LEAPFROG_H

#include"global.h"
#include"Vec3d.h"
#include"Massive_body.h"
#include"Test_particle.h"
#include"tools.h"

namespace leapfrog {

// This function returns the center of mass to allow to well compute the orbital elements
void pos_vel_sub(Vec3d& q_sub, Vec3d& v_sub, Massive_body* mb, int N_mb, double M_tot);
void pos_vel_sub_multi_step(Vec3d& q_sub, Vec3d& v_sub, Massive_body_qv* mb_qv_multi_step, Massive_body_mR* m_mR, int N_mb, double M_tot, int substep);

// CPU :

void kick_mb(Massive_body* mb, int N_mb, double tau);
void kick_tp_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau);
void kick_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau);

void drift_mb(Massive_body* mb, int N_mb, double tau);
void drift_tp_CPU(Test_particle* tp, int N_tp, double tau);
void drift_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau);

void step_leapfrog_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, Massive_body* aux_mb);


// GPU :

__device__ void kick_tp_GPU(Massive_body* mb, Test_particle& tp_i, int N_mb, int N_tp, double tau);
__device__ void drift_tp_GPU(Test_particle& tp_i, int N_tp, double tau);
__global__ void step_tp_GPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift);

__host__ void step_leapfrog_GPU(Massive_body* mb, Test_particle* tp, Massive_body* access_mb, Test_particle* access_tp, int N_mb, int N_tp, double tau, int nb_block, int nb_thread, Massive_body* aux_mb);


// GPU multi-step :

__host__ void kick_mb_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, int N_mb, double tau, int substep);
__host__ void drift_mb_multi_step(Massive_body_qv* mb_qv_multi_step, int N_mb, double tau, int substep);
__host__ void substep_leapfrog_mb_GPU_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, int N_mb,  double tau, int substep, Massive_body_qv* mb_qv_half_multi_step);

__device__ void kick_tp_GPU_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, Test_particle& tp_i_ss, int N_mb, int N_tp, double tau, int substep);
__global__ void step_update_tp_GPU_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, Test_particle* tp_multi_step, int N_mb, int N_tp, double tau_kick, double tau_drift, int N_substep);
__host__ void step_leapfrog_tp_GPU_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, Test_particle* tp_multi_step, Massive_body_qv* access_mb_qv_multi_step, Massive_body_mR* access_mb_mR, Test_particle* access_tp_multi_step, int N_mb, int N_tp, int N_substep, double tau, int nb_block, int nb_thread, Massive_body_qv* aux_mb_qv_multi_step);

}


#endif