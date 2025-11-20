#ifndef INTEGRATION_H
#define INTEGRATION_H

#include"global.h"
#include"Vec3d.h"
#include"Massive_body.h"
#include"Test_particle.h"
#include"tools.h"
#include"leapfrog.h"
#include"MVS.h"
#include"User_forces.h"

// Useful type definitions for the following functions :
typedef void (*step_function_CPU)(Massive_body*, Test_particle*, int, int, double, Massive_body*);
typedef void (*step_function_GPU)(Massive_body*, Test_particle*, Massive_body*, Test_particle*, int, int, double, int, int, Massive_body*);
typedef void (*substep_function_mb_GPU_multi_step)(Massive_body_qv*, Massive_body_mR*, int, double, int, Massive_body_qv*);
typedef void (*step_function_tp_GPU_multi_step)(Massive_body_qv*, Massive_body_mR*, Test_particle*, Massive_body_qv*, Massive_body_mR*, Test_particle*, int, int, int, double, int, int, Massive_body_qv*);
typedef void (*pos_vel_sub_func)(Vec3d&, Vec3d&, Massive_body*, int, double);
typedef void (*pos_vel_sub_func_multi_step)(Vec3d&, Vec3d&, Massive_body_qv*, Massive_body_mR*, int, double, int);

void print_info_start(string integration_method, string integration_mode, int N_mb, int N_tp, int N_step, double computation_time, bool def_N_step_in_cp_time, int N_substep, double tau, double freq_w, bool def_freq_w_in_steps);
void print_info_end(int N_step, bool def_N_step_in_cp_time, double tau, string suffix);

// Writes the data at t=0
__host__ void write_init(ofstream& file_general, ofstream& file_pos, double tau, double freq_w, Massive_body* mb, Test_particle* tp, int N_mb, int N_tp); 
__host__ void write_positions(ofstream& file_pos, int step, Massive_body* mb, Test_particle* tp, int N_mb, int N_tp);
__host__ void write_positions_multi_step(ofstream& file_pos, int step, int substep, Massive_body_qv* mb_qv_multi_step, Test_particle* tp_multi_step, int N_mb, int N_tp, int N_substep);
 // Writes the orbital elements :
__host__ void write_orb_param(ofstream& file_orb_param, int step, Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, pos_vel_sub_func pos_vel_del, double M_tot);

// Assign an id to the type of integration we want to perform
int integration_id(string integration_method, string integration_mode, bool def_N_step_in_cp_time);
// Start the integration with the right method and mode
void launch_integration(string integration_method, string integration_mode, Massive_body* mb,Test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, bool def_N_step_in_cp_time, int N_substep, string user_forces_type, int nb_block, int nb_thread, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps);


// Integration using the CPU :
__host__ void integration_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, int N_step, step_function_CPU step_function, string user_forces_type, pos_vel_sub_func pos_vel_del, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps);
__host__ void integration_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, step_function_CPU step_function, string user_forces_type, pos_vel_sub_func pos_vel_del, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps);

// Integration using the GPU :
__host__ void integration_GPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int nb_block, int nb_thread, step_function_GPU step_function, string user_forces_type, pos_vel_sub_func pos_vel_del, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps);
__host__ void integration_GPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, int nb_block, int nb_thread, step_function_GPU step_function, string user_forces_type, pos_vel_sub_func pos_vel_del, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps);

// Integration using the GPU multi-step method :
__host__ void integration_GPU_multi_step(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int N_substep, int nb_block, int nb_thread, substep_function_mb_GPU_multi_step substep_mb, step_function_tp_GPU_multi_step step_tp, string user_forces_type, pos_vel_sub_func_multi_step pos_vel_sub, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps);

#endif