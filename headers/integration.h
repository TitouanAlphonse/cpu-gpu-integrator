#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"
#include"leapfrog.h"

typedef void (*step_function_CPU)(massive_body*, test_particle*, int, int, double);
typedef void (*step_function_GPU)(massive_body*, test_particle*, massive_body*, test_particle*, int, int, double, int, int);

void print_info_start(string integration_method, string integration_mode, int N_mb, int N_tp, int N_step, double computation_time, bool def_N_step_in_cp_time, double tau, double freq_w, bool def_freq_w_in_steps);
void print_info_end(int N_step, bool def_N_step_in_cp_time, double tau, string suffix);

__host__ void write_init(ofstream& file_general, ofstream& file_pos, double tau, double freq_w, massive_body* mb, test_particle* tp, int N_mb, int N_tp);
__host__ void write_positions(ofstream& file_pos, int step, massive_body* mb, test_particle* tp, int N_mb, int N_tp);

int integration_id(string integration_method, string integration_mode, bool def_N_step_in_cp_time);
void launch_integration(string integration_method, string integration_mode, massive_body* mb,test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, bool def_N_step_in_cp_time, int nb_block, int nb_thread, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps);

__host__ void integration_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, step_function_CPU step_function, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps);
__host__ void integration_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, step_function_CPU step_function, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps);

__host__ void integration_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int nb_block, int nb_thread, step_function_GPU step_function, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps);
__host__ void integration_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, int nb_block, int nb_thread, step_function_GPU step_function, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps);