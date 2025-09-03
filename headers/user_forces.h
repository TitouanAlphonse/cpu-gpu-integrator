#ifndef USER_FORCES_H
#define USER_FORCES_H

#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"

typedef void (*pos_vel_sub_func)(Vec3d&, Vec3d&, massive_body*, int, double);
typedef void (*pos_vel_sub_func_multi_step)(Vec3d&, Vec3d&, massive_body_qv*, massive_body_mR*, int, double, int);

// Initializes the user forces (parameters are contained inside the function)
void init_user_forces(string user_forces, bool& enable_user_forces, int& nb_dis, int& N_mb_dis, int*& id_mb, double*& dur_dis, double*& tau_dis, double*& dvel, double*& dedt);

void apply_user_forces(massive_body* mb, int N_mb, double t, double tau, string user_forces, int nb_dis, int N_mb_dis, int* id_mb, double* dur_dis, double* tau_dis, double* dvel, double* dedt, pos_vel_sub_func pos_vel_sub, double M_tot);
void apply_user_forces_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, int N_mb, int substep, double t, double tau, string user_forces, int nb_dis, int N_mb_dis, int* id_mb, double* dur_dis, double* tau_dis, double* dvel, double* dedt, pos_vel_sub_func_multi_step pos_vel_sub, double M_tot);


#endif