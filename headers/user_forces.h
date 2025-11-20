#ifndef USER_FORCES_H
#define USER_FORCES_H

#include"global.h"
#include"Vec3d.h"
#include"Massive_body.h"

typedef void (*pos_vel_sub_func)(Vec3d&, Vec3d&, Massive_body*, int, double);
typedef void (*pos_vel_sub_func_multi_step)(Vec3d&, Vec3d&, Massive_body_qv*, Massive_body_mR*, int, double, int);

class User_forces {
private:
    int nb_dis;
    int N_mb_dis;
    vector<int> id_mb;
    vector<double> dur_dis;
    vector<double> tau_dis;
    vector<double> dvel;
    vector<double> dedt;

public:
    // Initializes the user forces (parameters are contained inside the function)
    User_forces(string user_forces_type, bool& enable_user_forces);

    void apply(Massive_body* mb, int N_mb, double t, double tau, pos_vel_sub_func pos_vel_sub, double M_tot);
    void apply_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, int N_mb, int substep, double t, double tau, pos_vel_sub_func_multi_step pos_vel_sub, double M_tot);
};


#endif