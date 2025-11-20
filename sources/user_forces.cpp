#include"../headers/User_forces.h"


User_forces::User_forces(string user_forces_type, bool& enable_user_forces) {
    enable_user_forces = false;

    if (user_forces_type != "none" && user_forces_type != "None") {
        enable_user_forces = true;

        if (user_forces_type == "dissipation_4p") {
            nb_dis = 1;
            N_mb_dis = 3;

            id_mb = vector<int>(N_mb_dis);
            id_mb[0] = 2;
            id_mb[1] = 3;
            id_mb[2] = 4;

            dur_dis = vector<double>(nb_dis);
            tau_dis = vector<double>(nb_dis);
            dvel = vector<double>(N_mb_dis*nb_dis);
            dedt = vector<double>(N_mb_dis*nb_dis);

            dur_dis[0] = 5e6;
            tau_dis[0] = 1e6;
            dvel[0] = 2.88e-7*0.9;
            dvel[1] = 1.92e-07*0.98;
            dvel[2] = 7.69e-08*1.4;
            dedt[0] = 6.02e-12;
            dedt[1] = 2.74e-12;
            dedt[2] = 1.37e-12;
        }

        if (user_forces_type == "dissipation_n") {
            nb_dis = 1;
            N_mb_dis = 1;

            id_mb = vector<int>(N_mb_dis);
            id_mb[0] = 1;

            dur_dis = vector<double>(nb_dis);
            tau_dis = vector<double>(nb_dis);
            dvel = vector<double>(N_mb_dis*nb_dis);
            dedt = vector<double>(N_mb_dis*nb_dis);

            dur_dis[0] = 15e6;
            tau_dis[0] = 2.5e6;
            dvel[0] = 4.39e-8;
            dedt[0] = 1.37e-12;
        }
    }
}


void User_forces::apply(Massive_body* mb, int N_mb, double t, double tau, pos_vel_sub_func pos_vel_sub, double M_tot) {
    int i_dis, i;
    double r2, rm1, k, omk, fr, ft;
    Vec3d ur, ut, acc, q_sub, v_sub, q_mb, v_mb;

    pos_vel_sub(q_sub, v_sub, mb, N_mb, M_tot);

    i_dis = 0;
    while (t > dur_dis[i_dis] && i_dis < nb_dis) {
        i_dis++;
    }
    if (i_dis < nb_dis) {
        for (int i_brut=0; i_brut<N_mb_dis; i_brut++) {
            i = id_mb[i_brut];
            q_mb = mb[i].q - q_sub;
            v_mb = mb[i].v - v_sub;

            // Force affecting the semimajor axis :
            acc = Vec3d();
            r2 = q_mb.get_x()*q_mb.get_x() + q_mb.get_y()*q_mb.get_y();
            rm1 = 1./sqrt(r2);
            ur = Vec3d(q_mb.get_x()*rm1, q_mb.get_y()*rm1, 0, "xyz");
            ut = Vec3d(-ur.get_y(), ur.get_x(), 0, "xyz");

            k = dvel[i_dis*N_mb_dis+i_brut]*exp(-t/tau_dis[i_dis])/v_mb.norm();
            acc += k*v_mb;
            
            // Force affecting the eccentricity :
            omk = sqrt((mb[0].m + mb[i].m)*rm1/r2);
            fr = scalar_product(v_mb, ur);
            ft = scalar_product(v_mb, ut) - omk/rm1;
            k = dedt[i_dis*N_mb_dis+i_brut]*exp(-t/tau_dis[i_dis]);
            acc -= k*(ur*fr + ut*ft);
            
            mb[i].v += tau*acc;
        }
    }
}


void User_forces::apply_multi_step(Massive_body_qv* mb_qv_multi_step, Massive_body_mR* mb_mR, int N_mb, int substep, double t, double tau, pos_vel_sub_func_multi_step pos_vel_sub, double M_tot) {
    int i_dis, i, i_ss;
    double r2, rm1, k, omk, fr, ft;
    Vec3d ur, ut, acc, q_sub, v_sub, q_mb, v_mb;

    pos_vel_sub(q_sub, v_sub, mb_qv_multi_step, mb_mR, N_mb, M_tot, substep);

    i_dis = 0;
    while (t > dur_dis[i_dis] && i_dis < nb_dis) {
        i_dis++;
    }
    if (i_dis < nb_dis) {
        for (int i_brut=0; i_brut<N_mb_dis; i_brut++) {
            i = id_mb[i_brut];
            i_ss = substep*N_mb + i;
            q_mb = mb_qv_multi_step[i_ss].q - q_sub;
            v_mb = mb_qv_multi_step[i_ss].v - v_sub;

            // Force affecting the semimajor axis :
            acc = Vec3d();
            r2 = q_mb.get_x()*q_mb.get_x() + q_mb.get_y()*q_mb.get_y();
            rm1 = 1./sqrt(r2);
            ur = Vec3d(q_mb.get_x()*rm1, q_mb.get_y()*rm1, 0, "xyz");
            ut = Vec3d(-ur.get_y(), ur.get_x(), 0, "xyz");

            k = dvel[i_dis*N_mb_dis+i_brut]*exp(-t/tau_dis[i_dis])/v_mb.norm();
            acc += k*v_mb;

            // Force affecting the eccentricity :
            omk = sqrt((mb_mR[0].m + mb_mR[i].m)*rm1/r2);
            fr = scalar_product(v_mb, ur);
            ft = scalar_product(v_mb, ut) - omk/rm1;
            k = dedt[i_dis*N_mb_dis+i_brut]*exp(-t/tau_dis[i_dis]);
            acc -= k*(ur*fr + ut*ft);
            
            mb_qv_multi_step[i_ss].v += tau*acc;
        }
    }
}