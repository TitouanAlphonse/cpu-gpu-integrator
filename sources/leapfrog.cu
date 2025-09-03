#include"../headers/leapfrog.h"


// #################
//   leapfrog.cu
// #################

// See headers/leapfrog.h for details about the aim of the functions


using namespace leapfrog;

void leapfrog::pos_vel_sub(Vec3d& q_sub, Vec3d& v_sub, massive_body* mb, int N_mb, double M_tot) {
    q_sub = Vec3d();
    v_sub = Vec3d();

    for (int i=0; i<N_mb; i++) {
        q_sub += mb[i].m*mb[i].q/M_tot;
        v_sub += mb[i].m*mb[i].v/M_tot;
    }
}

void leapfrog::pos_vel_sub_multi_step(Vec3d& q_sub, Vec3d& v_sub, massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, int N_mb, double M_tot, int substep) {
    q_sub = Vec3d();
    v_sub = Vec3d();

    int i_ss;
    for (int i=0; i<N_mb; i++) {
        i_ss = substep*N_mb + i;
        q_sub += mb_mR[i].m*mb_qv_multi_step[i_ss].q/M_tot;
        v_sub += mb_mR[i].m*mb_qv_multi_step[i_ss].v/M_tot;
    }
}


//--------------------------------------------//
//                     CPU                    //
//--------------------------------------------//


void leapfrog::kick_mb(massive_body* mb, int N_mb, double tau) {
    Vec3d r_ij;
    double norm_r_ij3;
    Vec3d V_ij;

    for (int i=0; i<N_mb-1; i++) {
        for (int j=i+1; j<N_mb; j++) {
            r_ij = mb[i].q - mb[j].q;
            norm_r_ij3 = r_ij.norm2()*r_ij.norm();
            V_ij = G/norm_r_ij3*r_ij;

            mb[i].v -= mb[j].m*tau*V_ij;
            mb[j].v += mb[i].m*tau*V_ij;
        }
    }
}

void leapfrog::kick_tp_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    Vec3d r_ij;
    double norm_r_ij3;
    Vec3d V_ij;

    for (int i=0; i<N_tp; i++) {
        for (int j=0; j<N_mb; j++) {
            r_ij = tp[i].q - mb[j].q;
            norm_r_ij3 = r_ij.norm2()*r_ij.norm();
            V_ij = G*mb[j].m/norm_r_ij3*r_ij;
            tp[i].v -= tau*V_ij;
        }
    }
}

void leapfrog::kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    kick_mb(mb, N_mb, tau);
    kick_tp_CPU(mb, tp, N_mb, N_tp, tau);
}


void leapfrog::drift_mb(massive_body* mb, int N_mb, double tau) {
    for (int i=0; i<N_mb; i++) {
        mb[i].q += tau*mb[i].v;
    }
}

void leapfrog::drift_tp_CPU(test_particle* tp, int N_tp, double tau) {
    for (int i=0; i<N_tp; i++) {
        tp[i].q += tau*tp[i].v;
    }
}

void leapfrog::drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    drift_mb(mb, N_mb, tau);
    drift_tp_CPU(tp, N_tp, tau);
}

void leapfrog::step_leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, massive_body* aux_mb) {
    drift_CPU(mb, tp, N_mb, N_tp, tau/2);
    kick_CPU(mb, tp, N_mb, N_tp, tau);
    drift_CPU(mb, tp, N_mb, N_tp, tau/2);
}

__host__ void leapfrog::kick_mb_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, int N_mb, double tau, int substep) {
    Vec3d r_ij;
    double norm_r_ij3;
    Vec3d V_ij;

    for (int i=0; i<N_mb-1; i++) {
        for (int j=i+1; j<N_mb; j++) {

            r_ij = mb_qv_multi_step[substep*N_mb+i].q - mb_qv_multi_step[substep*N_mb+j].q;
            norm_r_ij3 = r_ij.norm2()*r_ij.norm();
            V_ij = G/norm_r_ij3*r_ij;
            mb_qv_multi_step[substep*N_mb+i].v -= mb_mR[j].m*tau*V_ij;
            mb_qv_multi_step[substep*N_mb+j].v += mb_mR[i].m*tau*V_ij;
        }
    }
}

__host__ void leapfrog::drift_mb_multi_step(massive_body_qv* mb_qv_multi_step, int N_mb, double tau, int substep) {
    for (int i=0; i<N_mb; i++) {
        mb_qv_multi_step[substep*N_mb+i].q += tau*mb_qv_multi_step[substep*N_mb+i].v;
    }
}


__host__ void leapfrog::substep_leapfrog_mb_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, int N_mb,  double tau, int substep, massive_body_qv* mb_qv_half_multi_step) {
    for (int i=0; i<N_mb; i++) {
        mb_qv_multi_step[substep*N_mb+i].q = mb_qv_multi_step[(substep-1)*N_mb+i].q;
        mb_qv_multi_step[substep*N_mb+i].v = mb_qv_multi_step[(substep-1)*N_mb+i].v;
    }
    drift_mb_multi_step(mb_qv_multi_step, N_mb, tau/2, substep);
    for (int i=0; i<N_mb; i++) {
        mb_qv_half_multi_step[substep*N_mb+i].q = mb_qv_multi_step[substep*N_mb+i].q;
        mb_qv_half_multi_step[substep*N_mb+i].v = mb_qv_multi_step[substep*N_mb+i].v;
    }
    kick_mb_multi_step(mb_qv_multi_step, mb_mR, N_mb, tau, substep);
    drift_mb_multi_step(mb_qv_multi_step, N_mb, tau/2, substep);
}


//--------------------------------------------//
//                     GPU                    //
//--------------------------------------------//


__device__ void leapfrog::kick_tp_GPU(massive_body* mb, test_particle& tp_i, int N_mb, int N_tp, double tau) {
    double dx, dy, dz, r_ij3, k, new_vx, new_vy, new_vz;

    new_vx = tp_i.v.get_x();
    new_vy = tp_i.v.get_y();
    new_vz = tp_i.v.get_z();

    for (int j=0; j<N_mb; j++) {
        dx = tp_i.q.get_x() - mb[j].q.get_x();
        dy = tp_i.q.get_y() - mb[j].q.get_y();
        dz = tp_i.q.get_z() - mb[j].q.get_z();
        r_ij3 = (dx*dx + dy*dy + dz*dz)*sqrt(dx*dx + dy*dy + dz*dz);
        k = tau*G*mb[j].m/r_ij3;

        new_vx -= k*dx;
        new_vy -= k*dy;
        new_vz -= k*dz;    
    }

    tp_i.v.set_xyz(new_vx, new_vy, new_vz);
}

__device__ void leapfrog::drift_tp_GPU(test_particle& tp_i, int N_tp, double tau) {
    double new_x, new_y, new_z;
    
    new_x = tp_i.q.get_x() + tau*tp_i.v.get_x();
    new_y = tp_i.q.get_y() + tau*tp_i.v.get_y();
    new_z = tp_i.q.get_z() + tau*tp_i.v.get_z();
    tp_i.q.set_xyz(new_x, new_y, new_z);
}

__global__ void leapfrog::step_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < N_tp) {
        drift_tp_GPU(tp[i], N_tp, tau_drift);
        kick_tp_GPU(mb, tp[i], N_mb, N_tp, tau_kick);
        drift_tp_GPU(tp[i], N_tp, tau_drift);
    }
}


__host__ void leapfrog::step_leapfrog_GPU(massive_body* mb, test_particle* tp, massive_body* access_mb, test_particle* access_tp, int N_mb, int N_tp, double tau, int nb_block, int nb_thread, massive_body* aux_mb) {
    drift_mb(mb, N_mb, tau/2);
    cudaMemcpy(access_mb, mb, N_mb*sizeof(massive_body), cudaMemcpyHostToDevice); // Copy the data for time-step step+1/2
    kick_mb(mb, N_mb, tau);
    drift_mb(mb, N_mb, tau/2);

    step_tp_GPU<<<nb_block, nb_thread>>>(access_mb, access_tp, N_mb, N_tp, tau, tau/2);
    cudaMemcpy(tp, access_tp, N_tp*sizeof(test_particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}


__device__ void leapfrog::kick_tp_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle& tp_i_ss, int N_mb, int N_tp, double tau, int substep) {
    int j_ss;
    double dx, dy, dz, r_ij3, k, new_vx, new_vy, new_vz;

    new_vx = tp_i_ss.v.get_x();
    new_vy = tp_i_ss.v.get_y();
    new_vz = tp_i_ss.v.get_z();

    for (int j=0; j<N_mb; j++) {
        j_ss = substep*N_mb + j;

        dx = tp_i_ss.q.get_x() - mb_qv_multi_step[j_ss].q.get_x();
        dy = tp_i_ss.q.get_y() - mb_qv_multi_step[j_ss].q.get_y();
        dz = tp_i_ss.q.get_z() - mb_qv_multi_step[j_ss].q.get_z();
        r_ij3 = (dx*dx + dy*dy + dz*dz)*sqrt(dx*dx + dy*dy + dz*dz);
        k = tau*G*mb_mR[j].m/r_ij3;

        new_vx -= k*dx;
        new_vy -= k*dy;
        new_vz -= k*dz;
    }

    tp_i_ss.v.set_xyz(new_vx, new_vy, new_vz);
}


__global__ void leapfrog::step_update_tp_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle* tp_multi_step, int N_mb, int N_tp, double tau_kick, double tau_drift, int N_substep) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int i_ss;
    if (i < N_tp) {
        for (int substep=1; substep<=N_substep; substep++) {
            i_ss = substep*N_tp + i;
            tp_multi_step[i_ss].q = tp_multi_step[(substep-1)*N_tp + i].q;
            tp_multi_step[i_ss].v = tp_multi_step[(substep-1)*N_tp + i].v;

            drift_tp_GPU(tp_multi_step[i_ss], N_tp, tau_drift);
            kick_tp_GPU_multi_step(mb_qv_multi_step, mb_mR, tp_multi_step[i_ss], N_mb, N_tp, tau_kick, substep);
            drift_tp_GPU(tp_multi_step[i_ss], N_tp, tau_drift);
        
        }
        // Initialization for the next call :
        tp_multi_step[i] = tp_multi_step[N_substep*N_tp+i];
    }
}


__host__ void leapfrog::step_leapfrog_tp_GPU_multi_step(massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle* tp_multi_step, massive_body_qv* access_mb_qv_multi_step, massive_body_mR* access_mb_mR, test_particle* access_tp_multi_step, int N_mb, int N_tp, int N_substep, double tau, int nb_block, int nb_thread, massive_body_qv* aux_mb_qv_multi_step) {
    cudaMemcpy(access_mb_qv_multi_step, aux_mb_qv_multi_step, (N_substep+1)*N_mb*sizeof(massive_body_qv), cudaMemcpyHostToDevice); // Copy the data for time-step step+1/2
    step_update_tp_GPU_multi_step<<<nb_block, nb_thread>>>(access_mb_qv_multi_step, access_mb_mR, access_tp_multi_step, N_mb, N_tp, tau, tau/2, N_substep);
    cudaMemcpy(tp_multi_step, access_tp_multi_step, (N_substep+1)*N_tp*sizeof(test_particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}