#include"../headers/MVS.h"


// #################
//     MVS.cu
// #################

// See headers/MVS.h for details about the aim of the functions


using namespace MVS;

void MVS::pos_vel_sub(Vec3d& q_sub, Vec3d& v_sub, Massive_body* mb, int N_mb, double M_tot) {
    q_sub = Vec3d();
    v_sub = Vec3d();
}


__host__ __device__ void MVS::kepler_solver(double& delta_E, double tau, double n, double EC, double ES, double e) {
    double x, y, cos_x, sin_x, f, fp, fpp, fppp, dx, dx_max;
    dx_max = 1e-14;
    y = n*tau - ES;
    x = y - ES - 0.85*e;

    int i = 0, i_max = 20;
    do { 
        cos_x = cos(x);
        sin_x = sin(x);
        f = x - EC*sin_x + ES*(1-cos_x) - n*tau;
        fp = 1. - EC*cos_x + ES*sin_x;
        fpp = EC*sin_x + ES*cos_x;
        fppp = EC*cos_x - ES*sin_x;
        if (fp != 0) {
            dx = -f/fp;
            dx = -f/(fp + 0.5*dx*fpp);
            dx = -f/(fp + 0.5*dx*fpp + 1./6.*dx*dx*fppp);
        }
        else {
            dx = 0;
        }
        x += dx;
        i++;
    } while (dx > dx_max && i<i_max);

    if (dx <= dx_max && i<i_max) {
        delta_E = x;
    }
    else {
        printf("Error : No convergence\n");
        delta_E = 0;
    }
}

__host__ __device__ void gauss_func(Vec3d q0, Vec3d v0, double mu, double tau, double& f, double& g, double& fdot, double& gdot) {
    double u, a, r0, n, EC, ES, e, dE, r_over_a;
    u = q0.get_x()*v0.get_x() + q0.get_y()*v0.get_y() + q0.get_z()*v0.get_z();
    r0 = q0.norm();
    a = 1./(2/r0 - v0.norm2()/mu);

    n = sqrt(mu/(a*a*a));
    EC  = 1-r0/a;
    ES = u/(n*a*a);
    e = sqrt(EC*EC + ES*ES);

    kepler_solver(dE, tau, n, EC, ES, e);
    
    r_over_a = 1. - EC*cos(dE) + ES*sin(dE);

    f = a/r0*(cos(dE)-1)+1;
    g = tau + (sin(dE) - dE)/n;

    fdot = -a/(r0*r_over_a)*n*sin(dE);
    gdot = (cos(dE) - 1)/r_over_a + 1;
}



//--------------------------------------------//
//                     CPU                    //
//--------------------------------------------//


void MVS::kick_mb(Massive_body* mb_helio, Massive_body* mb_jacobi, int N_mb, double tau) {
    double sum_m = mb_helio[0].m;
    double sum_m2;

    Vec3d r_ij;
    Vec3d V_ij;

    for (int i=1; i<N_mb-1; i++) {
        
        sum_m2 = sum_m;
        sum_m += mb_helio[i].m;
        mb_helio[i].v += tau*G*mb_helio[0].m*(mb_jacobi[i].q/pow(mb_jacobi[i].q.norm(), 3) - mb_helio[i].q/pow(mb_helio[i].q.norm(), 3));

        for (int j=i+1; j<N_mb; j++) {
            
            sum_m2 += mb_helio[j-1].m;
            mb_helio[i].v += tau*G*mb_helio[0].m*mb_helio[i].m/(sum_m2*pow(mb_jacobi[i].q.norm(), 3))*mb_jacobi[i].q;

            r_ij = mb_helio[i].q - mb_helio[j].q;
            V_ij = G/pow(r_ij.norm(), 3)*r_ij;

            mb_helio[i].v -= tau*mb_helio[j].m*V_ij;
            mb_helio[j].v += tau*mb_helio[i].m*V_ij;
        }
    }
}


void MVS::drift_mb(Massive_body* mb_helio, Massive_body* mb_jacobi, int N_mb, double tau) {
    Vec3d q0, v0;
    double f, g, fdot, gdot;
    for (int i=1; i<N_mb; i++) {
        q0 = mb_jacobi[i].q;
        v0 = mb_jacobi[i].v;

        gauss_func(mb_jacobi[i].q, mb_jacobi[i].v, G*mb_helio[0].m*mb_helio[i].m/mb_jacobi[i].m, tau, f, g, fdot, gdot);

        mb_jacobi[i].q = f*q0 + g*v0;
        mb_jacobi[i].v = fdot*q0 + gdot*v0;
    }
}



void MVS::kick_tp(Massive_body* mb_helio, Test_particle* tp, int N_mb, int N_tp, double tau) {
    Vec3d r_ij;
    Vec3d V_ij;

    for (int i=0; i<N_tp; i++) {
        for (int j=1; j<N_mb; j++) {
            r_ij = tp[i].q - mb_helio[j].q;
            V_ij = G*mb_helio[j].m/pow(r_ij.norm(), 3)*r_ij;
            V_ij += G*mb_helio[j].m/pow(mb_helio[j].q.norm(), 3)*mb_helio[j].q;

            tp[i].v -= tau*V_ij;
        }
    }
}

void MVS::drift_tp_CPU(Test_particle* tp, double m0, int N_tp, double tau) {
    Vec3d q0, v0;
    double f, g, fdot, gdot;
    for (int i=0; i<N_tp; i++) {
        q0 = tp[i].q;
        v0 = tp[i].v;

        gauss_func(tp[i].q, tp[i].v, G*m0, tau, f, g, fdot, gdot);

        tp[i].q = f*q0 + g*v0;
        tp[i].v = fdot*q0 + gdot*v0;
    }
}

void MVS::kick_CPU(Massive_body* mb, Massive_body* aux_mb, Test_particle* tp, int N_mb, int N_tp, double tau) {
    kick_mb(mb, aux_mb, N_mb, tau);
    kick_tp(mb, tp, N_mb, N_tp, tau);
}


void MVS::drift_CPU(Massive_body* mb, Massive_body* aux_mb, Test_particle* tp, int N_mb, int N_tp, double tau) {
    helio_to_jacobi(mb, aux_mb, N_mb);
    drift_mb(mb, aux_mb, N_mb, tau);
    jacobi_to_helio(mb, aux_mb, N_mb);

    drift_tp_CPU(tp, mb[0].m, N_tp, tau);
}


void MVS::step_MVS_CPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau, Massive_body* aux_mb) {
    mb[0].q = Vec3d();
    mb[0].v = Vec3d();

    MVS::drift_CPU(mb, aux_mb, tp, N_mb, N_tp, tau/2);
    MVS::kick_CPU(mb, aux_mb, tp, N_mb, N_tp, tau);
    MVS::drift_CPU(mb, aux_mb, tp, N_mb, N_tp, tau/2);
}




//--------------------------------------------//
//                     GPU                    //
//--------------------------------------------//


__device__ void MVS::kick_tp_GPU(Massive_body* mb_helio, Test_particle& tp_i, int N_mb, int N_tp, double tau) {
    double dx, dy, dz, dx2, dy2, dz2, r_ij3, r_ij32, kx, ky, kz, new_vx, new_vy, new_vz;
    
    new_vx = tp_i.v.get_x();
    new_vy = tp_i.v.get_y();
    new_vz = tp_i.v.get_z();

    for (int j=1; j<N_mb; j++) {
        dx = tp_i.q.get_x() - mb_helio[j].q.get_x();
        dy = tp_i.q.get_y() - mb_helio[j].q.get_y();
        dz = tp_i.q.get_z() - mb_helio[j].q.get_z();

        r_ij3 = (dx*dx + dy*dy + dz*dz)*sqrt(dx*dx + dy*dy + dz*dz);

        dx2 = mb_helio[j].q.get_x();
        dy2 = mb_helio[j].q.get_y();
        dz2 = mb_helio[j].q.get_z();

        r_ij32 = (dx2*dx2 + dy2*dy2 + dz2*dz2)*sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);

        kx = tau*G*mb_helio[j].m*(1./r_ij3*dx + 1./r_ij32*dx2);
        ky = tau*G*mb_helio[j].m*(1./r_ij3*dy + 1./r_ij32*dy2);
        kz = tau*G*mb_helio[j].m*(1./r_ij3*dz + 1./r_ij32*dz2);


        new_vx -= kx;
        new_vy -= ky;
        new_vz -= kz;
    }
    

    tp_i.v.set_xyz(new_vx, new_vy, new_vz);
}


__device__ void MVS::drift_tp_GPU(Test_particle& tp_i, double m0, int N_tp, double tau) {
    double f, g, fdot, gdot, new_x, new_y, new_z, new_vx, new_vy, new_vz;

    gauss_func(tp_i.q, tp_i.v, G*m0, tau, f, g, fdot, gdot);


    new_x = f*tp_i.q.get_x() + g*tp_i.v.get_x();
    new_y = f*tp_i.q.get_y() + g*tp_i.v.get_y();
    new_z = f*tp_i.q.get_z() + g*tp_i.v.get_z();

    new_vx = fdot*tp_i.q.get_x() + gdot*tp_i.v.get_x();
    new_vy = fdot*tp_i.q.get_y() + gdot*tp_i.v.get_y();
    new_vz = fdot*tp_i.q.get_z() + gdot*tp_i.v.get_z();


    tp_i.q.set_xyz(new_x, new_y, new_z);
    tp_i.v.set_xyz(new_vx, new_vy, new_vz);
}


__global__ void MVS::step_tp_GPU(Massive_body* mb, Test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < N_tp) {
        drift_tp_GPU(tp[i], mb[0].m, N_tp, tau_drift);
        kick_tp_GPU(mb, tp[i], N_mb, N_tp, tau_kick);
        drift_tp_GPU(tp[i], mb[0].m, N_tp, tau_drift);
    }
}


__host__ void MVS::step_MVS_GPU(Massive_body* mb, Test_particle* tp, Massive_body* access_mb, Test_particle* access_tp, int N_mb, int N_tp, double tau, int nb_block, int nb_thread, Massive_body* aux_mb) {
    helio_to_jacobi(mb, aux_mb, N_mb);
    drift_mb(mb, aux_mb, N_mb, tau/2);
    jacobi_to_helio(mb, aux_mb, N_mb);

    cudaMemcpy(access_mb, mb, N_mb*sizeof(Massive_body), cudaMemcpyHostToDevice); // Copy the data for time-step step+1/2
    kick_mb(mb, aux_mb, N_mb, tau);

    helio_to_jacobi(mb, aux_mb, N_mb);
    drift_mb(mb, aux_mb, N_mb, tau/2);
    jacobi_to_helio(mb, aux_mb, N_mb);

    step_tp_GPU<<<nb_block, nb_thread>>>(access_mb, access_tp, N_mb, N_tp, tau, tau/2);
    cudaMemcpy(tp, access_tp, N_tp*sizeof(Test_particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}
