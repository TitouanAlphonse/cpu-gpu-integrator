#include"../headers/tools.h"


// #################
//     tools.cu
// #################


double deg_to_rad(double theta) {
    return theta*pi/180; 
}

double rad_to_deg(double theta) {
    return theta*180/pi; 
}


void orb_param_to_pos_vel(double a, double e, double i, double Omega, double omega, double M, double mu, Vec3d& q, Vec3d& v) {
    
    // Determination of E :
    double n, E, M_red, cos_E, sin_E, f, fp, fpp, fppp, dE, cos_i, sin_i, cos_Omega, sin_Omega, cos_omega, sin_omega, R_11, R_12, R_21, R_22, R_31, R_32;
    n = sqrt(mu/(a*a*a));
    M_red = (M - floor(M/(2*pi))*2*pi);

    int loop = 0, loop_max = 10;
    double dE_stop = 1e-14;
    E = M_red + sgn(sin(M_red))*0.85*e;

    do { 
        cos_E = cos(E);
        sin_E = sin(E);
        f = E - e*sin_E - M_red;
        fp = 1. - e*cos_E;
        fpp = e*sin_E;
        fppp = e*cos_E;
        if (fp != 0) {
            dE = -f/fp;
            dE = -f/(fp + 0.5*dE*fpp);
            dE = -f/(fp + 0.5*dE*fpp + 1./6.*dE*dE*fppp);
        }
        else {
            dE = 0;
        }
        E += dE;
        loop++;
    } while (abs(dE) > dE_stop && loop<loop_max);
    
    if (loop >= loop_max) {
        cout << "No convergence for the calculation of E" << endl;
    }

    cos_E = cos(E);
    sin_E = sin(E);

    Vec3d q2D = Vec3d(a*(cos_E-e), a*sqrt(1-e*e)*sin_E, 0, "xyz");
    Vec3d dq2Ddt = Vec3d(-n*a*sin_E/(1-e*cos_E), n*a*sqrt(1-e*e)*cos_E/(1-e*cos_E), 0, "xyz");

    cos_i = cos(i);
    sin_i = sin(i);
    cos_Omega = cos(Omega);
    sin_Omega = sin(Omega);
    cos_omega = cos(omega);
    sin_omega = sin(omega);

    R_11 = cos_Omega*cos_omega - sin_Omega*cos_i*sin_omega;
    R_12 = -cos_Omega*sin_omega - sin_Omega*cos_i*cos_omega;
    R_21 = sin_Omega*cos_omega + cos_Omega*cos_i*sin_omega;
    R_22 = -sin_Omega*sin_omega + cos_Omega*cos_i*cos_omega;
    R_31 = sin_i*sin_omega;
    R_32 = sin_i*cos_omega;

    q.set_x(R_11*q2D.get_x() + R_12*q2D.get_y());
    q.set_y(R_21*q2D.get_x() + R_22*q2D.get_y());
    q.set_z(R_31*q2D.get_x() + R_32*q2D.get_y());

    v.set_x(R_11*dq2Ddt.get_x() + R_12*dq2Ddt.get_y());
    v.set_y(R_21*dq2Ddt.get_x() + R_22*dq2Ddt.get_y());
    v.set_z(R_31*dq2Ddt.get_x() + R_32*dq2Ddt.get_y());
}


void pos_vel_to_orb_param(double mu, Vec3d q, Vec3d v, double& a, double& e, double& i, double& Omega, double& omega, double& M) {
    Vec3d h, n, e_vec;
    double E, nu;
    h = cross_product(q, v);
    n = Vec3d(-h.get_y(), h.get_x(), 0, "xyz");
    a = 1/(2/q.norm() - v.norm2()/mu);
    e_vec = cross_product(v, h)/mu - q/q.norm();
    e = e_vec.norm();
    i = acos(h.get_z()/h.norm());
    Omega = acos(n.get_x()/n.norm());
    if (n.get_y() < 0) {
        Omega = 2*pi - Omega;
    }
    omega = acos(scalar_product(n, e_vec)/(n.norm()*e));
    if (e_vec.get_z() < 0) {
        omega = 2*pi - omega;
    }
    nu = acos(scalar_product(e_vec, q)/(e*q.norm()));
    if (scalar_product(q, v) < 0) {
        nu = 2*pi - nu;
    }
    E = atan2(sqrt(1-e*e)*sin(nu), e+cos(nu));
    M = E - e*sin(E);
}



void helio_to_jacobi(massive_body* mb_helio, massive_body* mb_jacobi, int N_mb) {
    double sum_m = mb_helio[0].m;
    Vec3d sum_mq = mb_helio[0].m*mb_helio[0].q;
    Vec3d sum_mv = mb_helio[0].m*mb_helio[0].v;

    for (int i=1; i<N_mb; i++) {
        mb_jacobi[i].m = sum_m*mb_helio[i].m;
        mb_jacobi[i].q = mb_helio[i].q - sum_mq/sum_m;
        mb_jacobi[i].v = mb_helio[i].v - sum_mv/sum_m;

        sum_m += mb_helio[i].m;
        sum_mq += mb_helio[i].m*mb_helio[i].q;
        sum_mv += mb_helio[i].m*mb_helio[i].v;

        mb_jacobi[i].m /= sum_m;
    }

    mb_jacobi[0].m = sum_m;
    mb_jacobi[0].q = sum_mq/sum_m;
    mb_jacobi[0].v = sum_mv/sum_m;
}


void jacobi_to_helio(massive_body* mb_helio, massive_body* mb_jacobi, int N_mb) {
    double sum_m = mb_helio[0].m;
    Vec3d sum_mq = Vec3d();
    Vec3d sum_mv = Vec3d();

    mb_helio[0].q = Vec3d();
    mb_helio[0].v = Vec3d();

    mb_helio[1].q = mb_jacobi[1].q;
    mb_helio[1].v = mb_jacobi[1].v;

    for (int i=1; i<N_mb; i++) {
        mb_helio[i].q = mb_jacobi[i].q + sum_mq/sum_m;
        mb_helio[i].v = mb_jacobi[i].v + sum_mv/sum_m;

        sum_m += mb_helio[i].m;
        sum_mq += mb_helio[i].m*mb_helio[i].q;
        sum_mv += mb_helio[i].m*mb_helio[i].v;
    }

}