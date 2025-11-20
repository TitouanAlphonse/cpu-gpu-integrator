#ifndef TOOLS_H
#define TOOLS_H

#include"global.h"
#include"Vec3d.h"
#include"Massive_body.h"
#include"Test_particle.h"

template <typename T>
int sgn(T val) {
    return (val < T(0)) ? -1 : 1;
}

double deg_to_rad(double theta);
double rad_to_deg(double theta);

void orb_param_to_pos_vel(double a, double e, double i, double Omega, double omega, double M, double mu, Vec3d& q, Vec3d& v);
void pos_vel_to_orb_param(double mu, Vec3d q, Vec3d v, double& a, double& e, double& i, double& Omega, double& omega, double& M);

void helio_to_jacobi(Massive_body* mb_helio, Massive_body* mb_jacobi, int N_mb);
void jacobi_to_helio(Massive_body* mb_helio, Massive_body* mb_jacobi, int N_mb);


#endif