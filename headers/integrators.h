#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"

void leapfrog_2_bodies(massive_body& S, massive_body& P, double tau, int Nstep);
void leapfrog1(massive_body* bodies, int Nbodies, double tau, int Nstep);
void leapfrog2(massive_body* bodies, int Nbodies, double tau, int Nstep);
void leapfrog3(massive_body* bodies, int Nbodies, double tau, int Nstep);
void leapfrog_mbtp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep);