#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"

void kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, ofstream& fich);

void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep);