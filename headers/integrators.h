#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"

void kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);
void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, ofstream& fich);

void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, string suffix);
void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep);


__host__ void kick_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int nb_block, int nb_threads);
__global__ void kick_tp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);

__host__ void drift_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int nb_block, int nb_threads);
__global__ void drift_tp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau);

__host__ void writing(massive_body* mb, test_particle* tp, int N_mb, int N_tp, ofstream& fich);

__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, int nb_block, int nb_threads, string suffix);
__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, int nb_block, int nb_threads);