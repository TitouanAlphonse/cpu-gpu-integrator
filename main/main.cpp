#include"../headers/global.h"
#include"../headers/Vec3d.h"
#include"../headers/massive_body.h"
#include"../headers/integrators.h"

int main() {

    random_device rd;

    uniform_real_distribution<> uniform(0, 1);;

    int Nstep = 10000;
    double tau = 10*day_in_years;

    int N_mb = 5;
    int N_tp = 200;

    massive_body *mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
    test_particle *tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

    double angle = 0;

    mb[0] = massive_body(1, 1, Vec3d(), Vec3d());
    mb[1] = massive_body(in_SM(M_Jupiter), in_SR(R_Jupiter), Vec3d(dist_Jupiter/au, pi/2, angle, "spheric"), Vec3d(v_Jupiter, pi/2, angle + pi/2, "spheric"));
    mb[2] = massive_body(in_SM(M_Saturn), in_SR(R_Saturn), Vec3d(dist_Saturn/au, pi/2, angle, "spheric"), Vec3d(v_Saturn, pi/2, angle + pi/2, "spheric"));
    mb[3] = massive_body(in_SM(M_Uranus), in_SR(R_Uranus), Vec3d(dist_Uranus/au, pi/2, angle, "spheric"), Vec3d(v_Uranus, pi/2, angle + pi/2, "spheric"));
    mb[4] = massive_body(in_SM(M_Neptune), in_SR(R_Neptune), Vec3d(dist_Neptune/au, pi/2, angle, "spheric"), Vec3d(v_Neptune, pi/2, angle + pi/2, "spheric"));

    Vec3d q;
    Vec3d v;
    test_particle particle;

    for (int i=0; i<N_tp; i++) {
        q = Vec3d((1 + 29*uniform(rd)), pi/2 + (1-2*uniform(rd))*pi/8, uniform(rd)*2*pi, "spheric");
        v = -cross_product(q, Vec3d(0,0,1,"xyz"))*(0.9 + 0.2*uniform(rd))*sqrt(G*pow(au/year,2)/q.norm())/q.norm();
        v.set_theta(v.get_theta() + (1-2*uniform(rd))*pi/8);

        particle = test_particle(q, v);
        tp[i] = particle;
    }

    auto start = chrono::high_resolution_clock::now();
    leapfrog_CPU(mb, tp, N_mb, N_tp, tau, Nstep);
    auto end = chrono::high_resolution_clock::now();

    get_time(start, end);

    free(mb);
    free(tp);

    return 0;
}