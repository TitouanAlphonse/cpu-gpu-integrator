#include"../headers/integrators.h"


void kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    Vec3d r_ij;
    Vec3d V_ij;

    for (int i=0; i<N_mb-1; i++) {
        for (int j=i+1; j<N_mb; j++) {
            r_ij = mb[i].q - mb[j].q;
            V_ij = G/pow(r_ij.norm(), 3)*r_ij;
            mb[i].v -= mb[j].m*tau*V_ij;
            mb[j].v += mb[i].m*tau*V_ij;
        }
    }
    for (int i=0; i<N_tp; i++) {
        for (int j=0; j<N_mb; j++) {
            r_ij = tp[i].q - mb[j].q;
            V_ij = G*mb[j].m/pow(r_ij.norm(), 3)*r_ij;
            tp[i].v -= tau*V_ij;
        }
    }
}

void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    for (int i=0; i<N_mb; i++) {
        mb[i].q += tau*mb[i].v;
    }
    for (int i=0; i<N_tp; i++) {
        tp[i].q += tau*tp[i].v;
    }
}

void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, ofstream& fich) {
    for (int i=0; i<N_mb; i++) {
        mb[i].q += tau*mb[i].v;
        fich << mb[i].q.get_x() << " " << mb[i].q.get_y() << " " << mb[i].q.get_z() << " ";
    }
    for (int i=0; i<N_tp; i++) {
        tp[i].q += tau*tp[i].v;
        fich << tp[i].q.get_x() << " " << tp[i].q.get_y() << " " << tp[i].q.get_z() << " ";
    }
    fich << endl;
}


void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep) {

    ofstream fich;

    // general data writing
    fich.open("results/general_data.txt", ios::out);

    fich << tau << " " << N_tp << endl;

    for (int i=0; i<N_mb; i++) {
        fich << mb[i].m << " " << mb[i].R << endl;
    }

    fich.close();

    // integration
	fich.open("results/positions.txt", ios::out);
    
    for (int step=0; step<Nstep; step++) {
        kick_CPU(mb, tp, N_mb, N_tp, tau/2);
        drift_CPU(mb, tp, N_mb, N_tp, tau, fich);
        kick_CPU(mb, tp, N_mb, N_tp, tau/2);
    }

    fich.close();
}