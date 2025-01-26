#include"../headers/integrators.h"


void leapfrog_2_bodies(massive_body& S, massive_body& P, double tau, int Nstep) {

    ofstream fich;
	fich.open("results/positions.txt", ios::out);
    
    Vec3d S_p_half, P_p_half, V;
    
    for (int step=0; step<Nstep; step++){
        // cout << "Step n°" << step << " :" << endl;

        // Step + 1/2

        V = G*S.m*P.m/pow((S.q - P.q).norm(), 3)*(S.q - P.q);

        P_p_half = P.p + tau/2*V;
        S_p_half = S.p + tau/2*(-V);

        // step + 1

        P.q += tau*P_p_half/P.m;
        S.q += tau*S_p_half/S.m;

        V = G*S.m*P.m/pow((S.q - P.q).norm(), 3)*(S.q - P.q);


        P.p = P_p_half + tau/2*V;
        S.p = S_p_half + tau/2*(-V);


        fich << S.q.get_x()/au << " " << S.q.get_y()/au << " " << S.q.get_z()/au << " " << P.q.get_x()/au << " " << P.q.get_y()/au << " " << P.q.get_z()/au << endl; 

        // cout << "Sun :" << endl;
        // S.print();
        // cout << endl;
        // cout << "Planet :" << endl;
        // P.print();
        // cout << endl << "--------" << endl << endl;
    }
}


void leapfrog1(massive_body* bodies, int Nbodies, double tau, int Nstep) {
    ofstream fich;

    fich.open("results/general_info.txt", ios::out);

    fich << tau << " " << Nstep << endl;

    for (int i=0; i<Nbodies; i++) {
        fich << bodies[i].m << " " << bodies[i].R << endl;
    }

    fich.close();


	fich.open("results/positions.txt", ios::out);

    Vec3d r_ij;
    Vec3d *p_half = (Vec3d*)malloc(Nbodies*sizeof(Vec3d));
    
    for (int step=0; step<Nstep; step++) {
        // cout << "Step n°" << step << " :" << endl;

        // step + 1/2

        for (int i=0; i<Nbodies; i++) {
            p_half[i] = bodies[i].p;

            for (int j=0; j<Nbodies; j++) {
                if (i != j) {
                    r_ij = bodies[i].q - bodies[j].q;
                    p_half[i] -= tau/2*G*bodies[i].m*bodies[j].m/pow(r_ij.norm(), 3)*r_ij;
                }
            }
        }

        // step + 1

        for (int i=0; i<Nbodies; i++) {
            bodies[i].q += tau*p_half[i]/bodies[i].m;
            fich << bodies[i].q.get_x()/au << " " << bodies[i].q.get_y()/au << " " << bodies[i].q.get_z()/au << " ";
        }
        fich << endl;

        for (int i=0; i<Nbodies; i++) {
            bodies[i].p = p_half[i];

            for (int j=0; j<Nbodies; j++) {
                if (i != j) {
                    r_ij = bodies[i].q - bodies[j].q;
                    bodies[i].p -= tau/2*G*bodies[i].m*bodies[j].m/pow(r_ij.norm(), 3)*r_ij;
                }
            }
        }
    }

    fich.close();
    free(p_half);
}

void leapfrog2(massive_body* bodies, int Nbodies, double tau, int Nstep) {

    Vec3d r_ij;
    Vec3d V_ij;
    
    ofstream fich;

    fich.open("results/general_info.txt", ios::out);

    fich << tau << " " << Nstep << endl;

    for (int i=0; i<Nbodies; i++) {
        fich << bodies[i].m << " " << bodies[i].R << endl;
    }

    fich.close();

	fich.open("results/positions.txt", ios::out);
    
    for (int step=0; step<Nstep; step++) {
        // cout << "Step n°" << step << " :" << endl;

        // step + 1/2

        for (int i=0; i<Nbodies-1; i++) {
            for (int j=i+1; j<Nbodies; j++) {
                r_ij = bodies[i].q - bodies[j].q;
                V_ij = G*bodies[i].m*bodies[j].m/pow(r_ij.norm(), 3)*r_ij;
                bodies[i].p -= tau/2*V_ij;
                bodies[j].p += tau/2*V_ij;
            }
        }

        // step + 1

        for (int i=0; i<Nbodies; i++) {
            bodies[i].q += tau*bodies[i].p/bodies[i].m;
            fich << bodies[i].q.get_x()/au << " " << bodies[i].q.get_y()/au << " " << bodies[i].q.get_z()/au << " ";
        }
        fich << endl;

        for (int i=0; i<Nbodies-1; i++) {
            for (int j=i+1; j<Nbodies; j++) {
                r_ij = bodies[i].q - bodies[j].q;
                V_ij = G*bodies[i].m*bodies[j].m/pow(r_ij.norm(), 3)*r_ij;
                bodies[i].p -= tau/2*V_ij;
                bodies[j].p += tau/2*V_ij;
            }
        }
    }

    fich.close();
}


void leapfrog_mbtp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep) {

    Vec3d r_ij;
    Vec3d V_ij;
    
    ofstream fich;

    fich.open("results/general_info.txt", ios::out);

    fich << tau << " " << N_mb << endl;

    for (int i=0; i<N_mb; i++) {
        fich << mb[i].m << " " << mb[i].R << endl;
    }

    fich.close();

	fich.open("results/positions.txt", ios::out);
    
    for (int step=0; step<Nstep; step++) {
        // cout << "Step n°" << step << " :" << endl;

        // step + 1/2

        for (int i=0; i<N_mb-1; i++) {
            for (int j=i+1; j<N_mb; j++) {
                r_ij = mb[i].q - mb[j].q;
                V_ij = G*mb[i].m*mb[j].m/pow(r_ij.norm(), 3)*r_ij;
                mb[i].p -= tau/2*V_ij;
                mb[j].p += tau/2*V_ij;
            }
        }
        for (int i=0; i<N_tp; i++) {
            for (int j=0; j<N_mb; j++) {
                r_ij = tp[i].q - mb[j].q;
                V_ij = G*mb[j].m/pow(r_ij.norm(), 3)*r_ij;
                tp[i].v -= tau/2*V_ij;
            }
        }

        // step + 1

        for (int i=0; i<N_mb; i++) {
            mb[i].q += tau*mb[i].p/mb[i].m;
            fich << mb[i].q.get_x()/au << " " << mb[i].q.get_y()/au << " " << mb[i].q.get_z()/au << " ";
        }
        for (int i=0; i<N_tp; i++) {
            tp[i].q += tau*tp[i].v;
            fich << tp[i].q.get_x()/au << " " << tp[i].q.get_y()/au << " " << tp[i].q.get_z()/au << " ";
        }
        fich << endl;

        for (int i=0; i<N_mb-1; i++) {
            for (int j=i+1; j<N_mb; j++) {
                r_ij = mb[i].q - mb[j].q;
                V_ij = G*mb[i].m*mb[j].m/pow(r_ij.norm(), 3)*r_ij;
                mb[i].p -= tau/2*V_ij;
                mb[j].p += tau/2*V_ij;
            }
        }
        for (int i=0; i<N_tp; i++) {
            for (int j=0; j<N_mb; j++) {
                r_ij = tp[i].q - mb[j].q;
                V_ij = G*mb[j].m/pow(r_ij.norm(), 3)*r_ij;
                tp[i].v -= tau/2*V_ij;
            }
        }
    }

    fich.close();
}