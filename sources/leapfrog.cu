#include"../headers/leapfrog.h"


//--------------------------------------------//
//                     CPU                    //
//--------------------------------------------//


void kick_mb(massive_body* mb, int N_mb, double tau) {
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
}

void kick_tp_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    Vec3d r_ij;
    Vec3d V_ij;

    for (int i=0; i<N_tp; i++) {
        for (int j=0; j<N_mb; j++) {
            r_ij = tp[i].q - mb[j].q;
            V_ij = G*mb[j].m/pow(r_ij.norm(), 3)*r_ij;
            tp[i].v -= tau*V_ij;
        }
    }
}

void kick_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    kick_mb(mb, N_mb, tau);
    kick_tp_CPU(mb, tp, N_mb, N_tp, tau);
}


void drift_mb(massive_body* mb, int N_mb, double tau) {
    for (int i=0; i<N_mb; i++) {
        mb[i].q += tau*mb[i].v;
    }
}

void drift_tp_CPU(test_particle* tp, int N_tp, double tau) {
    for (int i=0; i<N_tp; i++) {
        tp[i].q += tau*tp[i].v;
    }
}

void drift_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    drift_mb(mb, N_mb, tau);
    drift_tp_CPU(tp, N_tp, tau);
}

void step_leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    drift_CPU(mb, tp, N_mb, N_tp, tau/2);
    kick_CPU(mb, tp, N_mb, N_tp, tau);
    drift_CPU(mb, tp, N_mb, N_tp, tau/2);
}


//--------------------------------------------//
//                     GPU                    //
//--------------------------------------------//


__device__ void kick_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    double r_ij;
    if (i < N_tp) {
        for (int j=0; j<N_mb; j++) {
            r_ij = sqrt(pow(tp[i].q.get_x() - mb[j].q.get_x(),2) + pow(tp[i].q.get_y() - mb[j].q.get_y(),2) + pow(tp[i].q.get_z() - mb[j].q.get_z(),2));
            
            tp[i].v.set_x(tp[i].v.get_x() - tau*G*mb[j].m/pow(r_ij,3)*(tp[i].q.get_x() - mb[j].q.get_x()));
            tp[i].v.set_y(tp[i].v.get_y() - tau*G*mb[j].m/pow(r_ij,3)*(tp[i].q.get_y() - mb[j].q.get_y()));
            tp[i].v.set_z(tp[i].v.get_z() - tau*G*mb[j].m/pow(r_ij,3)*(tp[i].q.get_z() - mb[j].q.get_z()));
        }
    }
}

__device__ void drift_tp_GPU(test_particle* tp, int N_tp, double tau) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < N_tp) {
        tp[i].q.set_x(tp[i].q.get_x() + tau*tp[i].v.get_x());
        tp[i].q.set_y(tp[i].q.get_y() + tau*tp[i].v.get_y());
        tp[i].q.set_z(tp[i].q.get_z() + tau*tp[i].v.get_z());
    }
}

__global__ void step_tp_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau_kick, double tau_drift) {
    drift_tp_GPU(tp, N_tp, tau_drift);
    kick_tp_GPU(mb, tp, N_mb, N_tp, tau_kick);
    drift_tp_GPU(tp, N_tp, tau_drift);
}


__host__ void step_leapfrog_GPU(massive_body* mb, test_particle* tp, massive_body* access_mb, test_particle* access_tp, int N_mb, int N_tp, double tau, int nb_block, int nb_thread) {
    drift_mb(mb, N_mb, tau/2);
    cudaMemcpy(access_mb, mb, N_mb*sizeof(massive_body), cudaMemcpyHostToDevice); // Copy the data for time-step step+1/2
    kick_mb(mb, N_mb, tau);
    drift_mb(mb, N_mb, tau/2);

    step_tp_GPU<<<nb_block, nb_thread>>>(access_mb, access_tp, N_mb, N_tp, tau, tau/2);
    cudaMemcpy(tp, access_tp, N_tp*sizeof(test_particle), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}


// __device__ drift1_tp_multi_t()


__host__ void kick1_mb_multi_t(massive_body* mb_multi_t, int N_mb, double tau, int substep) {
    Vec3d r_ij;
    Vec3d V_ij;

    for (int i=0; i<N_mb; i++) {
        mb_multi_t[substep*N_mb+i].v = mb_multi_t[(substep-1)*N_mb+i].v;
    }

    for (int i=0; i<N_mb-1; i++) {
        for (int j=i+1; j<N_mb; j++) {
            r_ij = mb_multi_t[(substep-1)*N_mb+i].q - mb_multi_t[(substep-1)*N_mb+j].q;
            V_ij = G/pow(r_ij.norm(), 3)*r_ij;
            mb_multi_t[substep*N_mb+i].v -= mb_multi_t[j].m*tau*V_ij;
            mb_multi_t[substep*N_mb+j].v += mb_multi_t[i].m*tau*V_ij;
        }
    }
}

__host__ void drift_mb_multi_t(massive_body* mb_multi_t, int N_mb, double tau, int substep) {
    for (int i=0; i<N_mb; i++) {
        mb_multi_t[substep*N_mb+i].q = mb_multi_t[(substep-1)*N_mb+i].q + tau*mb_multi_t[substep*N_mb+i].v;
    }
}

__host__ void kick2_mb_multi_t(massive_body* mb_multi_t, int N_mb, double tau, int substep) {
    Vec3d r_ij;
    Vec3d V_ij;

    for (int i=0; i<N_mb-1; i++) {
        for (int j=i+1; j<N_mb; j++) {
            r_ij = mb_multi_t[substep*N_mb+i].q - mb_multi_t[substep*N_mb+j].q;
            V_ij = G/pow(r_ij.norm(), 3)*r_ij;
            mb_multi_t[substep*N_mb+i].v -= mb_multi_t[j].m*tau*V_ij;
            mb_multi_t[substep*N_mb+j].v += mb_multi_t[i].m*tau*V_ij;
        }
    }
}


__host__ void writing_multi_t(massive_body* mb_multi_t, test_particle* tp_multi_t, int N_mb, int N_tp, int N_substep, ofstream& fich) {
    for (int substep=1; substep<=N_substep; substep++) {
        for (int i=0; i<N_mb; i++) {
            fich << mb_multi_t[substep*N_mb+i].q.get_x() << " " << mb_multi_t[substep*N_mb+i].q.get_y() << " " << mb_multi_t[substep*N_mb+i].q.get_z() << " ";
        }
        for (int i=0; i<N_tp; i++) {
            fich << tp_multi_t[substep*N_tp+i].q.get_x() << " " << tp_multi_t[substep*N_tp+i].q.get_y() << " " << tp_multi_t[substep*N_tp+i].q.get_z() << " ";
        }
        fich << endl;
    }
}



__global__ void step_tp_GPU_multi_t(massive_body* mb_multi_t, test_particle* tp_multi_t, int N_mb, int N_tp, double tau_kick, double tau_drift, int N_substep) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int i_ssm1, i_ss, j_ss, j_ssm1;
    double r_ij;
    if (i < N_tp) {
        for (int substep=1; substep<=N_substep; substep++) {
            i_ssm1 = (substep-1)*N_tp + i;
            i_ss = substep*N_tp + i;

            // Kick 1

            tp_multi_t[i_ss].v.set_x(tp_multi_t[i_ssm1].v.get_x());
            tp_multi_t[i_ss].v.set_y(tp_multi_t[i_ssm1].v.get_y());
            tp_multi_t[i_ss].v.set_z(tp_multi_t[i_ssm1].v.get_z());

            for (int j=0; j<N_mb; j++) {
                j_ssm1 = (substep-1)*N_mb + j;
                r_ij = sqrt(pow(tp_multi_t[i_ssm1].q.get_x() - mb_multi_t[j_ssm1].q.get_x(),2) + pow(tp_multi_t[i_ssm1].q.get_y() - mb_multi_t[j_ssm1].q.get_y(),2) + pow(tp_multi_t[i_ssm1].q.get_z() - mb_multi_t[j_ssm1].q.get_z(),2));

                tp_multi_t[i_ss].v.set_x(tp_multi_t[i_ss].v.get_x() - tau_kick*G*mb_multi_t[j_ssm1].m/pow(r_ij,3)*(tp_multi_t[i_ssm1].q.get_x() - mb_multi_t[j_ssm1].q.get_x()));
                tp_multi_t[i_ss].v.set_y(tp_multi_t[i_ss].v.get_y() - tau_kick*G*mb_multi_t[j_ssm1].m/pow(r_ij,3)*(tp_multi_t[i_ssm1].q.get_y() - mb_multi_t[j_ssm1].q.get_y()));
                tp_multi_t[i_ss].v.set_z(tp_multi_t[i_ss].v.get_z() - tau_kick*G*mb_multi_t[j_ssm1].m/pow(r_ij,3)*(tp_multi_t[i_ssm1].q.get_z() - mb_multi_t[j_ssm1].q.get_z()));
            }


            // Drift

            tp_multi_t[i_ss].q.set_x(tp_multi_t[i_ssm1].q.get_x() + tau_drift*tp_multi_t[i_ss].v.get_x());
            tp_multi_t[i_ss].q.set_y(tp_multi_t[i_ssm1].q.get_y() + tau_drift*tp_multi_t[i_ss].v.get_y());
            tp_multi_t[i_ss].q.set_z(tp_multi_t[i_ssm1].q.get_z() + tau_drift*tp_multi_t[i_ss].v.get_z());
            

            // Kick 2

            for (int j=0; j<N_mb; j++) {
                j_ss = substep*N_mb + j;
                r_ij = sqrt(pow(tp_multi_t[i_ss].q.get_x() - mb_multi_t[j_ss].q.get_x(),2) + pow(tp_multi_t[i_ss].q.get_y() - mb_multi_t[j_ss].q.get_y(),2) + pow(tp_multi_t[i_ss].q.get_z() - mb_multi_t[j_ss].q.get_z(),2));
                
                tp_multi_t[i_ss].v.set_x(tp_multi_t[i_ss].v.get_x() - tau_kick*G*mb_multi_t[j_ss].m/pow(r_ij,3)*(tp_multi_t[i_ss].q.get_x() - mb_multi_t[j_ss].q.get_x()));
                tp_multi_t[i_ss].v.set_y(tp_multi_t[i_ss].v.get_y() - tau_kick*G*mb_multi_t[j_ss].m/pow(r_ij,3)*(tp_multi_t[i_ss].q.get_y() - mb_multi_t[j_ss].q.get_y()));
                tp_multi_t[i_ss].v.set_z(tp_multi_t[i_ss].v.get_z() - tau_kick*G*mb_multi_t[j_ss].m/pow(r_ij,3)*(tp_multi_t[i_ss].q.get_z() - mb_multi_t[j_ss].q.get_z()));
            }
        }
        // Initialization for the next call :
        tp_multi_t[i] = tp_multi_t[N_substep*N_tp+i];
    }
}


__host__ void leapfrog_GPU_multi_t(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int N_substep, int nb_block, int nb_thread, string suffix) {

    cout << "------------------------" << endl;
    cout << "Leapfrog GPU multi-step" << endl;
    cout << "------------------------" << endl << endl;
    cout << "Integration performed over " << N_step*tau << " years" << endl;
    cout << "time-step : " << tau/day_in_years << " days (Total number of steps : " << N_step << ")" << endl;
    cout << "Number of sub-steps within a  step : " << N_substep << endl;
    cout << N_mb << " massive bodies, " << N_tp << " test-particles" << endl << endl;

    cout << "Initialization... " << flush;

    auto start = chrono::high_resolution_clock::now();

    int size_mb_multi_t = (N_substep+1)*N_mb*sizeof(massive_body);
    int size_tp_multi_t = (N_substep+1)*N_tp*sizeof(test_particle);

    massive_body *mb_multi_t = (massive_body*)malloc(size_mb_multi_t);
    test_particle *tp_multi_t = (test_particle*)malloc(size_tp_multi_t);

    // General data writing and initializations :

    ofstream fich_general;
    ofstream fich_pos;

    fich_general.open("outputs/general_data"+suffix+".txt", ios::out);
    fich_pos.open("outputs/positions"+suffix+".txt", ios::out);

    fich_general << tau << " " << N_tp << endl;

    for (int i=0; i<N_mb; i++) {
        fich_general << mb[i].m << " " << mb[i].R << endl;
        fich_pos << mb[i].q.get_x() << " " << mb[i].q.get_y() << " " << mb[i].q.get_z() << " ";
        mb_multi_t[i] = mb[i];
    }

    fich_general.close();

    for (int i=0; i<N_tp; i++) {
        fich_pos << tp[i].q.get_x() << " " << tp[i].q.get_y() << " " << tp[i].q.get_z() << " ";
        tp_multi_t[i] = tp[i];
    }
    fich_pos << endl;

    for (int step=1; step<=N_substep; step++) {
        for (int i=0; i<N_mb; i++) {
            mb_multi_t[step*N_mb+i] = massive_body(mb[i].m, mb[i].R, Vec3d(), Vec3d());
        }
        for (int i=0; i<N_tp; i++) {
            tp_multi_t[step*N_tp+i] = tp[i];
        }
    }


    massive_body *access_mb_multi_t;
    test_particle *access_tp_multi_t;

    cudaMalloc((void **) &access_mb_multi_t, size_mb_multi_t);
    cudaMalloc((void **) &access_tp_multi_t, size_tp_multi_t);

    cudaMemcpy(access_tp_multi_t, tp_multi_t, size_tp_multi_t, cudaMemcpyHostToDevice);

    cout << "Done" << endl;
    cout << "Integration... " << endl;

    // Integration :
    
    for (int step=0; step<N_step/N_substep; step++) {

        // Massive bodies :
        for (int substep=1; substep<=N_substep; substep++) {
            kick1_mb_multi_t(mb_multi_t, N_mb, tau/2, substep);
            drift_mb_multi_t(mb_multi_t, N_mb, tau, substep);
            kick2_mb_multi_t(mb_multi_t, N_mb, tau/2, substep);
        }
        // Initialization for the next call :
        for (int i=0; i<N_mb; i++) {
            mb_multi_t[i] = mb_multi_t[N_substep*N_mb+i];
        }

        // Test particles :
        cudaMemcpy(access_mb_multi_t, mb_multi_t, size_mb_multi_t, cudaMemcpyHostToDevice);
        step_tp_GPU_multi_t<<<nb_block, nb_thread>>>(access_mb_multi_t, access_tp_multi_t, N_mb, N_tp, tau/2, tau, N_substep);
        cudaMemcpy(tp_multi_t, access_tp_multi_t, size_tp_multi_t, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        writing_multi_t(mb_multi_t, tp_multi_t, N_mb, N_tp, N_substep, fich_pos);
    }

    fich_pos.close();

    free(mb_multi_t);
    free(tp_multi_t);
    cudaFree(access_tp_multi_t);
    cudaFree(access_mb_multi_t);

    auto end = chrono::high_resolution_clock::now();

    cout << "Done" << endl;
    cout << "Simulation complete" << endl << endl;

    cout << "General data written in : outputs/general_data" << suffix << ".txt" << endl;
    cout << "Position data written in  : outputs/positions" << suffix << ".txt" << endl;
    get_time(start, end);
    cout << endl;
}

__host__ void leapfrog_GPU_multi_t(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int N_substep, int nb_block, int nb_thread) {
    leapfrog_GPU_multi_t(mb, tp, N_mb, N_tp, tau, N_step, N_substep, nb_block, nb_thread);
}