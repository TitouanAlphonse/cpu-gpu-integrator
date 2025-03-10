#include"../headers/integrators.h"


//--------------------------------------------//
//                     CPU                    //
//--------------------------------------------//


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
    auto start = chrono::high_resolution_clock::now();
    for (int i=0; i<N_tp; i++) {
        tp[i].q += tau*tp[i].v;
    }
    auto end = chrono::high_resolution_clock::now();

    get_time(start,end);
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


void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, string suffix) {

    ofstream fich;

    // general data writing
    fich.open("results/general_data"+suffix+".txt", ios::out);

    fich << tau << " " << N_tp << endl;

    for (int i=0; i<N_mb; i++) {
        fich << mb[i].m << " " << mb[i].R << endl;
    }

    fich.close();

    // integration
	fich.open("results/positions"+suffix+".txt", ios::out);
    
    for (int step=0; step<Nstep; step++) {
        kick_CPU(mb, tp, N_mb, N_tp, tau/2);
        drift_CPU(mb, tp, N_mb, N_tp, tau, fich);
        kick_CPU(mb, tp, N_mb, N_tp, tau/2);
    }

    fich.close();
}


void leapfrog_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep) {
    leapfrog_CPU(mb, tp, N_mb, N_tp, tau, Nstep, "");
}



//--------------------------------------------//
//                     GPU                    //
//--------------------------------------------//


__host__ void kick_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int nb_block, int nb_threads) {
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

    int size_tp = N_tp*sizeof(test_particle);
    int size_mb = N_mb*sizeof(massive_body);

    test_particle *access_tp;
    massive_body *access_mb;

    cudaMalloc((void **) &access_tp, size_tp);
    cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &access_mb, size_mb);
    cudaMemcpy(access_mb, mb, size_mb, cudaMemcpyHostToDevice);

    kick_tp<<<nb_block, nb_threads>>>(access_mb, access_tp, N_mb, N_tp, tau);

    cudaMemcpy(tp, access_tp, size_tp, cudaMemcpyDeviceToHost);
    cudaMemcpy(mb, access_mb, size_mb, cudaMemcpyDeviceToHost);

    cudaFree(access_tp);
    cudaFree(access_mb);
}

__global__ void kick_tp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
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


__host__ void drift_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int nb_block, int nb_threads) {
    
    for (int i=0; i<N_mb; i++) {
        mb[i].q += tau*mb[i].v;
    }

    int size_tp = N_tp*sizeof(test_particle);
    int size_mb = N_mb*sizeof(massive_body);

    test_particle *access_tp;
    massive_body *access_mb;

    cudaMalloc((void **) &access_tp, size_tp);
    cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &access_mb, size_mb);
    cudaMemcpy(access_mb, mb, size_mb, cudaMemcpyHostToDevice);

    drift_tp<<<nb_block, nb_threads>>>(access_mb, access_tp, N_mb, N_tp, tau);

    cudaMemcpy(tp, access_tp, size_tp, cudaMemcpyDeviceToHost);
    cudaMemcpy(mb, access_mb, size_mb, cudaMemcpyDeviceToHost);

    cudaFree(access_tp);
    cudaFree(access_mb);
}

__global__ void drift_tp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < N_tp) {
        tp[i].q.set_x(tp[i].q.get_x() + tau*tp[i].v.get_x());
        tp[i].q.set_y(tp[i].q.get_y() + tau*tp[i].v.get_y());
        tp[i].q.set_z(tp[i].q.get_z() + tau*tp[i].v.get_z());
    }
}


__host__ void writing(massive_body* mb, test_particle* tp, int N_mb, int N_tp, ofstream& fich) {
    for (int i=0; i<N_mb; i++) {
        fich << mb[i].q.get_x() << " " << mb[i].q.get_y() << " " << mb[i].q.get_z() << " ";
    }
    for (int i=0; i<N_tp; i++) {
        fich << tp[i].q.get_x() << " " << tp[i].q.get_y() << " " << tp[i].q.get_z() << " ";
    }
    fich << endl;
}

__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, int nb_block, int nb_threads, string suffix) {

    ofstream fich;

    // general data writing :

    fich.open("results/general_data"+suffix+".txt", ios::out);

    fich << tau << " " << N_tp << endl;

    for (int i=0; i<N_mb; i++) {
        fich << mb[i].m << " " << mb[i].R << endl;
    }

    fich.close();


    // integration :

	fich.open("results/positions"+suffix+".txt", ios::out);
    
    for (int step=0; step<Nstep; step++) {
        kick_GPU(mb, tp, N_mb, N_tp, tau/2, nb_block, nb_threads);
        drift_GPU(mb, tp, N_mb, N_tp, tau, nb_block, nb_threads);
        kick_GPU(mb, tp, N_mb, N_tp, tau/2, nb_block, nb_threads);
        writing(mb, tp, N_mb, N_tp, fich);
    }

    fich.close();
}

__host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, int nb_block, int nb_threads) {
    leapfrog_GPU(mb, tp, N_mb, N_tp, tau, Nstep, nb_block, nb_threads, "");
}


// __host__ void kick_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int nb_block, int nb_threads) {
//     Vec3d r_ij;
//     Vec3d V_ij;

//     for (int i=0; i<N_mb-1; i++) {
//         for (int j=i+1; j<N_mb; j++) {
//             r_ij = mb[i].q - mb[j].q;
//             V_ij = G/pow(r_ij.norm(), 3)*r_ij;
//             mb[i].v -= mb[j].m*tau*V_ij;
//             mb[j].v += mb[i].m*tau*V_ij;
//         }
//     }

//     int size_tp = N_tp*sizeof(test_particle);
//     int size_mb = N_mb*sizeof(massive_body);

//     test_particle *access_tp;
//     massive_body *access_mb;

//     cudaMalloc((void **) &access_tp, size_tp);
//     cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);

//     cudaMalloc((void **) &access_mb, size_mb);
//     cudaMemcpy(access_mb, mb, size_mb, cudaMemcpyHostToDevice);

//     kick_tp<<<nb_block, nb_threads>>>(access_mb, access_tp, N_mb, N_tp, tau);

//     cudaMemcpy(tp, access_tp, size_tp, cudaMemcpyDeviceToHost);
//     cudaMemcpy(mb, access_mb, size_mb, cudaMemcpyDeviceToHost);

//     cudaFree(access_tp);
//     cudaFree(access_mb);
// }

// __global__ void kick_tp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
//     int i = threadIdx.x + blockDim.x*blockIdx.x;
//     double r_ij;
//     if (i < N_tp) {
//         for (int j=0; j<N_mb; j++) {
//             r_ij = sqrt(pow(tp[i].q.get_x() - mb[j].q.get_x(),2) + pow(tp[i].q.get_y() - mb[j].q.get_y(),2) + pow(tp[i].q.get_z() - mb[j].q.get_z(),2));
            
//             tp[i].v.set_x(tp[i].v.get_x() - tau*G*mb[j].m/pow(r_ij,3)*(tp[i].q.get_x() - mb[j].q.get_x()));
//             tp[i].v.set_y(tp[i].v.get_y() - tau*G*mb[j].m/pow(r_ij,3)*(tp[i].q.get_y() - mb[j].q.get_y()));
//             tp[i].v.set_z(tp[i].v.get_z() - tau*G*mb[j].m/pow(r_ij,3)*(tp[i].q.get_z() - mb[j].q.get_z()));
//         }
//     }
// }


// __host__ void drift_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int nb_block, int nb_threads) {
    
//     for (int i=0; i<N_mb; i++) {
//         mb[i].q += tau*mb[i].v;
//     }

//     int size_tp = N_tp*sizeof(test_particle);
//     int size_mb = N_mb*sizeof(massive_body);

//     test_particle *access_tp;
//     massive_body *access_mb;

//     cudaMalloc((void **) &access_tp, size_tp);
//     cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);

//     cudaMalloc((void **) &access_mb, size_mb);
//     cudaMemcpy(access_mb, mb, size_mb, cudaMemcpyHostToDevice);

//     drift_tp<<<nb_block, nb_threads>>>(access_mb, access_tp, N_mb, N_tp, tau);

//     cudaMemcpy(tp, access_tp, size_tp, cudaMemcpyDeviceToHost);
//     cudaMemcpy(mb, access_mb, size_mb, cudaMemcpyDeviceToHost);

//     cudaFree(access_tp);
//     cudaFree(access_mb);
// }

// __global__ void drift_tp(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau) {
//     int i = threadIdx.x + blockDim.x*blockIdx.x;
//     if (i < N_tp) {
//         tp[i].q.set_x(tp[i].q.get_x() + tau*tp[i].v.get_x());
//         tp[i].q.set_y(tp[i].q.get_y() + tau*tp[i].v.get_y());
//         tp[i].q.set_z(tp[i].q.get_z() + tau*tp[i].v.get_z());
//     }
// }


// __host__ void writing(massive_body* mb, test_particle* tp, int N_mb, int N_tp, ofstream& fich) {
//     for (int i=0; i<N_mb; i++) {
//         fich << mb[i].q.get_x() << " " << mb[i].q.get_y() << " " << mb[i].q.get_z() << " ";
//     }
//     for (int i=0; i<N_tp; i++) {
//         fich << tp[i].q.get_x() << " " << tp[i].q.get_y() << " " << tp[i].q.get_z() << " ";
//     }
//     fich << endl;
// }

// __host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, int nb_block, int nb_threads, string suffix) {

//     ofstream fich;

//     // general data writing :

//     fich.open("results/general_data"+suffix+".txt", ios::out);

//     fich << tau << " " << N_tp << endl;

//     for (int i=0; i<N_mb; i++) {
//         fich << mb[i].m << " " << mb[i].R << endl;
//     }

//     fich.close();


//     // integration :

// 	fich.open("results/positions"+suffix+".txt", ios::out);
    
//     for (int step=0; step<Nstep; step++) {
//         kick_GPU(mb, tp, N_mb, N_tp, tau/2, nb_block, nb_threads);
//         drift_GPU(mb, tp, N_mb, N_tp, tau, nb_block, nb_threads);
//         kick_GPU(mb, tp, N_mb, N_tp, tau/2, nb_block, nb_threads);
//         writing(mb, tp, N_mb, N_tp, fich);
//     }

//     fich.close();
// }

// __host__ void leapfrog_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int Nstep, int nb_block, int nb_threads) {
//     leapfrog_GPU(mb, tp, N_mb, N_tp, tau, Nstep, nb_block, nb_threads, "");
// }