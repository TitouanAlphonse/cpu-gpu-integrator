#include"../headers/integration.h"


void print_info_start(string integration_method, string integration_mode, int N_mb, int N_tp, int N_step, double computation_time, bool def_N_step_in_cp_time, double tau, double freq_w, bool def_freq_w_in_steps) {
    if (integration_method == "leapfrog") {
        cout << "-------------" << endl;
        cout << "Leapfrog ";
        if (integration_mode == "cpu") {cout << "CPU" << endl;}
        if (integration_mode == "gpu") {cout << "GPU" << endl;}
        if (integration_mode == "gpu_multi-step") {cout << "GPU multi-step" << endl;}
        cout << "-------------" << endl << endl;
    }
    
    if (def_N_step_in_cp_time) {
        cout << "Computation time : " << computation_time << " minutes" << endl;
    }
    else {
        cout << "Integration performed over " << N_step*tau << " years" << endl;
        cout << "time-step : " << tau/day_in_years << " days (Total number of steps : " << N_step << ")" << endl;
    }
    cout << N_mb << " massive bodies, " << N_tp << " test-particles" << endl << endl;
    cout << "Writing positions every : ";
    if (def_freq_w_in_steps) {
        cout << (int) freq_w << " steps" << endl;
    }
    else {
        cout << freq_w << " % of the total simulation" << endl; 
    }

    cout << endl;
    cout << "Initialization... " << flush;
}


void print_info_end(int N_step, bool def_N_step_in_cp_time, double tau, string suffix) {
    cout << "Done" << endl;
    cout << "Simulation complete" << endl << endl;
    if (def_N_step_in_cp_time) {
        cout << "Integration performed over " << N_step*tau << " years (Total number of steps : " << N_step << ")" << endl << endl;
    }
    cout << "General data written in : outputs/general_data" << suffix << ".txt" << endl;
    cout << "Position data written in  : outputs/positions" << suffix << ".txt" << endl << endl;
}

__host__ void write_init(ofstream& file_general, ofstream& file_pos, double tau, double freq_w, massive_body* mb, test_particle* tp, int N_mb, int N_tp) {

    // file_general << tau << " " << N_tp << endl;
    // file_pos << 0 << " ";

    // for (int i=0; i<N_mb; i++) {
    //     file_general << mb[i].m << " " << mb[i].R << endl;
    //     file_pos << mb[i].q.get_x() << " " << mb[i].q.get_y() << " " << mb[i].q.get_z() << " ";
    // }

    // for (int i=0; i<N_tp; i++) {
    //     file_pos << tp[i].q.get_x() << " " << tp[i].q.get_y() << " " << tp[i].q.get_z() << " ";
    // }
    // file_pos << endl;

    file_general << tau << " " << N_tp << endl;

    for (int i=0; i<N_mb; i++) {
        file_general << mb[i].m << " " << mb[i].R << endl;
    }

    write_positions(file_pos, 0, mb, tp, N_mb, N_tp);
}

__host__ void write_positions(ofstream& file_pos, int step, massive_body* mb, test_particle* tp, int N_mb, int N_tp) {
    file_pos << step << " " ;
    for (int i=0; i<N_mb; i++) {
        file_pos << mb[i].q.get_x() << " " << mb[i].q.get_y() << " " << mb[i].q.get_z() << " ";
    }
    for (int i=0; i<N_tp; i++) {
        file_pos << tp[i].q.get_x() << " " << tp[i].q.get_y() << " " << tp[i].q.get_z() << " ";
    }
    file_pos << endl;
}


int integration_id(string integration_method, string integration_mode, bool def_N_step_in_cp_time) {
    int id;
    if (integration_mode == "cpu" && !def_N_step_in_cp_time) {id = 1;}
    if (integration_mode == "cpu" && def_N_step_in_cp_time) {id = 2;}
    if (integration_mode == "gpu" && !def_N_step_in_cp_time) {id = 3;}
    if (integration_mode == "gpu" && def_N_step_in_cp_time) {id = 4;}
    if (integration_mode == "gpu_multi-step" && !def_N_step_in_cp_time) {id = 5;}
    if (integration_mode == "gpu_multi-step" && def_N_step_in_cp_time) {id = 6;}

    if (integration_method == "leapfrog") {
        return id;
    }

    return -1;
}

void launch_integration(string integration_method, string integration_mode, massive_body* mb,test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, bool def_N_step_in_cp_time, int nb_block, int nb_thread, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps) {
    switch (integration_id(integration_method, integration_mode, def_N_step_in_cp_time)) {
    case 1:
        integration_CPU(mb, tp, N_mb, N_tp, tau, N_step, step_leapfrog_CPU, file_pos, freq_w, def_freq_w_in_steps);
        break;
    case 2:
        integration_CPU(mb, tp, N_mb, N_tp, tau, N_step, computation_time, step_leapfrog_CPU, file_pos, freq_w, def_freq_w_in_steps);
        break;
    case 3:
        integration_GPU(mb, tp, N_mb, N_tp, tau, N_step, nb_block, nb_thread, step_leapfrog_GPU, file_pos, freq_w, def_freq_w_in_steps);
        break;
    case 4:
        integration_GPU(mb, tp, N_mb, N_tp, tau, N_step, computation_time, nb_block, nb_thread, step_leapfrog_GPU, file_pos, freq_w, def_freq_w_in_steps);
        break;
    default:
        break;
    }
}

__host__ void integration_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, step_function_CPU step_update, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    int freq_w_eff = (int)freq_w;
    if (!def_freq_w_in_steps) {
        freq_w_eff = (int)(N_step*freq_w/100);
    }
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    for (int step=1; step<=N_step; step++) {
        step_update(mb, tp, N_mb, N_tp, tau);
        if (step%freq_w_eff == 0) {
            write_positions(file_pos, step, mb, tp, N_mb, N_tp);
        }
    }
}


__host__ void integration_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, step_function_CPU step_update, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    auto start = chrono::high_resolution_clock::now();
    auto timer = chrono::high_resolution_clock::now();
    N_step = 0;
    int part_w = 0;
    int part_now;
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    while (chrono::duration_cast<chrono::milliseconds>(timer - start).count() < computation_time*60000) {
        N_step++;
        step_update(mb, tp, N_mb, N_tp, tau);

        timer = chrono::high_resolution_clock::now();
        if (def_freq_w_in_steps) {
            if (N_step%(int)freq_w == 0) {
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
            }
        }
        else {
            part_now = chrono::duration_cast<chrono::milliseconds>(timer - start).count()/(computation_time*60000*freq_w/100.);
            if (part_now > part_w) {
                part_w = part_now;
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
            }
        }
        timer = chrono::high_resolution_clock::now();
    }
}


__host__ void integration_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int nb_block, int nb_thread, step_function_GPU step_update, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    int freq_w_eff = (int)freq_w;
    if (!def_freq_w_in_steps) {
        freq_w_eff = (int)(N_step*freq_w/100);
    }

    int size_tp = N_tp*sizeof(test_particle);
    int size_mb = N_mb*sizeof(massive_body);

    test_particle *access_tp;
    massive_body *access_mb;

    cudaMalloc((void **) &access_mb, size_mb);
    cudaMalloc((void **) &access_tp, size_tp);

    cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    for (int step=1; step<=N_step; step++) {
        step_update(mb, tp, access_mb, access_tp, N_mb, N_tp, tau, nb_block, nb_thread);
        if (step%freq_w_eff == 0) {
            write_positions(file_pos, step, mb, tp, N_mb, N_tp);
        }
    }

    cudaFree(access_mb);
    cudaFree(access_tp);
}


__host__ void integration_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, int nb_block, int nb_thread, step_function_GPU step_update, ofstream& file_pos, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    auto start = chrono::high_resolution_clock::now();
    auto timer = chrono::high_resolution_clock::now();
    N_step = 0;
    int part_w = 0;
    int part_now;
    
    int size_tp = N_tp*sizeof(test_particle);
    int size_mb = N_mb*sizeof(massive_body);

    test_particle *access_tp;
    massive_body *access_mb;

    cudaMalloc((void **) &access_mb, size_mb);
    cudaMalloc((void **) &access_tp, size_tp);

    cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    while (chrono::duration_cast<chrono::milliseconds>(timer - start).count() < computation_time*60000) {
        N_step++;
        step_update(mb, tp, access_mb, access_tp, N_mb, N_tp, tau, nb_block, nb_thread);

        timer = chrono::high_resolution_clock::now();
        if (def_freq_w_in_steps) {
            if (N_step%(int)freq_w == 0) {
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
            }
        }
        else {
            part_now = chrono::duration_cast<chrono::milliseconds>(timer - start).count()/(computation_time*60000*freq_w/100);
            if (part_now > part_w) {
                part_w = part_now;
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
            }
        }
        timer = chrono::high_resolution_clock::now();
    }

    cudaFree(access_mb);
    cudaFree(access_tp);
}