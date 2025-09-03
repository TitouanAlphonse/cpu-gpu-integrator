#include"../headers/integration.h"


// #################
//  integration.cu
// #################

// See headers/integration.h for details about the aim of the functions


void print_info_start(string integration_method, string integration_mode, int N_mb, int N_tp, int N_step, double computation_time, bool def_N_step_in_cp_time, int N_substep, double tau, double freq_w, bool def_freq_w_in_steps) {
    cout << "-------------" << endl;
    if (integration_method == "leapfrog") {
        cout << "Leapfrog ";
    }
    if (integration_method == "MVS") {
        cout << "   MVS ";
    }
    if (integration_mode == "cpu") {cout << "CPU" << endl;}
    if (integration_mode == "gpu") {cout << "GPU" << endl;}
    if (integration_mode == "gpu_multi_step") {cout << "GPU multi-step" << endl;}
    cout << "-------------" << endl << endl;
    
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
    if (integration_mode == "gpu") {
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        int size_tp = N_tp*sizeof(test_particle);
        int size_mb = N_mb*sizeof(massive_body);
        cout << "Percentage of GPU memory used : " << (double) (size_tp+size_mb)/free_memory*100 << "% ("<< (double) (N_mb*sizeof(massive_body)+N_tp*sizeof(test_particle))/1e6 << " MB out of " << free_memory/1000000 << " MB) "<< endl;
    }
    if (integration_mode == "gpu_multi_step") {
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        int size_mb_qv_multi_step = (N_substep+1)*N_mb*sizeof(massive_body_qv);
        int size_mb_mR = N_mb*sizeof(massive_body_mR);
        int size_tp_multi_step = (N_substep+1)*N_tp*sizeof(test_particle);
        cout << "Percentage of GPU memory used : " << (double) (size_mb_qv_multi_step+size_mb_mR+size_tp_multi_step)/free_memory*100 << " % ("<< (double) (size_mb_qv_multi_step+size_mb_mR+size_tp_multi_step)/1e6 << " MB out of " << free_memory/1000000 << " MB) "<< endl;
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

__host__ void write_positions_multi_step(ofstream& file_pos, int step, int substep, massive_body_qv* mb_qv_multi_step, test_particle* tp_multi_step, int N_mb, int N_tp, int N_substep) {
    int i_ss;
    file_pos << step*N_substep + substep << " " ;
    for (int i=0; i<N_mb; i++) {
        i_ss = substep*N_mb + i;
        file_pos << mb_qv_multi_step[i_ss].q.get_x() << " " << mb_qv_multi_step[i_ss].q.get_y() << " " << mb_qv_multi_step[i_ss].q.get_z() << " ";
    }
    for (int i=0; i<N_tp; i++) {
        i_ss = substep*N_tp + i;
        file_pos << tp_multi_step[i_ss].q.get_x() << " " << tp_multi_step[i_ss].q.get_y() << " " << tp_multi_step[i_ss].q.get_z() << " ";
    }
    file_pos << endl;
}


__host__ void write_orb_param(ofstream& file_orb_param, int step, massive_body* mb, test_particle* tp, int N_mb, int N_tp, pos_vel_sub_func pos_vel_sub, double M_tot) {
    double a, e, inc, omega, Omega, M;
    Vec3d q_sub, v_sub;
    pos_vel_sub(q_sub, v_sub, mb, N_mb, M_tot);
    
    file_orb_param << step << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " ";
    
    for (int i=1; i<N_mb; i++) {
        pos_vel_to_orb_param(G*(mb[0].m+mb[i].m), mb[i].q - q_sub, mb[i].v - v_sub, a, e, inc, Omega, omega, M);
        file_orb_param << a << " " << e << " " << rad_to_deg(inc) << " " << rad_to_deg(Omega) << " " << rad_to_deg(omega) << " " << rad_to_deg(M) << " ";
    }
    for (int i=0; i<N_tp; i++) {
        pos_vel_to_orb_param(G*mb[0].m, tp[i].q - q_sub, tp[i].v - v_sub, a, e, inc, Omega, omega, M);
        file_orb_param << a << " " << e << " " << rad_to_deg(inc) << " " << rad_to_deg(Omega) << " " << rad_to_deg(omega) << " " << rad_to_deg(M) << " ";
    }
    file_orb_param << endl;
}

__host__ void write_orb_param_multi_step(ofstream& file_orb_param, int step, int substep, massive_body_qv* mb_qv_multi_step, massive_body_mR* mb_mR, test_particle* tp_multi_step, int N_mb, int N_tp, int N_substep, pos_vel_sub_func_multi_step pos_vel_sub, double M_tot) {
    int i_ss;    
    double a, e, inc, omega, Omega, M;
    Vec3d q_sub, v_sub;
    pos_vel_sub(q_sub, v_sub, mb_qv_multi_step, mb_mR, N_mb, M_tot, substep);
    
    file_orb_param << step*N_substep + substep << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " ";
    
    for (int i=1; i<N_mb; i++) {
        i_ss = substep*N_mb + i;
        pos_vel_to_orb_param(G*(mb_mR[0].m+mb_mR[i].m), mb_qv_multi_step[i_ss].q - q_sub, mb_qv_multi_step[i_ss].v - v_sub, a, e, inc, Omega, omega, M);
        file_orb_param << a << " " << e << " " << rad_to_deg(inc) << " " << rad_to_deg(Omega) << " " << rad_to_deg(omega) << " " << rad_to_deg(M) << " ";
    }
    for (int i=0; i<N_tp; i++) {
        i_ss = substep*N_tp + i;
        pos_vel_to_orb_param(G*mb_mR[0].m, tp_multi_step[i_ss].q - q_sub, tp_multi_step[i_ss].v - v_sub, a, e, inc, Omega, omega, M);
        file_orb_param << a << " " << e << " " << rad_to_deg(inc) << " " << rad_to_deg(Omega) << " " << rad_to_deg(omega) << " " << rad_to_deg(M) << " ";
    }
    file_orb_param << endl;
}


int integration_id(string integration_method, string integration_mode, bool def_N_step_in_cp_time) {
    int id;
    if (integration_mode == "cpu" && !def_N_step_in_cp_time) {id = 1;}
    if (integration_mode == "cpu" && def_N_step_in_cp_time) {id = 2;}
    if (integration_mode == "gpu" && !def_N_step_in_cp_time) {id = 3;}
    if (integration_mode == "gpu" && def_N_step_in_cp_time) {id = 4;}
    if (integration_mode == "gpu_multi_step" && !def_N_step_in_cp_time) {id = 5;}
    if (integration_mode == "gpu_multi_step" && def_N_step_in_cp_time) {id = 6;}

    if (integration_method == "leapfrog") {
        return id;
    }
    if (integration_method == "MVS") {
        return id+6;
    }

    return -1;
}

void launch_integration(string integration_method, string integration_mode, massive_body* mb,test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, bool def_N_step_in_cp_time, int N_substep, string user_forces, int nb_block, int nb_thread, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps) {
    switch (integration_id(integration_method, integration_mode, def_N_step_in_cp_time)) {
    case 1:
        integration_CPU(mb, tp, N_mb, N_tp, tau, N_step, leapfrog::step_leapfrog_CPU, user_forces, leapfrog::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 2:
        integration_CPU(mb, tp, N_mb, N_tp, tau, N_step, computation_time, leapfrog::step_leapfrog_CPU, user_forces, leapfrog::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 3:
        integration_GPU(mb, tp, N_mb, N_tp, tau, N_step, nb_block, nb_thread, leapfrog::step_leapfrog_GPU, user_forces, leapfrog::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 4:
        integration_GPU(mb, tp, N_mb, N_tp, tau, N_step, computation_time, nb_block, nb_thread, leapfrog::step_leapfrog_GPU, user_forces, leapfrog::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 5:
        integration_GPU_multi_step(mb, tp, N_mb, N_tp, tau, N_step, N_substep, nb_block, nb_thread, leapfrog::substep_leapfrog_mb_GPU_multi_step, leapfrog::step_leapfrog_tp_GPU_multi_step, user_forces, leapfrog::pos_vel_sub_multi_step, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 7:
        integration_CPU(mb, tp, N_mb, N_tp, tau, N_step, MVS::step_MVS_CPU, user_forces, MVS::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 8:
        integration_CPU(mb, tp, N_mb, N_tp, tau, N_step, computation_time, MVS::step_MVS_CPU, user_forces, leapfrog::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 9:
        integration_GPU(mb, tp, N_mb, N_tp, tau, N_step, nb_block, nb_thread, MVS::step_MVS_GPU, user_forces, MVS::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
        break;
    case 10:
        integration_GPU(mb, tp, N_mb, N_tp, tau, N_step, computation_time, nb_block, nb_thread, MVS::step_MVS_GPU, user_forces, MVS::pos_vel_sub, file_pos, file_orb_param, freq_w, def_freq_w_in_steps);
    default:
    case 11:
        cout << "MVS GPU multi-step not implemented yet" << endl;
        break;
    case 12:
        cout << "MVS GPU multi-step not implemented yet" << endl;
        break; 
    }
}

__host__ void integration_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, step_function_CPU step_update, string user_forces, pos_vel_sub_func pos_vel_sub, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    double M_tot = 0;
    for (int i=0; i<N_mb; i++) {
        M_tot += mb[i].m;
    }

    int freq_w_eff = (int)freq_w;
    if (!def_freq_w_in_steps) {
        freq_w_eff = max((int)(N_step*freq_w/100), 1);
    }

    massive_body *aux_mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
    test_particle *aux_tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

    int nb_dis, N_mb_dis;
    int* id_mb;
    double *dur_dis, *tau_dis, *dvel, *dedt;
    bool enable_user_forces;

    init_user_forces(user_forces, enable_user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    for (int step=1; step<=N_step; step++) {

        step_update(mb, tp, N_mb, N_tp, tau, aux_mb);

        if (enable_user_forces) {
            apply_user_forces(mb, N_mb, step*tau, tau, user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt, pos_vel_sub, M_tot);
        }

        if (step%freq_w_eff == 0) {
            write_positions(file_pos, step, mb, tp, N_mb, N_tp);
            write_orb_param(file_orb_param, step, mb, tp, N_mb, N_tp, pos_vel_sub, M_tot);
        }
    }


    free(aux_mb);
    free(aux_tp);
    if (enable_user_forces) {
        free(id_mb);
        free(dur_dis);
        free(tau_dis);
        free(dvel);
        free(dedt);
    }
}


__host__ void integration_CPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, step_function_CPU step_update, string user_forces, pos_vel_sub_func pos_vel_sub, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    double M_tot = 0;
    for (int i=0; i<N_mb; i++) {
        M_tot += mb[i].m;
    }

    auto start = chrono::high_resolution_clock::now();
    auto timer = chrono::high_resolution_clock::now();
    N_step = 0;
    int part_w = 0;
    int part_now;

    massive_body *aux_mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
    test_particle *aux_tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

    int nb_dis, N_mb_dis;
    int* id_mb;
    double *dur_dis, *tau_dis, *dvel, *dedt;
    bool enable_user_forces;

    init_user_forces(user_forces, enable_user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    while (chrono::duration_cast<chrono::milliseconds>(timer - start).count() < computation_time*60000) {
        N_step++;
        step_update(mb, tp, N_mb, N_tp, tau, aux_mb);
        if (enable_user_forces) {
            apply_user_forces(mb, N_mb, N_step*tau, tau, user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt, pos_vel_sub, M_tot);
        }

        timer = chrono::high_resolution_clock::now();
        if (def_freq_w_in_steps) {
            if (N_step%(int)freq_w == 0) {
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
                write_orb_param(file_orb_param, N_step, mb, tp, N_mb, N_tp, pos_vel_sub, M_tot);
            }
        }
        else {
            part_now = chrono::duration_cast<chrono::milliseconds>(timer - start).count()/(computation_time*60000*freq_w/100.);
            if (part_now > part_w) {
                part_w = part_now;
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
                write_orb_param(file_orb_param, N_step, mb, tp, N_mb, N_tp, pos_vel_sub, M_tot);
            }
        }
        timer = chrono::high_resolution_clock::now();
    }

    free(aux_mb);
    free(aux_tp);
    if (enable_user_forces) {
        free(id_mb);
        free(dur_dis);
        free(tau_dis);
        free(dvel);
        free(dedt);
    }
}


__host__ void integration_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int nb_block, int nb_thread, step_function_GPU step_update, string user_forces, pos_vel_sub_func pos_vel_sub, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    double M_tot = 0;
    for (int i=0; i<N_mb; i++) {
        M_tot += mb[i].m;
    }

    int freq_w_eff = (int)freq_w;
    if (!def_freq_w_in_steps) {
        freq_w_eff = max((int)(N_step*freq_w/100), 1);
        
    }

    int size_tp = N_tp*sizeof(test_particle);
    int size_mb = N_mb*sizeof(massive_body);

    test_particle *access_tp;
    massive_body *access_mb;

    cudaMalloc((void **) &access_mb, size_mb);
    cudaMalloc((void **) &access_tp, size_tp);

    cudaMemcpy(access_tp, tp, size_tp, cudaMemcpyHostToDevice);

    massive_body *aux_mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
    test_particle *aux_tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

    int nb_dis, N_mb_dis;
    int* id_mb;
    double *dur_dis, *tau_dis, *dvel, *dedt;
    bool enable_user_forces;

    init_user_forces(user_forces, enable_user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    for (int step=1; step<=N_step; step++) {

        step_update(mb, tp, access_mb, access_tp, N_mb, N_tp, tau, nb_block, nb_thread, aux_mb);
        if (enable_user_forces) {
            apply_user_forces(mb, N_mb, step*tau, tau, user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt, pos_vel_sub, M_tot);
        }

        if (step%freq_w_eff == 0) {
            write_positions(file_pos, step, mb, tp, N_mb, N_tp);
            write_orb_param(file_orb_param, step, mb, tp, N_mb, N_tp, pos_vel_sub, M_tot);
        }
    }

    free(aux_mb);
    free(aux_tp);
    cudaFree(access_mb);
    cudaFree(access_tp);
    if (enable_user_forces) {
        free(id_mb);
        free(dur_dis);
        free(tau_dis);
        free(dvel);
        free(dedt);
    }
}


__host__ void integration_GPU(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int& N_step, double computation_time, int nb_block, int nb_thread, step_function_GPU step_update, string user_forces, pos_vel_sub_func pos_vel_sub, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    double M_tot = 0;
    for (int i=0; i<N_mb; i++) {
        M_tot += mb[i].m;
    }

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

    massive_body *aux_mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
    test_particle *aux_tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

    int nb_dis, N_mb_dis;
    int* id_mb;
    double *dur_dis, *tau_dis, *dvel, *dedt;
    bool enable_user_forces;

    init_user_forces(user_forces, enable_user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    while (chrono::duration_cast<chrono::milliseconds>(timer - start).count() < computation_time*60000) {
        N_step++;
        step_update(mb, tp, access_mb, access_tp, N_mb, N_tp, tau, nb_block, nb_thread, aux_mb);
        if (enable_user_forces) {
            apply_user_forces(mb, N_mb, N_step*tau, tau, user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt, pos_vel_sub, M_tot);
        }

        timer = chrono::high_resolution_clock::now();
        if (def_freq_w_in_steps) {
            if (N_step%(int)freq_w == 0) {
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
                write_orb_param(file_orb_param, N_step, mb, tp, N_mb, N_tp, pos_vel_sub, M_tot);
            }
        }
        else {
            part_now = chrono::duration_cast<chrono::milliseconds>(timer - start).count()/(computation_time*60000*freq_w/100);
            if (part_now > part_w) {
                part_w = part_now;
                write_positions(file_pos, N_step, mb, tp, N_mb, N_tp);
                write_orb_param(file_orb_param, N_step, mb, tp, N_mb, N_tp, pos_vel_sub, M_tot);
            }
        }
        timer = chrono::high_resolution_clock::now();
    }

    free(aux_mb);
    free(aux_tp);
    cudaFree(access_mb);
    cudaFree(access_tp);
    if (enable_user_forces) {
        free(id_mb);
        free(dur_dis);
        free(tau_dis);
        free(dvel);
        free(dedt);
    }
}




// =======================================


__host__ void integration_GPU_multi_step(massive_body* mb, test_particle* tp, int N_mb, int N_tp, double tau, int N_step, int N_substep, int nb_block, int nb_thread, substep_function_mb_GPU_multi_step substep_mb, step_function_tp_GPU_multi_step step_tp, string user_forces, pos_vel_sub_func_multi_step pos_vel_sub, ofstream& file_pos, ofstream& file_orb_param, double freq_w, bool def_freq_w_in_steps) {

    // Initialization :

    double M_tot = 0;
    for (int i=0; i<N_mb; i++) {
        M_tot += mb[i].m;
    }

    int freq_w_eff = (int)freq_w;
    if (!def_freq_w_in_steps) {
        freq_w_eff = max((int)(N_step*freq_w/100), 1);
        
    }

    int size_mb_qv_multi_step = (N_substep+1)*N_mb*sizeof(massive_body_qv);
    int size_mb_mR = N_mb*sizeof(massive_body_mR);
    int size_tp_multi_step = (N_substep+1)*N_tp*sizeof(test_particle);

    massive_body_qv *mb_qv_multi_step = (massive_body_qv*)malloc(size_mb_qv_multi_step);
    massive_body_mR *mb_mR = (massive_body_mR*)malloc(size_mb_mR);
    test_particle *tp_multi_step = (test_particle*)malloc(size_tp_multi_step);

    for (int i=0; i<N_mb; i++) {
        mb_qv_multi_step[i].q = mb[i].q;
        mb_qv_multi_step[i].v = mb[i].v;
        mb_mR[i].m = mb[i].m;
        mb_mR[i].R = mb[i].R;
    }

    for (int i=0; i<N_tp; i++) {
        tp_multi_step[i].q = tp[i].q;
        tp_multi_step[i].v = tp[i].v;
    }

    for (int substep=1; substep<=N_substep; substep++) {
        for (int i=0; i<N_mb; i++) {
            mb_qv_multi_step[substep*N_mb+i] = massive_body_qv(Vec3d(), Vec3d());
        }
        for (int i=0; i<N_tp; i++) {
            tp_multi_step[substep*N_tp+i] = test_particle(Vec3d(), Vec3d());
        }
    }

    massive_body_qv *access_mb_qv_multi_step;
    massive_body_mR *access_mb_mR;
    test_particle *access_tp_multi_step;

    cudaMalloc((void **) &access_mb_qv_multi_step, size_mb_qv_multi_step);
    cudaMalloc((void **) &access_mb_mR, size_mb_mR);
    cudaMalloc((void **) &access_tp_multi_step, size_tp_multi_step);

    cudaMemcpy(access_mb_mR, mb_mR, size_mb_mR, cudaMemcpyHostToDevice);
    cudaMemcpy(access_tp_multi_step, tp_multi_step, size_tp_multi_step, cudaMemcpyHostToDevice);

    massive_body_qv *aux_mb_qv_multi_step = (massive_body_qv*)malloc(size_mb_qv_multi_step);

    int nb_dis, N_mb_dis;
    int* id_mb;
    double *dur_dis, *tau_dis, *dvel, *dedt;
    bool enable_user_forces;

    init_user_forces(user_forces, enable_user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt);
    
    cout << "Done" << endl;
    cout << "Integration... " << flush;

    // Integration :
    
    for (int step=0; step<N_step/N_substep; step++) {

        for (int substep=1; substep<=N_substep; substep++) {
            substep_mb(mb_qv_multi_step, mb_mR, N_mb, tau, substep, aux_mb_qv_multi_step);
            if (enable_user_forces) {
                apply_user_forces_multi_step(mb_qv_multi_step, mb_mR, N_mb, substep, (step*N_substep+substep)*tau, tau, user_forces, nb_dis, N_mb_dis, id_mb, dur_dis, tau_dis, dvel, dedt, pos_vel_sub, M_tot);
            }
            // Initialization for the next call :
            for (int i=0; i<N_mb; i++) {
                // aux_mb_qv_multi_step[i].q.print();
                mb_qv_multi_step[i] = mb_qv_multi_step[N_substep*N_mb+i];
            }
        }

        step_tp(mb_qv_multi_step, mb_mR, tp_multi_step, access_mb_qv_multi_step, access_mb_mR, access_tp_multi_step, N_mb, N_tp, N_substep, tau, nb_block, nb_thread, aux_mb_qv_multi_step);

        for (int substep=1; substep<=N_substep; substep++) {
            if ((step*N_substep + substep)%freq_w_eff == 0) {
                write_positions_multi_step(file_pos, step, substep, mb_qv_multi_step, tp_multi_step, N_mb, N_tp, N_substep);
                write_orb_param_multi_step(file_orb_param, step, substep, mb_qv_multi_step, mb_mR, tp_multi_step, N_mb, N_tp, N_substep, pos_vel_sub, M_tot);
            }
        }
        
    }

    free(mb_qv_multi_step);
    free(mb_mR);
    free(tp_multi_step);
    free(aux_mb_qv_multi_step);
    cudaFree(access_mb_qv_multi_step);
    cudaFree(access_mb_mR);
    cudaFree(access_tp_multi_step);
    if (enable_user_forces) {
        free(id_mb);
        free(dur_dis);
        free(tau_dis);
        free(dvel);
        free(dedt);
    }
}
