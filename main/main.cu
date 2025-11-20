#include"../headers/global.h"
#include"../headers/Vec3d.h"
#include"../headers/Massive_body.h"
#include"../headers/init.h"
#include"../headers/tools.h"
#include"../headers/leapfrog.h"
#include"../headers/MVS.h"
#include"../headers/integration.h"


// #################
//     main.cu
// #################

// This file uses all the functions contained in the files in headers and sources to launch the computation


int main (int argc, char *argv[]) {

    int N_mb, N_tp, N_step, nb_block, nb_thread, N_substep;
    double computation_time, tau, freq_w;
    string input_file_path, integration_method, integration_mode, init_config_mb, init_config_tp, suffix, user_forces_type;
    bool error_init = false, def_freq_w_in_steps, def_N_step_in_cp_time;

    if (argc > 1) {
        input_file_path = argv[1];
    }
    else {
        cout << "Please specify the input file path : ";
        cin >> input_file_path;
    }

    read_and_init(input_file_path, error_init, N_mb, N_tp, N_step, computation_time, def_N_step_in_cp_time,
                  tau, integration_method, integration_mode, nb_block, nb_thread, N_substep, init_config_mb,
                  init_config_tp, suffix, freq_w, def_freq_w_in_steps, user_forces_type);

    if (!error_init) {

        print_info_start(integration_method, integration_mode, N_mb, N_tp, N_step, computation_time,
                         def_N_step_in_cp_time, N_substep, tau, freq_w, def_freq_w_in_steps);

        auto start = chrono::high_resolution_clock::now();

        Massive_body *mb = (Massive_body*)malloc(N_mb*sizeof(Massive_body));
        Test_particle *tp = (Test_particle*)malloc(N_tp*sizeof(Test_particle));        

        init_mb(mb, N_mb, init_config_mb);
        init_tp(tp, N_tp, init_config_tp);

        ofstream file_general;
        ofstream file_pos;
        ofstream file_orb_param;

        file_general.open("outputs/general_data"+suffix+".txt", ios::out);
        file_pos.open("outputs/positions"+suffix+".txt", ios::out);
        file_orb_param.open("outputs/orb_param"+suffix+".txt", ios::out);

        write_init(file_general, file_pos, tau, freq_w, mb, tp, N_mb, N_tp);
        file_general.close();

        launch_integration(integration_method, integration_mode, mb, tp, N_mb, N_tp,
                           tau, N_step, computation_time, def_N_step_in_cp_time, N_substep,
                           user_forces_type, nb_block, nb_thread, file_pos, file_orb_param,
                           freq_w, def_freq_w_in_steps);

        file_pos.close();
        file_orb_param.close();

        auto end = chrono::high_resolution_clock::now();

        print_info_end(N_step, def_N_step_in_cp_time, tau, suffix);

        get_time(start, end);
        cout << endl;

        free(mb);
        free(tp);
    }

    return 0;
}