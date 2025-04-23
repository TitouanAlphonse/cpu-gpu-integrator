#include"../headers/global.h"
#include"../headers/Vec3d.h"
#include"../headers/massive_body.h"
#include"../headers/init.h"
#include"../headers/leapfrog.h"
#include"../headers/integration.h"


int main (int argc, char *argv[]) {

    int N_mb, N_tp, N_step, nb_block, nb_thread, N_substep;
    double computation_time, tau, freq_w;
    string input_file_path, integration_method, integration_mode, init_config_mb, init_config_tp, suffix;
    bool error_init = false, def_freq_w_in_steps, def_N_step_in_cp_time;

    if (argc > 1) {
        input_file_path = argv[1];
    }
    else {
        cout << "Please specify the input file path : ";
        cin >> input_file_path;
    }

    read_and_init(input_file_path, error_init, N_mb, N_tp, N_step, computation_time, def_N_step_in_cp_time, tau, integration_method, integration_mode, nb_block, nb_thread, N_substep, init_config_mb, init_config_tp, suffix, freq_w, def_freq_w_in_steps);

    if (!error_init) {

        print_info_start(integration_method, integration_mode, N_mb, N_tp, N_step, computation_time, def_N_step_in_cp_time, tau, freq_w, def_freq_w_in_steps);

        auto start = chrono::high_resolution_clock::now();

        massive_body *mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
        test_particle *tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

        init_mb(mb, N_mb, init_config_mb);
        init_tp(tp, N_tp, init_config_tp);

        ofstream file_general;
        ofstream file_pos;

        file_general.open("outputs/general_data"+suffix+".txt", ios::out);
        file_pos.open("outputs/positions"+suffix+".txt", ios::out);

        write_init(file_general, file_pos, tau, freq_w, mb, tp, N_mb, N_tp);
        file_general.close();
        
        launch_integration(integration_method, integration_mode, mb, tp, N_mb, N_tp, tau, N_step, computation_time, def_N_step_in_cp_time, nb_block, nb_thread, file_pos, freq_w, def_freq_w_in_steps);

        file_pos.close();

        free(mb);
        free(tp);

        auto end = chrono::high_resolution_clock::now();

        print_info_end(N_step, def_N_step_in_cp_time, tau, suffix);

        get_time(start, end);
        cout << endl;
    }

    return 0;
}