#include"../headers/init.h"


int str_id(string str_desc) {
    str_desc.erase(0, str_desc.find_first_not_of(" "));
    str_desc.erase(str_desc.find_last_not_of(" ") + 1);

    if (str_desc == "Number of massive bodies" || str_desc == "number of massive bodies") return 1;
    if (str_desc == "Number of test-particles" || str_desc == "number of test-particles") return 2;
    if (str_desc == "Integration over" || str_desc == "integration over") return 3;
    if (str_desc == "Time-step" || str_desc == "time-step") return 4;
    if (str_desc == "Integration method" || str_desc == "integration method") return 5;
    if (str_desc == "Integration mode" || str_desc == "integration mode") return 6;
    if (str_desc == "Number of blocks used" || str_desc == "number of blocks used") return 7;
    if (str_desc == "Number of threads per block" || str_desc == "number of threads per block") return 8;
    if (str_desc == "Number of sub-steps" || str_desc == "number of sub-steps") return 9;
    if (str_desc == "Initial massive bodies configuration" || str_desc == "initial massive bodies configuration") return 10;
    if (str_desc == "Initial test-particles configuration" || str_desc == "initial test-particles configuration") return 11;
    if (str_desc == "File name suffix" || str_desc == "file name suffix") return 12;
    if (str_desc == "Positions writing frequency" || str_desc == "positions writing frequency") return 13;
    
    return -1;
}

void init_param(ifstream& fich_input, int id_desc, bool& error_init, int& N_mb, bool& N_mb_init, int& N_tp, bool& N_tp_init, int& N_step, double& computation_time, bool& def_N_step_in_cp_time, bool& N_step_init, bool& def_N_step_in_years, double& tau, bool& tau_init, string& integration_method, bool& integration_method_init, string& integration_mode, bool& integration_mode_init, int& nb_block, bool& nb_block_init, int& nb_thread, bool& nb_thread_init, int& N_substep, bool& N_substep_init, string& init_config_mb, bool& init_config_mb_init, string& init_config_tp, bool& init_config_tp_init, string& suffix, bool& def_suffix_mode, double& freq_w, bool& def_freq_w_in_steps) {
    string str_p;
    int int_p;
    double double_p;
    switch (id_desc) {
    case 1:
        fich_input >> N_mb;
        if (N_mb <= 0) {
            cout << "Initialization error : invalid number of massive bodies." << endl;
            error_init = true;
        }
        else {
            N_mb_init = true;
        }
        fich_input.clear();
        break;

    case 2:
        fich_input >> N_tp;
        if (N_tp < 0) {
            cout << "Initialization error : invalid number of test-particles." << endl;
            error_init = true;
        }
        else {
            N_tp_init = true;
        }
        fich_input.clear();
        break;

    case 3:
        N_step = -1;
        computation_time = -1;
        fich_input >> double_p >> str_p;
        if (str_p == "steps" || str_p == "step") {
            N_step = (int) double_p;
            def_N_step_in_years = false;
            def_N_step_in_cp_time = false;
        }
        if (str_p == "years" || str_p == "year") {
            computation_time = double_p;
            if (tau_init) {
                N_step = (int) (computation_time/tau);
                computation_time = -1;
            }
            def_N_step_in_years = true;
            def_N_step_in_cp_time = false;
        }
        if (str_p == "minutes" || str_p == "minute") {
            computation_time = double_p;
            def_N_step_in_years = false;
            def_N_step_in_cp_time = true;
        }

        if (double_p <= 0 || (str_p != "steps" && str_p != "years" && str_p != "minutes" && str_p != "step" && str_p != "year" && str_p != "minute")) {
            cout << "Initialization error : invalid value for integration time." << endl;
            error_init = true;
        }
        else {
            N_step_init = true;
        }
        fich_input.clear();
        break;

    case 4:
        fich_input >> double_p;
        tau = double_p*day_in_years;
        if (tau <= 0) {
            cout << "Initialization error : invalid value of time-step." << endl;
            error_init = true;
        }
        else {
            tau_init = true;
            if (N_step_init && def_N_step_in_years) {
                N_step = (int) (computation_time/tau);
                computation_time = -1;
            }
        }
        fich_input.clear();
        break;

    case 5:
        fich_input >> integration_method;
        if (integration_method != "leapfrog") {
            cout << "Initialization error : invalid method of integration." << endl;
            error_init = true;
        }
        else {
            integration_method_init = true;
        }
        fich_input.clear();
        break;
    
    case 6:
        fich_input >> integration_mode;
        if (integration_mode != "cpu" && integration_mode != "gpu" && integration_mode != "gpu_multi-step") {
            cout << "Initialization error : invalid mode of integration." << endl;
            error_init = true;
        }
        else {
            integration_mode_init = true;
        }
        fich_input.clear();
        break;
    
    case 7:
        fich_input >> nb_block;
        if (nb_block > 0) {
            nb_block_init = true;
        }
        fich_input.clear();
        break;

    case 8:
        fich_input >> nb_thread;
        if (nb_thread > 0) {
            nb_thread_init = true;
        }
        fich_input.clear();
        break;

    case 9:
        fich_input >> N_substep;
        if (N_substep > 0) {
            N_substep_init = true;
        }
        fich_input.clear();
        break;

    case 10:
        fich_input >> init_config_mb;
        if (init_config_mb != "sun+gas_planets") {
            cout << "Initialization error : invalid initial massive bodies configuration." << endl;
            error_init = true;
        }
        else {
            init_config_mb_init = true;
        }
        fich_input.clear();
        break;

    case 11:
        fich_input >> init_config_tp;
        if (init_config_tp != "random") {
            cout << "Initialization error : invalid initial test-particles configuration." << endl;
            error_init = true;
        }
        else {
            init_config_tp_init = true;
        }
        fich_input.clear();
        break;
    
    case 12:
        fich_input >> str_p;
        if (str_p == "none") {
            suffix = "";
        }
        else {
            if (str_p == "mode") {
                def_suffix_mode = true;
            }
            else {
                suffix = "_" + str_p;
            }
        }
        fich_input.clear();
        break;
    
    case 13:
        fich_input >> double_p >> str_p;
        if (double_p > 0) {
            if (str_p == "steps" || str_p == "step") {
                freq_w = double_p;
                def_freq_w_in_steps = true;
            }
            else {
                if (str_p == "%") {
                    freq_w = double_p;
                    def_freq_w_in_steps = false;
                }
                else {
                    cout << "Warning : invalid writing frequency input. Default value will be used." << endl;
                }
            }
        }
        else {
            cout << "Warning : invalid writing frequency value. Default value will be used." << endl;
        }
        fich_input.clear();
        break;

    default:
        break;
    }
}

void read_and_init(string file_name, bool& error_init, int& N_mb, int& N_tp, int& N_step, double& computation_time, bool& def_N_step_in_cp_time, double& tau, string& integration_method, string& integration_mode, int& nb_block, int& nb_thread, int& N_substep, string& init_config_mb, string& init_config_tp, string& suffix, double& freq_w, bool& def_freq_w_in_steps) {

    // Default values :
    suffix = "";
    freq_w = 1;
    def_freq_w_in_steps = true;

    // Reading of the input file :
    ifstream fich_input;
    fich_input.open(file_name, ios::in);

    if (!fich_input.is_open()) {
        cout << "Error : unable to open the input file" << endl;
        error_init = true;
    }
    else {
        int id;
        char c;
        string str_desc;

        bool N_mb_init = false, N_tp_init = false, N_step_init = false, tau_init = false, integration_method_init = false, integration_mode_init = false, nb_block_init = false, nb_thread_init = false, N_substep_init = false, init_config_mb_init = false, init_config_tp_init = false, def_N_step_in_years = false, def_suffix_mode = false;

        while (fich_input.get(c)) {
            if (c == ':') {
                id = str_id(str_desc);
                // cout << str_desc << endl;
                init_param(fich_input, id, error_init, N_mb, N_mb_init, N_tp, N_tp_init, N_step, computation_time, def_N_step_in_cp_time, N_step_init, def_N_step_in_years, tau, tau_init, integration_method, integration_method_init, integration_mode, integration_mode_init, nb_block, nb_block_init, nb_thread, nb_thread_init, N_substep, N_substep_init, init_config_mb, init_config_mb_init, init_config_tp, init_config_tp_init, suffix, def_suffix_mode, freq_w, def_freq_w_in_steps);
                str_desc = "";
            }
            else {
                if (c == '\n') {
                    str_desc = "";
                }
                else {
                    str_desc += c;
                }
            }
        }

        if (integration_mode_init && def_suffix_mode) {
            suffix = "_" + integration_mode;
        }

        if (init_config_mb == "sun+gas_planets" && N_mb != 5) {
            cout << "Initialization error : incompatible number of massive bodies with their given initial configuration." << endl;
            error_init = true;
        }

        if ((integration_mode == "gpu" || integration_mode == "gpu_multi-step") && (!nb_block_init || !nb_thread_init)) {
            cout << "Initialization error : invalid or missing GPU inputs." << endl;
            cout << "Please ensure that the following parameters are specified and valid : number of blocks used, number of threads per block." << endl;
            error_init = true;
        }

        if (integration_mode == "gpu_multi-step" && !N_substep_init) {
            cout << "Initialization error : invalid or missing inputs for multi-step integration." << endl;
            cout << "Please ensure that the following parameters are specified and valid : number of sub-steps." << endl;
            error_init = true;
        }

        if (!N_mb_init || !N_tp_init || !N_step_init || !tau_init || !integration_method_init || !integration_mode_init || !init_config_mb_init || !init_config_tp_init) {
            cout << "Initialization error : invalid or missing variables." << endl;
            cout << "Please ensure that the following parameters are specified and valid : number of massive bodies, number of test-particles, number of steps or computation time (Integration over :), time-step, integration method, integration mode, initial test-particles configuration." << endl;
            error_init = true;
        }

        if (!error_init) {
            if ((integration_mode == "gpu" || integration_mode == "gpu_multi-step") && N_tp > nb_block*nb_thread) {
                cout << "Warning : number of test-particles greater than the number of threads used." << endl;
            }
        }
        cout << endl;
    }
}


void init_mb(massive_body* mb, int N_mb, string init_config_mb) {
    if (init_config_mb == "sun+gas_planets" && N_mb == 5) {
        double angle = 0;

        mb[0] = massive_body(1, 1, Vec3d(), Vec3d());
        mb[1] = massive_body(in_SM(M_Jupiter), in_SR(R_Jupiter), Vec3d(dist_Jupiter/au, pi/2, angle, "spheric"), Vec3d(v_Jupiter, pi/2, angle + pi/2, "spheric"));
        mb[2] = massive_body(in_SM(M_Saturn), in_SR(R_Saturn), Vec3d(dist_Saturn/au, pi/2, angle, "spheric"), Vec3d(v_Saturn, pi/2, angle + pi/2, "spheric"));
        mb[3] = massive_body(in_SM(M_Uranus), in_SR(R_Uranus), Vec3d(dist_Uranus/au, pi/2, angle, "spheric"), Vec3d(v_Uranus, pi/2, angle + pi/2, "spheric"));
        mb[4] = massive_body(in_SM(M_Neptune), in_SR(R_Neptune), Vec3d(dist_Neptune/au, pi/2, angle, "spheric"), Vec3d(v_Neptune, pi/2, angle + pi/2, "spheric"));

        double M = 1 + in_SM(M_Jupiter) + in_SM(M_Saturn) + in_SM(M_Uranus) + in_SM(M_Neptune);
        Vec3d q_cm = Vec3d();
        Vec3d v_cm = Vec3d();

        for (int i=0; i<5; i++) {
            q_cm += mb[i].m*mb[i].q/M;
            v_cm += mb[i].m*mb[i].v/M;
        }

        for (int i=0; i<5; i++) {
            mb[i].q -= q_cm;
            mb[i].v -= v_cm;
        }
    }
}

void init_tp(test_particle* tp, int N_tp, string init_config_tp) {
    if (init_config_tp == "random") {
        random_device rd;
        uniform_real_distribution<> uniform(0, 1);
        Vec3d q;
        Vec3d v;
        test_particle particle;

        double rmin = 5;
        double rmax = 5.4;
        double dev_theta_q = pi/100;
        double dev_vmin = 0.9;
        double dev_vmax = 1.1;
        double dev_theta_v = pi/100;


        for (int i=0; i<N_tp; i++) {
            q = Vec3d((rmin + (rmax-rmin)*uniform(rd)), pi/2 + (1-2*uniform(rd))*dev_theta_q, uniform(rd)*2*pi, "spheric");
            v = -cross_product(q, Vec3d(0,0,1,"xyz"))*(dev_vmin + (dev_vmax-dev_vmin)*uniform(rd))*sqrt(G*pow(au/year,2)/q.norm())/q.norm();
            v.set_theta(v.get_theta() + (1-2*uniform(rd))*dev_theta_v);

            particle = test_particle(q, v);
            tp[i] = particle;
        }
    }
}