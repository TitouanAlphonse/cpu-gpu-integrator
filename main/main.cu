#include"../headers/global.h"
#include"../headers/Vec3d.h"
#include"../headers/massive_body.h"
#include"../headers/init.h"
#include"../headers/leapfrog.h"


int main() {

    int N_mb, N_tp, N_step, nb_block, nb_thread, N_substep;
    double integration_duration, tau;
    string integration_mode, init_config_mb, init_config_tp;
    bool error_init = false;

    read_and_init("input.in", error_init, N_mb, N_tp, N_step, integration_duration, tau, integration_mode, nb_block, nb_thread, N_substep, init_config_mb, init_config_tp);

    // cout << "N_mb : " << N_mb << endl;
    // cout << "N_tp : " << N_tp << endl;
    // cout << "N_step : " << N_step << endl;
    // cout << "integration_duration : " << integration_duration << endl;
    // cout << "tau : " << tau/day_in_years << endl;
    // cout << "integration mode : " << integration_mode << endl;
    // cout << "nb_block : " << nb_block << endl;
    // cout << "nb_thread : " << nb_thread << endl;
    // cout << "N_substep : " << N_substep << endl;
    // cout << "init_config_mb : " << init_config_mb << endl;
    // cout << "init_config_tp : " << init_config_tp << endl;
    // cout << "error : " << error_init << endl << endl;

    massive_body *mb = (massive_body*)malloc(N_mb*sizeof(massive_body));
    test_particle *tp = (test_particle*)malloc(N_tp*sizeof(test_particle));

    init_mb(mb, N_mb, init_config_mb);
    init_tp(tp, N_tp, init_config_tp);


    if (!error_init) {
        if (integration_mode == "cpu") {
            if (N_step == -1) {
                cout << "Not implemented yet" << endl;
            }
            else {
                leapfrog_CPU(mb, tp, N_mb, N_tp, tau, N_step, "_CPU");
            }
        }
        if (integration_mode == "gpu") {
            if (N_step == -1) {
                leapfrog_GPU(mb, tp, N_mb, N_tp, tau, integration_duration, nb_block, nb_thread, "_GPU");
            }
            else {
                leapfrog_GPU(mb, tp, N_mb, N_tp, tau, N_step, nb_block, nb_thread, "_GPU");
            }
        }
        if (integration_mode == "gpu_multi-step") {
            if (N_step == -1) {
                cout << "Not implemented yet" << endl;
            }
            else {
                leapfrog_GPU_multi_t(mb, tp, N_mb, N_tp, tau, N_step, N_substep, nb_block, nb_thread, "_GPU_multi_t");
            }
        }
    }

    free(mb);
    free(tp);

    return 0;
}