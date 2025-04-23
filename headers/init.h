#include"global.h"
#include"Vec3d.h"
#include"massive_body.h"
#include"test_particle.h"

void read_and_init(string file_name, bool& error_init, int& N_mb, int& N_tp, int& N_step, double& computation_time, bool& def_N_step_in_cp_time, double& tau, string& integration_method, string& integration_mode, int& nb_block, int& nb_thread, int& N_substep, string& init_config_mb, string& init_config_tp, string& suffix, double& freq_w, bool& def_freq_w_in_steps);

void init_mb(massive_body* mb, int N_mb, string init_config_mb);
void init_tp(test_particle* tp, int N_tp, string init_config_tp);