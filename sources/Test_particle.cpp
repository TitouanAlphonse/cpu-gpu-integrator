#include"../headers/Test_particle.h"

// #################
// Test_particle.cpp
// #################

// See headers/Test_particle_body.h for details about the Test_particle class


Test_particle::Test_particle(Vec3d q_init, Vec3d v_init) {
    q = q_init;
    v = v_init;
}


void Test_particle::print() {
    cout << "Position (in a.u.):" << endl;
    q.print();
}
