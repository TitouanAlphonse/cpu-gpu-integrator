#ifndef TEST_PARTICLE_H
#define TEST_PARTICLE_H

#include"global.h"
#include"Vec3d.h"

class test_particle {
public:

    Vec3d q;    // position (in a.u.)
    Vec3d v;    // velocity (in a.u./years)

    test_particle() : q(Vec3d()), v(Vec3d()) {}
    test_particle(Vec3d q_init, Vec3d v_init); // q_init norm in a.u., v_init norm in m/s

    void print();
};

#endif