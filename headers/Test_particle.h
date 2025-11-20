#ifndef TEST_PARTICLE_H
#define TEST_PARTICLE_H

#include"global.h"
#include"Vec3d.h"

class Test_particle {
public:

    Vec3d q;    // position (in a.u.)
    Vec3d v;    // velocity (in a.u./years)

    Test_particle() : q(Vec3d()), v(Vec3d()) {}
    Test_particle(Vec3d q_init, Vec3d v_init); // q_init norm in a.u., v_init norm in in a.u./years

    void print();

    Test_particle& operator=(const Test_particle&) = default;
};

#endif
