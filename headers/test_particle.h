#ifndef TEST_PARTICLE_H
#define TEST_PARTICLE_H

#include"global.h"
#include"Vec3d.h"

class test_particle {
// private:
public:

    Vec3d q;    // position
    Vec3d v;    // momentum

// public:
    test_particle() : q(Vec3d()), v(Vec3d()) {}
    test_particle(Vec3d q_init, Vec3d v_init);

    void print();
};

#endif