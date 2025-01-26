#ifndef MASSIVE_BODY_H
#define MASSIVE_BODY_H

#include"global.h"
#include"Vec3d.h"

class massive_body {
// private:
public:

    double m;   // mass
    double R;   // radius

    Vec3d q;    // position
    Vec3d p;    // momentum

// public:
    massive_body() : m(0), R(0), q(Vec3d()), p(Vec3d()) {}
    massive_body(double m, double R, Vec3d q_init, Vec3d v_init);

    void print();
};

#endif