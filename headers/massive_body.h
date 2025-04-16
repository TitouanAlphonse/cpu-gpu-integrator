#ifndef MASSIVE_BODY_H
#define MASSIVE_BODY_H

#include"global.h"
#include"Vec3d.h"

class massive_body {
public:

    double m;   // mass (in solar masses)
    double R;   // radius (in solar radii)

    Vec3d q;    // position (in a.u.)
    Vec3d v;    // velocity (in a.u./years)

    massive_body() : m(0), R(0), q(Vec3d()), v(Vec3d()) {}
    massive_body(double m_, double R_, Vec3d q_init, Vec3d v_init); // m_ in Sun masses, R_ in sun radii, q_init norm in a.u., v_init norm in m/s

    void print();

    massive_body& operator=(const massive_body&) = default;
};

#endif