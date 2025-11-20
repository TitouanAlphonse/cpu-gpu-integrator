#ifndef MASSIVE_BODY_H
#define MASSIVE_BODY_H

#include"global.h"
#include"Vec3d.h"

class Massive_body {
public:

    double m;   // mass (in solar masses)
    double R;   // radius (in solar radii)

    Vec3d q;    // position (in a.u.)
    Vec3d v;    // velocity (in a.u./years)

    Massive_body() : m(0), R(0), q(Vec3d()), v(Vec3d()) {}
    Massive_body(double m_, double R_, Vec3d q_init, Vec3d v_init); // m_ in Sun masses, R_ in sun radii, q_init norm in a.u., v_init norm in in a.u./years

    void print();

    Massive_body& operator=(const Massive_body&) = default;
};


class Massive_body_qv {
public:

    Vec3d q;    // position (in a.u.)
    Vec3d v;    // velocity (in a.u./years)

    Massive_body_qv() : q(Vec3d()), v(Vec3d()) {}
    Massive_body_qv(Vec3d q_init, Vec3d v_init) {q = q_init; v = v_init;}; // q_init norm in a.u., v_init norm in in a.u./years

    void print();

    Massive_body_qv& operator=(const Massive_body_qv&) = default;
};


class Massive_body_mR {
public:

    double m;   // mass (in solar masses)
    double R;   // radius (in solar radii)

    Massive_body_mR() : m(0), R(0) {}
    Massive_body_mR(double m_, double R_) {m = m_; R = R_;}; // q_init norm in a.u., v_init norm in in a.u./years

    void print();

    Massive_body_mR& operator=(const Massive_body_mR&) = default;
};

#endif
