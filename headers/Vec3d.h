#ifndef VEC3D_H
#define VEC3D_H

#include"global.h"

class Vec3d {
private:
    double x;   // in [-inf, +inf]
    double y;   // in [-inf, +inf]
    double z;   // in [-inf, +inf]

    double r;   // in [0, +inf]
    double theta;   // in [0, +pi]
    double phi;   // in [0, +2pi]

public:
    Vec3d() : x(0), y(0), z(0), r(0), theta(0), phi(0) {};
    Vec3d(double p1, double p2, double p3, string type_def); // type_def = "xyz" for cartesian coordinates, type_def = "spheric" for spherical coordinates

    void update_xyz();
    void update_spheric();

    void set_x(double x_);
    void set_y(double y_);
    void set_z(double z_);
    void set_xyz(double x_, double y_, double z_);
    void set_r(double r_);
    void set_theta(double theta_);
    void set_phi(double phi_);
    void set_spheric(double r_, double theta_, double phi_);

    double get_x();
    double get_y();
    double get_z();
    double norm();
    double get_theta();
    double get_phi();

    void print();

    Vec3d operator+(Vec3d right);
    Vec3d& operator+=(Vec3d right);

    Vec3d operator-(Vec3d right);
    Vec3d& operator-=(Vec3d right);

    Vec3d& operator-();

    Vec3d operator*(double scalar);
    Vec3d& operator*=(double scalar);

    Vec3d operator/(double scalar);
    Vec3d& operator/=(double scalar);

    double operator*(Vec3d right);    // Scalar product

    Vec3d (const Vec3d&) = default;
    Vec3d& operator=(const Vec3d&) = default;

};

Vec3d operator*(double scalar, Vec3d v);

double scalar_product(Vec3d v1, Vec3d v2);
Vec3d cross_product(Vec3d v1, Vec3d v2);

#endif