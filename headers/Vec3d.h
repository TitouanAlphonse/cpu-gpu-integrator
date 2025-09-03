#ifndef VEC3D_H
#define VEC3D_H

#include <cuda_runtime.h>
#include"global.h"

class Vec3d {
private:
    double x;
    double y;
    double z;

    double r2;
    

public:
    Vec3d() : x(0), y(0), z(0), r2(0) {};
    Vec3d(double p1, double p2, double p3, string type_def) { // type_def = "xyz" for cartesian coordinates, type_def = "spheric" for spherical coordinates
        bool flag = false;

        if (type_def == "xyz") {
            set_xyz(p1, p2, p3);
            flag = true;
        }
        if (type_def == "spheric") {
            set_spheric(p1, p2, p3);
            flag = true;
        }

        if (!flag) {
            // cout << "Not recognized type of definition" << endl;
            printf("Not recognized type of definition \n");
        }
    };

    void update_xyz(double r, double theta, double phi) {
        x = r*sin(theta)*cos(phi);
        y = r*sin(theta)*sin(phi);
        z = r*cos(theta);
    };

    __host__ __device__ void update_r2() {
        r2 = x*x + y*y + z*z;
    };

    __host__ __device__ void set_x(double x_) {
        x = x_;
        update_r2();
    };
    __host__ __device__ void set_y(double y_) {
        y = y_;
        update_r2();
    };
    __host__ __device__ void set_z(double z_) {
        z = z_;
        update_r2();
    };
    __host__ __device__ void set_xyz(double x_, double y_, double z_) {
        x = x_;
        y = y_;
        z = z_;
        update_r2();
    };

    void set_norm(double r_) {
        x /= sqrt(r2);
        y /= sqrt(r2);
        z /= sqrt(r2);

        if (r_ >= 0) {
            x *= r_;
            y *= r_;
            z *= r_;
            r2 = r_*r_;
        }
        else {
            x *= -r_;
            y *= -r_;
            z *= -r_;
            r2 = r_*r_;
        }
    };

    void set_spheric(double r_, double theta_, double phi_) {
        double r, theta, phi;
        theta = fmod(theta_, 2*pi);
        if (theta < 0) {
            theta += 2*pi;
        }
        if (theta >= pi) {
            theta = 2*pi - theta; 
        }

        phi = fmod(phi_, 2*pi);
        if (phi < 0) {
            phi += 2*pi;
        }

        if (r_ >= 0) {
            r = r_;
        }
        else {
            r = -r_;
            theta = pi - theta;
            phi = fmod(phi + pi, 2*pi);
        }
        
        update_xyz(r, theta, phi);
        r2 = r*r;
    };

    __host__ __device__ double get_x() {return x;};
    __host__ __device__ double get_y() {return y;};
    __host__ __device__ double get_z() {return z;};
    __host__ __device__ double norm() {return sqrt(r2);};
    __host__ __device__ double norm2() {return r2;};

    void print() {
        cout << "Cartesian coordinates : (" << x << ";" << y << ";" << z << ")" << endl;
        cout << "Norm : " << sqrt(r2) << endl;
    };

    Vec3d operator+(Vec3d right) {return Vec3d(x + right.x, y + right.y, z + right.z, "xyz");};
    Vec3d& operator+=(Vec3d right) {
        set_xyz(x + right.x, y + right.y, z + right.z);
        return *this;
    };

    Vec3d operator-(Vec3d right) {return Vec3d(x - right.x, y - right.y, z - right.z, "xyz");};
    Vec3d& operator-=(Vec3d right) {
        set_xyz(x - right.x, y - right.y, z - right.z);
        return *this;
    };

    Vec3d& operator-() {
        set_xyz(-x, -y, -z);
        return *this;
    };

    Vec3d operator*(double scalar) {return Vec3d(x*scalar, y*scalar, z*scalar, "xyz");};
    Vec3d& operator*=(double scalar) {
        set_xyz(x*scalar, y*scalar, z*scalar);
        return *this;
    };

    Vec3d operator/(double scalar) {return Vec3d(x/scalar, y/scalar, z/scalar, "xyz");};
    Vec3d& operator/=(double scalar) {
        set_xyz(x/scalar, y/scalar, z/scalar);
        return *this;
    };

    double operator*(Vec3d right) {return x*right.x + y*right.y + z*right.z;}; // Scalar product

    Vec3d (const Vec3d&) = default;
    Vec3d& operator=(const Vec3d&) = default;

};


Vec3d operator*(double scalar, Vec3d v);

double scalar_product(Vec3d v1, Vec3d v2);
Vec3d cross_product(Vec3d v1, Vec3d v2);

#endif