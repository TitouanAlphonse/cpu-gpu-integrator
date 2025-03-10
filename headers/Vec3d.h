#ifndef VEC3D_H
#define VEC3D_H

#include <cuda_runtime.h>
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
    Vec3d(double p1, double p2, double p3, string type_def) { // type_def = "xyz" for cartesian coordinates, type_def = "spheric" for spherical coordinates
        bool flag = false;

        if (type_def == "xyz") {
            set_xyz(p1, p2, p3);
            flag = true;
        }
        if (type_def == "cylindric") {
            set_spheric(p1, pi/2, p2);
            set_z(p3);
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

    void update_xyz() {
        x = r*sin(theta)*cos(phi);
        y = r*sin(theta)*sin(phi);
        z = r*cos(theta);
    };
    __host__ __device__ void update_spheric() {
        r = sqrt(x*x + y*y + z*z);
        if (r != 0){
            theta = acos(z/r);
        }
        else {
            theta = 0;
        }
        
        phi = atan2(y,x);
        if (phi < 0) {
            phi += 2*pi;
        }
    };

    __host__ __device__ void set_x(double x_) {
        x = x_;
        update_spheric();
    };
    __host__ __device__ void set_y(double y_) {
        y = y_;
        update_spheric();
    };
    __host__ __device__ void set_z(double z_) {
        z = z_;
        update_spheric();
    };
    void set_xyz(double x_, double y_, double z_) {
        x = x_;
        y = y_;
        z = z_;
        update_spheric();
    };
    void set_r(double r_) {
        if (r_ >= 0) {
            r = r_;
        }
        else {
            r = -r_;
            theta = pi - theta;
            phi = fmod(phi + pi, 2*pi);
        }
        update_xyz();
    };
    void set_theta(double theta_) {
        theta = fmod(theta_, 2*pi);
        if (theta < 0) {
            theta += 2*pi;
        }
        if (theta >= pi) {
            theta = 2*pi - theta; 
        }
        update_xyz();
    };
    void set_phi(double phi_) {
        phi = fmod(phi_, 2*pi);
        if (phi < 0) {
            phi += 2*pi;
        }
        update_xyz();
    };
    void set_spheric(double r_, double theta_, double phi_) {
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
        
        update_xyz();
    };

    __host__ __device__ double get_x() {return x;};
    __host__ __device__ double get_y() {return y;};
    __host__ __device__ double get_z() {return z;};
    double norm() {return r;};
    double get_theta() {return theta;};
    double get_phi() {return phi;};

    void print() {
        cout << "Cartesian coordinates : (" << x << ";" << y << ";" << z << ")" << endl;
        cout << "Spherical coordinates : (" << r << ";" << theta << ";" << phi << ")" << endl;
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
        set_spheric(r, pi - theta, phi + pi);
        return *this;
    };

    Vec3d operator*(double scalar) {return Vec3d(x*scalar, y*scalar, z*scalar, "xyz");};
    Vec3d& operator*=(double scalar) {
        set_r(r*scalar);
        return *this;
    };

    Vec3d operator/(double scalar) {return Vec3d(r/scalar, theta, phi, "spheric");};
    Vec3d& operator/=(double scalar) {
        set_r(r/scalar);
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