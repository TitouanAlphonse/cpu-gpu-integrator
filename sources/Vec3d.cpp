#include"../headers/Vec3d.h"

void Vec3d::update_xyz() {
    x = r*sin(theta)*cos(phi);
    y = r*sin(theta)*sin(phi);
    z = r*cos(theta);
}

void Vec3d::update_spheric() {
    r = sqrt(x*x + y*y + z*z);
    if (r != 0){
        theta = acos(z/r);
    }
    else {
        theta = 0;
    }
    
    phi = atan2(y,x);
    if (phi < 0) {
        phi += 2*M_PI;
    }
}

// ----------------

void Vec3d::set_x(double x_) {
    x = x_;
    update_spheric();
}

void Vec3d::set_y(double y_) {
    y = y_;
    update_spheric();
}

void Vec3d::set_z(double z_) {
    z = z_;
    update_spheric();
}

void Vec3d::set_xyz(double x_, double y_, double z_) {
    x = x_;
    y = y_;
    z = z_;
    update_spheric();
}

void Vec3d::set_r(double r_) {
    if (r_ >= 0) {
        r = r_;
    }
    else {
        r = -r_;
        theta = M_PI - theta;
        phi = fmod(phi + M_PI, 2*M_PI);
    }
    update_xyz();
}

void Vec3d::set_theta(double theta_) {
    theta = fmod(theta_, 2*M_PI);
    if (theta < 0) {
        theta += 2*M_PI;
    }
    if (theta >= M_PI) {
        theta = 2*M_PI - theta; 
    }
    update_xyz();
}

void Vec3d::set_phi(double phi_) {
    phi = fmod(phi_, 2*M_PI);
    if (phi < 0) {
        phi += 2*M_PI;
    }
    update_xyz();
}

void Vec3d::set_spheric(double r_, double theta_, double phi_) {
    theta = fmod(theta_, 2*M_PI);
    if (theta < 0) {
        theta += 2*M_PI;
    }
    if (theta >= M_PI) {
        theta = 2*M_PI - theta; 
    }

    phi = fmod(phi_, 2*M_PI);
    if (phi < 0) {
        phi += 2*M_PI;
    }

    if (r_ >= 0) {
        r = r_;
    }
    else {
        r = -r_;
        theta = M_PI - theta;
        phi = fmod(phi + M_PI, 2*M_PI);
    }
    
    update_xyz();
}

// ----------------

Vec3d::Vec3d(double p1, double p2, double p3, string type_def) {
    if (type_def == "xyz") {
        set_xyz(p1, p2, p3);
    }
    else {
        if (type_def == "spheric") {
            set_spheric(p1, p2, p3);
        }
        else {
            cout << "Not recognized type of définition" << endl;
        }
    }
}

// ----------------

double Vec3d::get_x() {
    return x;
}

double Vec3d::get_y() {
    return y;
}

double Vec3d::get_z() {
    return z;
}

double Vec3d::norm() {
    return r;
}

double Vec3d::get_theta() {
    return theta;
}

double Vec3d::get_phi() {
    return phi;
}

// ----------------

void Vec3d::print() {
    // cout << "Cartesian coordinates : (" << x << ";" << y << ";" << z << ")" << endl;
    cout << "Spherical coordinates : (" << r << ";" << theta << ";" << phi << ")" << endl;
}

// ----------------

Vec3d Vec3d::operator+(Vec3d right) {
    return Vec3d(x + right.x, y + right.y, z + right.z, "xyz");
}

Vec3d& Vec3d::operator+=(Vec3d right) {
    set_xyz(x + right.x, y + right.y, z + right.z);
    return *this;
}

Vec3d Vec3d::operator-(Vec3d right) {
    return Vec3d(x - right.x, y - right.y, z - right.z, "xyz");
}

Vec3d& Vec3d::operator-=(Vec3d right) {
    set_xyz(x - right.x, y - right.y, z - right.z);
    return *this;
}

Vec3d& Vec3d::operator-() {
    set_spheric(r, M_PI - theta, phi + M_PI);
    return *this;
}

Vec3d Vec3d::operator*(double scalar) {
    return Vec3d(x*scalar, y*scalar, z*scalar, "xyz");
}

Vec3d& Vec3d::operator*=(double scalar) {
    set_r(r*scalar);
    return *this;
}

Vec3d Vec3d::operator/(double scalar) {
    return Vec3d(r/scalar, theta, phi, "spheric");
}

Vec3d& Vec3d::operator/=(double scalar) {
    set_r(r/scalar);
    return *this;
}

Vec3d operator*(double scalar, Vec3d v) {
    return v*scalar;
}

// ----------------

double scalar_product(Vec3d v1, Vec3d v2) {
    return v1.get_x()*v2.get_x() + v1.get_y()*v2.get_y() + v1.get_z()*v2.get_z();
}

double Vec3d::operator*(Vec3d right) {
    return scalar_product(*this, right);
}

Vec3d cross_product(Vec3d v1, Vec3d v2) {
    double res_x = v1.get_y()*v2.get_z() - v1.get_z()*v2.get_y();
    double res_y = v1.get_z()*v2.get_x() - v1.get_x()*v2.get_z();
    double res_z = v1.get_x()*v2.get_y() - v1.get_y()*v2.get_x();

    return Vec3d(res_x, res_y, res_z, "xyz");
}