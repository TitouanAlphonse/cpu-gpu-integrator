#include"../headers/Vec3d.h"

// #################
//     Vec3d.cu
// #################


// See headers/Vec3d.h for details about the Vec3d class

Vec3d operator*(double scalar, Vec3d v) {
    return v*scalar;
}

double scalar_product(Vec3d v1, Vec3d v2) {
    return v1.get_x()*v2.get_x() + v1.get_y()*v2.get_y() + v1.get_z()*v2.get_z();
}

Vec3d cross_product(Vec3d v1, Vec3d v2) {
    double res_x = v1.get_y()*v2.get_z() - v1.get_z()*v2.get_y();
    double res_y = v1.get_z()*v2.get_x() - v1.get_x()*v2.get_z();
    double res_z = v1.get_x()*v2.get_y() - v1.get_y()*v2.get_x();

    return Vec3d(res_x, res_y, res_z, "xyz");
}