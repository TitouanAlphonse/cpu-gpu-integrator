#include"../headers/massive_body.h"

// #################
// massive_body.cpp
// #################

// See headers/massive_body.h for details about the massive_body class


massive_body::massive_body(double m_, double R_, Vec3d q_init, Vec3d v_init) {
    if (m_ < 0) {
        cout << "Error : negative mass" << endl;
        m = 0;
    }
    else {
        m = m_;
    }

    if (R_ < 0) {
        cout << "Error : negative radius" << endl;
        R = 0;
    }
    else {
        R = R_;
    }

    q = q_init;
    v = v_init;

}


void massive_body::print() {
    cout << "Mass : " << m << " Sun masses" << endl;
    cout << "Radius : " << R*R_Sun/1e3 << " km (" << R << " Sun radii)" << endl;
    cout << "Position (in a.u.):" << endl;
    q.print();
}