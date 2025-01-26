#include"../headers/massive_body.h"

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
    p = m*v_init;

}


void massive_body::print() {
    cout << "Mass : " << m << " kg (" << in_MJ(m) << " Jupiter masses)" << endl;
    cout << "Radius : " << R << " m (" << in_RJ(R) << " Jupiter radii)" << endl;
    cout << "Position (in number of a.u.):" << endl;
    (q/au).print();
}