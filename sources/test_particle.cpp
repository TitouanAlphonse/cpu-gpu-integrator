#include"../headers/test_particle.h"

test_particle::test_particle(Vec3d q_init, Vec3d v_init) {
    q = q_init;
    v = year/au*v_init;
}


void test_particle::print() {
    cout << "Position (in a.u.):" << endl;
    q.print();
}