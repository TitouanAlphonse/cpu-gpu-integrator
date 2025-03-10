#include"../headers/global.h"

double au = 149597870700; // in m
double year = 365.25*60*60*24; // in s
double day_in_years = 1/365.25; // in years


double M_Sun = 1.98892e30; // in kg
double R_Sun = 696340*1e3; // in m

double M_Jupiter = 1.898e27; // in kg
double R_Jupiter = 71492*1e3; // in m
double v_Jupiter = 13.058*1e3; // in m/s
double dist_Jupiter = 778.5e6*1e3; // in m

double M_Saturn = 5.684e26; // in kg
double R_Saturn = 60268*1e3; // in m
double v_Saturn = 9.68*1e3; // in m/s
double dist_Saturn = 1.434e9*1e3; // in m

double M_Uranus = 8.681e25; // in kg
double R_Uranus = 25559*1e3; // in m
double v_Uranus = 6.796*1e3; // in m/s
double dist_Uranus = 2.871e9*1e3; // in m

double M_Neptune = 1.024e26; // in kg
double R_Neptune = 24764*1e3; // in m
double v_Neptune = 5.432*1e3; // in m/s
double dist_Neptune = 4.495e9*1e3; // in m


double in_SM(double m) {
    return m/M_Sun;
}

double in_JM(double m) {
    return m/M_Jupiter;
}

double in_SR(double d) {
    return d/R_Sun;
}

double in_JR(double d) {
    return d/R_Jupiter;
}


double in_au(double d) {
    return d/au;
}

int get_time(chrono::time_point<chrono::high_resolution_clock> start, chrono::time_point<chrono::high_resolution_clock> end) {
    auto duree_mus = chrono::duration_cast<chrono::microseconds>(end - start);
    int mus = duree_mus.count();
    auto duree_ms = chrono::duration_cast<chrono::milliseconds>(end - start);
    int ms = duree_ms.count();
    auto duree_s = chrono::duration_cast<chrono::seconds>(end - start);
    int s = duree_s.count();
    auto duree_min = chrono::duration_cast<chrono::minutes>(end - start);
    int min = duree_min.count();

    cout << "Execution time : ";
    if (min != 0) {
        cout << min << " min  ";
    }
    if (s != 0) {
        cout << s - min*60 << " s  ";
    }
    if (ms != 0) {
        cout <<  ms - s*1000 << " ms ";
    }

    cout <<  mus - ms*1000 << " µs";
    cout << endl;

    // auto start = chrono::high_resolution_clock::now();
    // auto end = chrono::high_resolution_clock::now();

    return mus;
}