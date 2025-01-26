#ifndef GLOBAL_H
#define GLOBAL_H

#include<iostream>
#include<cmath>
#include<chrono>
#include<fstream>
#include<random>
using namespace std;

extern double au;
extern double day;
extern double G;


extern double M_Sun;
extern double R_Sun;

extern double M_Jupiter;
extern double R_Jupiter;
extern double v_Jupiter;
extern double dist_Jupiter;

extern double M_Saturn;
extern double R_Saturn;
extern double v_Saturn;
extern double dist_Saturn;

extern double M_Uranus;
extern double R_Uranus;
extern double v_Uranus;
extern double dist_Uranus;

extern double M_Neptune;
extern double R_Neptune;
extern double v_Neptune;
extern double dist_Neptune;


extern double in_MS(double m);  // gives the mass in number of solar masses
extern double in_MJ(double m);  // gives the mass in number of Jupiter masses
extern double in_RJ(double d);  // gives the distance in number of Jupiter radii
extern double in_au(double d);  // gives the distance in number of astronomical unit

int get_time(chrono::time_point<chrono::high_resolution_clock> start, chrono::time_point<chrono::high_resolution_clock> end);

#endif