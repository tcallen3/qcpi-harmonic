#ifndef QCPI_HARMONIC_H
#define QCPI_HARMONIC_H

#include <fstream>
#include <vector>
#include <map>
#include <stdio.h>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <stdexcept>
#include <gsl/gsl_rng.h>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

// NOTE: comment out (?)
using namespace std;

typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
typedef std::vector<complex<double> > cvector;

// define some effective constants and for simulation

namespace qcpiConstNS
{
const double kcalToHartree = 1.5936e-3;     
const double kb = 3.1668114e-6;             // Boltzmann's constant for K to hartree
const double tlsFreq = 0.00016445;          // system coupling matrix element 
const double mass = 1.0;                
const double hbar = 1.0;                    // using atomic units
const int DSTATES = 2;                      // number of system DVR states
const complex<double> I(0.0, 1.0);          // the imaginary unit

// DVR eigenvals (a multi-state code would generalize here)

const double dvrLeft = 1.0;
const double dvrRight = -1.0;
}

// types to track dynamically consistent hopping

enum Branch {BRANCH_LEFT, BRANCH_MID, BRANCH_RIGHT};
enum Ref {REF_LEFT, REF_RIGHT};

struct Path
{
    vector<unsigned> fwdPath;
    vector<unsigned> bwdPath;
    complex<double> product;
    vector<double> x0;
    vector<double> p0;
};


struct Mode
{
    vector<double> xt;
    
    double c;
    double omega;
    double phase1;
    double phase2;
};

#endif
