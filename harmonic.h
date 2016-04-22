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

using namespace std;

typedef boost::tokenizer<boost::char_separator<char> > Tokenizer;
typedef std::vector<complex<double> > cvector;

namespace qcpiConstNS
{
// file buffer size
const int FLEN = 1024;

// EDIT NOTE: (move to namespace and header)
// semi-constants (working in a.u.) 
const double kcalToHartree = 1.5936e-3; // For converting asymmetry to hartree
const double kb = 3.1668114e-6;         // Boltzmann's constant for K to hartree
const double tlsFreq = 0.00016445;        // off-diagonal element of hamiltonian
const double mass = 1.0;                // all masses taken normalized
const double hbar = 1.0;                // using atomic units
const int DSTATES = 2;                  // number of DVR basis states
const complex<double> I(0.0, 1.0);

// EDIT NOTE: (need to generalize)
// DVR eigenvals (fixed for now) 
const double dvrLeft = 1.0;
const double dvrRight = -1.0;
}
// EDIT NOTE: (Move data structs to header and clean up)

// branching state enums

enum Branch {BRANCH_LEFT, BRANCH_MID, BRANCH_RIGHT};

// reference state enums

enum Ref {REF_LEFT, REF_RIGHT};

// structure definitions
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
