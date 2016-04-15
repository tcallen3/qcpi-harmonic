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

namespace qcpiConstNS
{
// file buffer size
const int FLEN = 1024;

// EDIT NOTE: (move to namespace and header)
// semi-constants (working in a.u.) 
const double kcal_to_hartree = 1.5936e-3; // For converting asymmetry to hartree
const double kb = 3.1668114e-6;         // Boltzmann's constant for K to hartree
const double tls_freq = 0.00016445;        // off-diagonal element of hamiltonian
const double mass = 1.0;                // all masses taken normalized
const double hbar = 1.0;                // using atomic units
const int DSTATES = 2;                  // number of DVR basis states
const complex<double> I(0.0, 1.0);

// EDIT NOTE: (need to generalize)
// DVR eigenvals (fixed for now) 
const double dvr_left = 1.0;
const double dvr_right = -1.0;
}
// EDIT NOTE: (Move data structs to header and clean up)

// branching state enums

enum Branch {BRANCH_LEFT, BRANCH_MID, BRANCH_RIGHT};

// reference state enums

enum Ref {REF_LEFT, REF_RIGHT};

// type definitions
typedef pair<unsigned, unsigned long long> iter_pair;

// structure definitions
struct Path
{
    vector<unsigned> fwd_path;
    vector<unsigned> bwd_path;
    complex<double> product;
    vector<double> x0;
    vector<double> p0;
};

struct Propagator
{
    complex<double> * prop;
    complex<double> * ham;
    complex<double> * ptemp;
    vector<double> x0_free;
    vector<double> p0_free;
};

struct Mode
{
    vector<double> x_t;
    
    double c;
    double omega;
    double first_phase;
    double second_phase;
};

struct SimInfo
{
    // basic simulation params

    Branch branch_state;     // what state to use for iterative branching
    double asym;
    int bath_modes;
    long mc_steps;
    int ic_tot;
    double dt;
    double rho_dt;
    int qm_steps;
    int kmax;
    int step_pts;
    int chunks;
    int rho_steps;
    double bath_temp;
    double beta; 
    std::string input_name;
    std::string output_name;
    unsigned long seed;
};

#endif
