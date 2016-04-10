/* hbath_mem_analytic.cpp - a program to read in
    a spectral density in atomic units,
    calculate the equivalent harmonic bath 
    modes and couplings, and then run 
    a full QCPI calculation using this bath;
    note that this program only uses the 
    Verlet algorithm to integrate the bath,
    ensuring consistency with later work. 

    This code uses an iterative method on
    top of the QCPI calculation, to allow
    longer time points to be reached without
    performing the full 2^(2N) path sum 

    It also makes use of the EACP trajectories
    as a reference state, in the hopes of
    improving convergence behavior. 
*/

// NOTE: If we take dvr_left = 1.0 and dvr_right = -1.0,
//         then the correctly coordinate-shifted states are 
//         dvr_left = 0.0 and dvr_right = -2.0. Other desired 
//          shifts can be extrapolated from this example.

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

using namespace std;

// file buffer size
const int FLEN = 1024;

// EDIT NOTE: (move to namespace and header)
// semi-constants (working in a.u.) 
const double kcal_to_hartree = 1.5936e-3; // For converting asymmetry to hartree
const double kb = 3.1668114e-6;         // Boltzmann's constant for K to hartree
const double tls_freq = 0.00016445;        // off-diagonal element of hamiltonian
const double mass = 1.0;                // all masses taken normalized
const double hbar = 1.0;                // using atomic units
const double range = 10.0;              // IC range for MC run
const int DSTATES = 2;                  // number of DVR basis states
complex<double> I(0.0,1.0);                // imaginary unit
const long mc_buff = 10000;             // avg. steps per bath mode in MC

// EDIT NOTE: (need to generalize)
// DVR eigenvals (fixed for now) 
const double dvr_left = 1.0;
const double dvr_right = -1.0;

// EDIT NOTE: (eliminate dupe with data structs)
// variables set in config file 

int kmax;                                // memory length

int step_pts; // = 100;                // number of action integration points per QM timestep
int chunks; // = 5;
int chunksize; // = step_pts/chunks;        // must be integer divisor of step_pts
double rho_dt;
int rho_steps; // = 100;                // number of ODE integration points

double bath_temp; // = 247;                 // Az-B temperature in K
double beta; // = 1.0/(kb*bath_temp);       // reciprocal temperature in a.u.
// RNG flag for repeatability
unsigned long seed; // = 179524;

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
    int qm_steps;
    int kmax;
    int step_pts;
    int chunks;
    int rho_steps;
    double bath_temp;
    char infile[FLEN];
    char outfile[FLEN];
};

struct FlagInfo
{
    // debugging flags and RNG seed

    unsigned long seed;
};

// EDIT NOTES: (clean up function list and move to header)

// startup and helper functions
void startup(std::string, struct SimInfo *, struct FlagInfo *, 
        MPI_Comm);
char * nextword(char *, char **);
void print_header(const char *, const char *, int, FILE *);

// funcs to read file
long get_flines(FILE *);
void get_spec(FILE *, double *, double *);

// func to calc w vals
double bath_setup(double *, double *, double *, long, int);

// funcs to perform MC walk
void calibrate_mc(double *, double *, double *, double *, gsl_rng *, SimInfo &);
double ic_gen(double *, double *, double *, double *, double *, double *, 
        gsl_rng *, SimInfo &);
double dist(double, double, double, double, double, double);

// EDIT NOTE: (Need to remove NR ODE functions and reimplement)
// propagator integration
void ho_update_exact(Propagator &, Mode *, double, SimInfo &);
void build_ham(Propagator &, Mode *, int, SimInfo &);
void prop_eqns(double, complex<double> *, complex<double> *, void *);
void rk4(complex<double> * y, complex<double> * dydx, int n, double x, double h, 
    complex<double> * yout, void (*derivs)(double, complex<double> *, complex<double> *, void * params), void * params);
void rkdriver(complex<double> * vstart, complex<double> * out, int nvar, double x1,
    double x2, int nstep, void (*derivs)(double, complex<double> *, complex<double> *, void * params), void * params);

// QCPI functions
void qcpi_update_exact(Path &, Mode *, SimInfo &);
double action_calc_exact(Path &, Mode *, Mode *, SimInfo &);

// simple matrix multiply
void mat_mul(complex<double> *, complex<double> *, complex<double> *, int);

// mapping functions
void map_paths(map<unsigned long long, unsigned> &, 
    vector<Path> &);
void map_single(map<unsigned long long, unsigned> &, 
    Path &, unsigned);
unsigned long long get_binary(Path &);
unsigned long long get_binary(vector<unsigned> &, vector<unsigned> &);

int main(int argc, char * argv[])
{
    // initialize MPI

    MPI_Init(&argc, &argv);
    MPI_Comm w_comm = MPI_COMM_WORLD;

    int me, nprocs;

    MPI_Comm_rank(w_comm, &me);
    MPI_Comm_size(w_comm, &nprocs);

    // process arguments, format should be:
    //      mpirun -n <proc_num> ./prog_name <config_file>

    if (argc < 2)
    {
        std::string exe_name = argv[0];
        throw std::runtime_error("Usage: " + exe_name + " <config_file>\n");
    }
    std::string config_file;   

    config_file = argv[1];

    // create structs to hold configuration parameters

    struct SimInfo simData;
    struct FlagInfo flagData;

    // assign values to config. vars

    startup(config_file.c_str(), &simData, &flagData, w_comm);

    // set global parameters from startup() output

    simData.asym *= kcal_to_hartree;

    kmax = simData.kmax;

    step_pts = simData.step_pts;
    chunks = simData.chunks;
    chunksize = step_pts/chunks;
    rho_steps = simData.rho_steps;

    bath_temp = simData.bath_temp;
    beta = 1.0/(kb*bath_temp);

    seed = flagData.seed;

    // EDIT NOTE: (these should be moved to startup function)
    // sanity checks

    if (kmax <= 0)
        throw std::runtime_error("kmax must be positive\n");

    if (simData.dt <= 0)
        throw std::runtime_error("Quantum timestep must be positive\n");

    if (simData.qm_steps < kmax)
        throw std::runtime_error("Quantum step number smaller than kmax\n");

    if (simData.ic_tot < nprocs)
        throw std::runtime_error("Too few ICs for processor number\n");

    // make sure MC run is long enough

    if (simData.mc_steps < mc_buff * simData.bath_modes)
        simData.mc_steps = mc_buff * simData.bath_modes;

    // open files for I/O

    FILE * infile;
    FILE * outfile;

    infile = fopen(simData.infile, "r");

    if (infile == NULL)
    {
        std::string input_file = simData.infile;
        throw std::runtime_error("Could not open file " + 
                input_file + "\n");
    }

    // only open output file on I/O proc

    if (me == 0)
    {
        outfile = fopen(simData.outfile, "w");

        if (outfile == NULL)
        {
            std::string output_file = simData.outfile;
            throw std::runtime_error("Could not open file " + 
                    output_file + "\n");
        }
    }

    // begin timing calculation

    double start = MPI_Wtime();

    // read in J(w)/w

    long npoints = get_flines(infile);

    double * omega = new double [npoints];
    double * jvals = new double [npoints];

    get_spec(infile, omega, jvals);

    // discretize bath modes

    double * bath_freq = new double [simData.bath_modes];
    double * bath_coup = new double [simData.bath_modes];

    double w0 = bath_setup(omega, jvals, bath_freq, npoints, simData.bath_modes);

    // calculate couplings from frequencies

    double pi = acos(-1.0);

    for (int i = 0; i < simData.bath_modes; i++)
        bath_coup[i] = sqrt(2.0*w0/pi) * bath_freq[i];

    // EDIT NOTES: (enforce even division in code)

    // divide up ICs across procs
    // if IC num evenly divides, we just portion out
    // a block; otherwise we give remainder out block-cyclically
    // with blocksize 1
    
    int base_share = simData.ic_tot/nprocs;
    int remain = simData.ic_tot % nprocs;
    int my_ics;

    if (simData.ic_tot < nprocs)
        throw std::runtime_error("Too few ICs for processor group\n");

    if (remain == 0)
        my_ics = base_share;
    else
    {
        if (me < remain)
            my_ics = base_share+1;
        else
            my_ics = base_share;
    }

    double * xvals = new double [simData.bath_modes];
    double * pvals = new double [simData.bath_modes];

    double * x_step = new double [simData.bath_modes];
    double * p_step = new double [simData.bath_modes];

    // initialize RNG; can change seed to change trajectory behavior

    gsl_rng * gen = gsl_rng_alloc(gsl_rng_mt19937);
    unsigned long s_val = seed * (me+1);
   
    // ensure rng seed is not zero
 
    if (s_val == 0)
        s_val = seed + (me+1);

    gsl_rng_set(gen, s_val);

    // run test MC trials to determine optimal step sizes

    calibrate_mc(x_step, p_step, bath_freq, bath_coup, gen, simData);

    // initialize IC arrays

    for (int i = 0; i < simData.bath_modes; i++)
    {
        if (gsl_rng_uniform_int(gen, 2) == 0)
            xvals[i] = bath_coup[i] * ( dvr_left/(mass*bath_freq[i]*bath_freq[i]) ) +
                gsl_rng_uniform(gen) * x_step[i];
        else
            xvals[i] = bath_coup[i] * ( dvr_left/(mass*bath_freq[i]*bath_freq[i]) ) -
                gsl_rng_uniform(gen) * x_step[i];

        if (gsl_rng_uniform_int(gen, 2) == 0)
            pvals[i] = gsl_rng_uniform(gen) * p_step[i];
        else
            pvals[i] = -gsl_rng_uniform(gen) * p_step[i];
    }


    // define ODE timestep

    rho_dt = simData.dt/chunks;

    // create propagator object

    Propagator curr_prop;

    curr_prop.prop = new complex<double> [DSTATES*DSTATES];
    curr_prop.ptemp = new complex<double> [DSTATES*DSTATES];
    curr_prop.ham = new complex<double> [DSTATES*DSTATES];

    // EDIT NOTES: (consider using small object/struct here)

    // allocate density matrix

    complex<double> ** rho_proc = new complex<double> * [simData.qm_steps];
    complex<double> * rho_ic_proc = new complex<double> [simData.qm_steps];

    for (int i = 0; i < simData.qm_steps; i++)
    {
        rho_proc[i] = new complex<double> [DSTATES*DSTATES];

        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_proc[i][j] = 0.0;
        }
    }

    // initialize path vector

    vector<Path> pathList;
    vector<Path> tempList;

    Path pstart;

    // EDIT NOTES: (change old_ref_list to queue type?)

    // set up variables to store current and past reference state

    Ref * old_ref_list = new Ref [simData.qm_steps];

    old_ref_list[0] = REF_LEFT;

    // initialize harmonic bath arrays

    Mode * modes = new Mode [simData.bath_modes];
    Mode * ref_modes = new Mode [simData.bath_modes];

    for (int i = 0; i < simData.bath_modes; i++)
    {
        modes[i].omega = bath_freq[i];
        modes[i].c = bath_coup[i];

        ref_modes[i].omega = bath_freq[i];
        ref_modes[i].c = bath_coup[i];
    }

    map<unsigned long long, unsigned> pathMap;

    // set up variable to store current reference state

    double ho_ref_state = dvr_left;

    // Following loops are the core computational ones

    // Outer loop goes over initial conditions (selected by MC runs)

    // Inner loop goes over time points - for times less than
    // kmax, this does a full path sum, and for other points,
    // the iterative tensor propagation scheme is used

    for (int ic_curr = 0; ic_curr < my_ics; ic_curr++)
    {
        // zero per-proc rho(t)

        for (int i = 0; i < simData.qm_steps; i++)
            rho_ic_proc[i] = 0.0;

        // generate ICs for HOs using MC walk (we use last step's
        // ICs as current step's IC seed)

        double ratio = ic_gen(xvals, pvals, bath_freq, bath_coup, 
                x_step, p_step, gen, simData);

        // for time points < kmax

        // loop over kmax segments of fwd/bwd paths

        pathList.clear();
        pathMap.clear();
        tempList.clear();
        pstart.fwd_path.clear();
        pstart.bwd_path.clear();

        pstart.fwd_path.push_back(0);   // left-localized
        pstart.bwd_path.push_back(0);   // left-localized
        pstart.product = 1.0;
        pstart.x0.assign(xvals, xvals+simData.bath_modes);
        pstart.p0.assign(pvals, pvals+simData.bath_modes);

        pathList.push_back(pstart);

        // initialize propagator ICs

        curr_prop.x0_free.assign(xvals, xvals+simData.bath_modes);
        curr_prop.p0_free.assign(pvals, pvals+simData.bath_modes);

        // loop over first kmax time points

        for (int seg = 0; seg < kmax; seg++)
        {
            // EDIT NOTES: (this block seems good candidate for function)
            // grow path list vector with child paths

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                Path temp = pathList[path];
        
                for (int fwd = 0; fwd < DSTATES; fwd++)
                {
                    for (int bwd = 0; bwd < DSTATES; bwd++)
                    {
                        Path * incr_path = new Path;
                        (*incr_path) = temp;
                        incr_path->fwd_path.push_back(fwd);
                        incr_path->bwd_path.push_back(bwd);
                
                        tempList.push_back(*incr_path);

                        // free allocation to save memory
            
                        delete incr_path;
                    }
                }
            }

            pathList.clear();
            tempList.swap(pathList);
            tempList.clear();

            // run unforced trajectory and integrate U(t)

            for (int i = 0; i < DSTATES; i++)
            {
                for (int j = 0; j < DSTATES; j++)
                {
                    if (i == j)
                        curr_prop.prop[i*DSTATES+j] = 1.0;
                    else
                        curr_prop.prop[i*DSTATES+j] = 0.0;
                }
            }

            // first find unforced (x,p)
            // note that ho_update_exact clears ref_modes x(t) and p(t) list

            // EDIT NOTES: (rename ho_update fns to something better)

            ho_update_exact(curr_prop, ref_modes, ho_ref_state, simData);

                // chunk trajectory into pieces for greater
                // accuracy in integrating U(t)

            for (int chunk_num = 0; chunk_num < chunks; chunk_num++)
            {
                // construct H(x,p) from bath configuration
            
                build_ham(curr_prop, ref_modes, chunk_num, simData);

                // integrate TDSE for U(t) w/ piece-wise constant
                // Hamiltonian approx.

                rkdriver(curr_prop.prop, curr_prop.ptemp, DSTATES*DSTATES, 
                    0.0, rho_dt, rho_steps, prop_eqns, curr_prop.ham);

                // swap out true and temp pointers

                complex<double> * swap;

                swap = curr_prop.prop;
                curr_prop.prop = curr_prop.ptemp;
                curr_prop.ptemp = swap;

            } // end chunk loop            

            // loop over all paths at this time point

            for (unsigned path = 0; path < pathList.size(); path++)
            { 
                // calculate x(t) and p(t) at integration points
                // along all paths

                qcpi_update_exact(pathList[path], modes, simData);

                // use integration points to find new phase contribution

                double phi;

                phi = action_calc_exact(pathList[path], modes, ref_modes, simData);

                // EDIT NOTES: (block this into function)
                // calculate proper rho contribution

                unsigned size = pathList[path].fwd_path.size();
    
                unsigned splus0 = pathList[path].fwd_path[size-2];
                unsigned splus1 = pathList[path].fwd_path[size-1];

                unsigned sminus0 = pathList[path].bwd_path[size-2];
                unsigned sminus1 = pathList[path].bwd_path[size-1];
                
                unsigned findex = splus1*DSTATES + splus0;
                unsigned bindex = sminus1*DSTATES + sminus0;

                pathList[path].product *= curr_prop.prop[findex] * 
                    conj(curr_prop.prop[bindex]) * exp(I*phi);

                // EDIT NOTES: (block into function)
                // pull out density matrix at each time point

                unsigned rindex = splus1*DSTATES + sminus1;

                rho_proc[seg][rindex] += pathList[path].product;

                if (rindex == 0)
                    rho_ic_proc[seg] += pathList[path].product;

            } // end path loop (full path phase)

            // select reference state for next timestep

            double xi = gsl_rng_uniform(gen);

            if (xi < rho_ic_proc[seg].real())
            {
                ho_ref_state = dvr_left;

                old_ref_list[seg+1] = REF_LEFT;
            }
            else    
            {
                ho_ref_state = dvr_right;

                old_ref_list[seg+1] = REF_RIGHT;
            }

            tempList.clear();

        } // end seg loop (full path phase)

        // EDIT NOTES: (be more explicit about how this works?)

        // slide paths forward one step, i.e. (010) -> (10)
        // note that our system IC means we only
        // have 1/4 of all paths, or this would
        // need to be handled with full T matrix as below

        vector<unsigned> tPath;

        for (unsigned path = 0; path < pathList.size(); path++)
        {
            tPath.assign(pathList[path].fwd_path.begin()+1, 
                pathList[path].fwd_path.end());

            pathList[path].fwd_path.swap(tPath);

            tPath.assign(pathList[path].bwd_path.begin()+1, 
                pathList[path].bwd_path.end());

            pathList[path].bwd_path.swap(tPath);
        }


        // map paths to vector location

        map_paths(pathMap, pathList);

        // loop over time points beyond kmax
        // and propagate system iteratively

        for (int seg = kmax; seg < simData.qm_steps; seg++)
        {
            // choose branch to propagate on stochastically,
            // based on state of system at start of memory span

            unsigned fRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));
            unsigned bRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));

            // using stochastically determined branching
            // and harmonic reference states

                double xi = gsl_rng_uniform(gen);

                // choose reference state for EACP

                if (xi < rho_ic_proc[seg-1].real() )    
                {
                    ho_ref_state = dvr_left;

                    old_ref_list[seg] = REF_LEFT;
                }
                else
                {
                    ho_ref_state = dvr_right;

                    old_ref_list[seg] = REF_RIGHT;
                }

                // choose branching kmax steps back

                if (old_ref_list[seg-kmax] == REF_LEFT)
                    fRand = bRand = 0;
                else
                    fRand = bRand = 1;


            // integrate unforced equations and find U(t)

            for (int i = 0; i < DSTATES; i++)
            {
                for (int j = 0; j < DSTATES; j++)
                {
                    if (i == j)
                        curr_prop.prop[i*DSTATES+j] = 1.0;
                    else
                        curr_prop.prop[i*DSTATES+j] = 0.0;    
                }
            }    

            // first find unforced (x,p)

            ho_update_exact(curr_prop, ref_modes, ho_ref_state, simData);

            // chunk trajectory into pieces for greater
            // accuracy in integrating U(t)

            for (int chunk_num = 0; chunk_num < chunks; chunk_num++)
            {
                // construct H(x,p) from bath configuration
            
                build_ham(curr_prop, ref_modes, chunk_num, simData);

                // integrate TDSE for U(t) w/ piece-wise constant
                // Hamiltonian approx.

                rkdriver(curr_prop.prop, curr_prop.ptemp, DSTATES*DSTATES, 
                    0.0, rho_dt, rho_steps, prop_eqns, curr_prop.ham);

                // swap out true and temp pointers

                complex<double> * swap;

                swap = curr_prop.prop;
                curr_prop.prop = curr_prop.ptemp;
                curr_prop.ptemp = swap;

            } // end chunk loop            

            // set up tempList to hold our matrix mult.
            // intermediates

            tempList.clear();
            tempList = pathList;

            for (unsigned tp = 0; tp < tempList.size(); tp++)
            {
                tempList[tp].product = 0.0;
            }

            // loop over paths to construct tensor contributions
            // in this loop, path variable indexes the input path array
            // since this contains data used to extend path one step

            // fix size of path so any additions aren't double-counted            

            unsigned currSize = pathList.size();
    
            for (unsigned path = 0; path < currSize; path++)
            {
                complex<double> tensorProd;
                complex<double> tensorEacp;

                // loop over all pairs of fwd/bwd system states
                // to generate next step element

                for (int fwd = 0; fwd < DSTATES; fwd++)
                {
                    for (int bwd = 0; bwd < DSTATES; bwd++)
                    {
                        // temporarily update paths for calculations
                        // path length should now be (kmax+1)

                        Path temp;
                        temp = pathList[path];

                        temp.fwd_path.push_back(fwd);
                        temp.bwd_path.push_back(bwd);

                        // calculate x(t) and p(t) at integration points
                        // along new path (note this changes x0, p0 in temp)

                        qcpi_update_exact(temp, modes, simData);

                        // use integration points to find new phase contribution

                        double phi;

                        phi = action_calc_exact(temp, modes, ref_modes, simData);

                        // calculate proper rho contribution

                        unsigned size = temp.fwd_path.size();
    
                        unsigned splus0 = temp.fwd_path[size-2];
                        unsigned splus1 = temp.fwd_path[size-1];

                        unsigned sminus0 = temp.bwd_path[size-2];
                        unsigned sminus1 = temp.bwd_path[size-1];
                
                        unsigned findex = splus1*DSTATES + splus0;
                        unsigned bindex = sminus1*DSTATES + sminus0;

                        // evaluate tensor element

                        tensorProd = curr_prop.prop[findex] * 
                            conj(curr_prop.prop[bindex]) * exp(I*phi);    

                        tensorEacp = curr_prop.prop[findex] * 
                            conj(curr_prop.prop[bindex]);

                        // add tensor result to correct path via index calcs
                        // note that this index comes from the last kmax 
                        // entries of our current kmax+1 list

                        vector<unsigned> ftemp, btemp;

                        ftemp.assign(temp.fwd_path.begin()+1, 
                            temp.fwd_path.end() );

                        btemp.assign(temp.bwd_path.begin()+1, 
                            temp.bwd_path.end() );

                        unsigned long long target = get_binary(ftemp, btemp);
                        
                        // check to see if target path is in list

                        unsigned outPath = pathMap[target];

                        // path is already present in list, so update it

                        tempList[outPath].product += tensorProd * temp.product;
        
                        // update ICs if we have correct donor element

                        if (temp.fwd_path[0] == fRand && temp.bwd_path[0] == bRand)
                        {
                            tempList[outPath].x0.assign(temp.x0.begin(), temp.x0.end());
                            tempList[outPath].p0.assign(temp.p0.begin(), temp.p0.end());
                        }

                        // pop off previous update (returning length to kmax)

                        temp.fwd_path.pop_back();
                        temp.bwd_path.pop_back();

                    } // end bwd loop

                } // end fwd loop

            } // end path loop (iter. phase)    

            // swap tempList and pathList

            pathList.swap(tempList);

            // pull out current density matrix

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                unsigned size = pathList[path].fwd_path.size();
                unsigned splus1 = pathList[path].fwd_path[size-1];
                unsigned sminus1 = pathList[path].bwd_path[size-1];

                unsigned rindex = splus1*DSTATES + sminus1;

                rho_proc[seg][rindex] += pathList[path].product;

                if (rindex == 0)
                    rho_ic_proc[seg] += pathList[path].product;

            } // end update loop (iter. phase)

        } // end seg loop (iter. phase)

    } // end IC loop

    // collect timing data

    double local_time = MPI_Wtime() - start;
    double g_runtime = 0.0;

    MPI_Allreduce(&local_time, &g_runtime, 1, MPI_DOUBLE,
        MPI_MAX, w_comm);

    // EDIT NOTES: (chunk MPI reduction into new functions)

    // collect real and imag parts of rho into separate
    // arrays for MPI communication

    double * rho_real_proc = new double [simData.qm_steps*DSTATES*DSTATES];
    double * rho_imag_proc = new double [simData.qm_steps*DSTATES*DSTATES];

    double * rho_real = new double [simData.qm_steps*DSTATES*DSTATES];
    double * rho_imag = new double [simData.qm_steps*DSTATES*DSTATES];

    for (int i = 0; i < simData.qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real_proc[i*DSTATES*DSTATES+j] = rho_proc[i][j].real();
            rho_imag_proc[i*DSTATES*DSTATES+j] = rho_proc[i][j].imag();

            rho_real[i*DSTATES*DSTATES+j] = 0.0;
            rho_imag[i*DSTATES*DSTATES+j] = 0.0;
        }
    }

    // Allreduce the real and imaginary arrays

    MPI_Allreduce(rho_real_proc, rho_real, simData.qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    MPI_Allreduce(rho_imag_proc, rho_imag, simData.qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    // scale arrays by Monte Carlo factor

    for (int i = 0; i < simData.qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real[i*DSTATES*DSTATES+j] /= simData.ic_tot;
            rho_imag[i*DSTATES*DSTATES+j] /= simData.ic_tot;
        }
    }

    // output summary

      if (me == 0)
      {
        int repeat = 50;
        
        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nSimulation Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Quantum steps: %d\n", simData.qm_steps);
        fprintf(outfile, "Memory length (kmax): %d\n", kmax);
        fprintf(outfile, "Step length (a.u): %.5f\n", simData.dt);
        fprintf(outfile, "IC num: %d\n", simData.ic_tot);
        fprintf(outfile, "RNG seed: %lu\n", seed);
        fprintf(outfile, "MC skip: %ld\n", simData.mc_steps);

        fprintf(outfile, "Analytic trajectory integration: on\n");

        fprintf(outfile, "Simulation used EACP reference hopping\n");

        fprintf(outfile, "Input spectral density: %s\n", simData.infile);
        fprintf(outfile, "Configuration file: %s\n\n", config_file.c_str());

        fprintf(outfile, "Total simulated time (a.u.): %.4f\n", simData.qm_steps*simData.dt);

        fprintf(outfile, "Processors: %d\n\n", nprocs);
        fprintf(outfile, "Total simulation time: %.3f min\n\n", g_runtime/60.0);

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nBath Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Bath modes: %d\n", simData.bath_modes);
        fprintf(outfile, "Bath temperature: %.2f\n", bath_temp);
        fprintf(outfile, "Inverse temperature: %.4f\n", beta);
        fprintf(outfile, "Bath mode mass parameter: %.3f\n", mass);
        fprintf(outfile, "Using shifted W(x,p) (minimum at x=lambda)\n\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nSystem Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Off-diagonal TLS element: %f\n", tls_freq);
        fprintf(outfile, "Asymmetry: %.7e hartree\n", simData.asym);
        fprintf(outfile, "Left DVR state: %.3f\n", dvr_left);
        fprintf(outfile, "Right DVR state: %.3f\n\n", dvr_right);

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nDensity Matrix Values\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

       
            for (int i = 0; i < simData.qm_steps; i++)
            {
                int entry = i*DSTATES*DSTATES;

                fprintf(outfile, "%7.4f %8.5f %6.3f (Tr = %13.10f)\n", (i+1)*simData.dt, 
                    rho_real[entry], 0.0, rho_real[entry]+rho_real[entry+3]);
            }


        fprintf(outfile, "\n");

      } // end output conditional

    // cleanup

    delete [] omega;
    delete [] jvals;
    delete [] bath_freq;
    delete [] bath_coup;

    delete [] xvals;
    delete [] pvals;
    delete [] x_step;
    delete [] p_step;

    //delete [] prop;
    delete [] curr_prop.prop;
    delete [] curr_prop.ptemp;
    delete [] curr_prop.ham;

    for (int i = 0; i < simData.qm_steps; i++)
    {
        delete [] rho_proc[i];
    }

    delete [] rho_proc;
    delete [] rho_ic_proc;
    delete [] modes;
    delete [] ref_modes;
    delete [] old_ref_list;
    delete [] rho_real_proc;
    delete [] rho_imag_proc;
    delete [] rho_real;
    delete [] rho_imag;

    MPI_Finalize();

    return 0;
}

/*----------------------------------------------------------------------*/

// EDIT NOTES: (get rid of LAMMPS-based tokenizing)
// startup() -- read in and process configuration file 

void startup(std::string config, struct SimInfo * sim, struct FlagInfo * flag, 
        MPI_Comm comm)
{
    int me;

    MPI_Comm_rank(comm, &me);

    ifstream conf_file;
    char buffer[FLEN];
    char * arg1;
    char * arg2;
    char * next;
    char * ptr;

    // set defaults

    sim->branch_state = BRANCH_LEFT; // defaults to left state branch
    sim->asym = 0.5;                // system asymmetry (in kcal/mol)
    sim->bath_modes = 60;           // number of bath oscillators
    sim->mc_steps = 50000;          // default MC burn
    sim->step_pts = 100;            // number of action integration points
    sim->chunks = 5;                // must evenly divide step_pts
    sim->rho_steps = 100;            // points used to integrate U(t)

    // note that original definitions use integer
    // flags, which is why these are ints and not bool

    flag->seed = 179524;            // GSL RNG seed

    // initialize flags to check validity of
    // configuration file input

    bool qmstepsflg = false;
    bool qmdtflg = false;
    bool bathtempflg = false;

    bool icnumflg = false;
    bool kmaxflg = false;

    bool inflg = false;
    bool outflg = false;

    // set category flags

    bool timeflg;
    bool simflg;
    bool fileflg;

    bool reqflg;

    conf_file.open(config.c_str(), ios_base::in);
    if (!conf_file.is_open())
        throw std::runtime_error("Could not open configuration file\n");

    // read in file line-by-line

    while (conf_file.getline(buffer, FLEN))
    {   
        // tokenize

        arg1 = nextword(buffer, &next);

        if (arg1 == NULL || next == NULL)   // skip blank lines
            continue;

        ptr = next;

        arg2 = nextword(ptr, &next);      
           
        // execute logic on tokens (normalize case?)

        if (strcmp(arg1, "rho_steps") == 0)
            sim->rho_steps = atoi(arg2);

        else if (strcmp(arg1, "qm_steps") == 0)
        {
            sim->qm_steps = atoi(arg2);
            qmstepsflg = true;
        }

        else if (strcmp(arg1, "timestep") == 0)
        {
            sim->dt = atof(arg2);
            qmdtflg = true;
        }

        else if (strcmp(arg1, "ic_num") == 0 || strcmp(arg1, "tot_ics") == 0)
        {
            sim->ic_tot = atoi(arg2);
            icnumflg = true;
        }

        else if (strcmp(arg1, "kmax") == 0)
        {
            sim->kmax = atoi(arg2);
            kmaxflg = true;
        }

        else if (strcmp(arg1, "temperature") == 0)
        {
            sim->bath_temp = atof(arg2);
            bathtempflg = true;
        }

        else if (strcmp(arg1, "in_file") == 0 || strcmp(arg1, "input") == 0)
        {
            sprintf(sim->infile, "%s", arg2);
            inflg = true;
        }

        else if (strcmp(arg1, "out_file") == 0 || strcmp(arg1, "output") == 0)
        {
            sprintf(sim->outfile, "%s", arg2);
            outflg = true;
        }

        else if (strcmp(arg1, "asymmetry") == 0)
        {
            // set system asymmetry

            sim->asym = atof(arg2);
        }

        else if (strcmp(arg1, "bath_modes") == 0)
        {
            // set # of bath oscillators

            sim->bath_modes = atoi(arg2);
        }

        else if (strcmp(arg1, "mc_steps") == 0)
        {
            // set MC burn size

            sim->mc_steps = atol(arg2);
        }

        else if (strcmp(arg1, "step_pts") == 0)
        {
            // set action integration points per step

            sim->step_pts = atoi(arg2);
        }

        else if (strcmp(arg1, "chunks") == 0)
        {
            // set # of action grouping chunks

            sim->chunks = atoi(arg2);
        }

        else if (strcmp(arg1, "rng_seed") == 0)
        {
            // set RNG seed (ensure seed isn't 0)

            float f_seed = atof(arg2);

            flag->seed = static_cast<unsigned long>(f_seed);

            if (flag->seed == 0)
                flag->seed = 179524;
        }

        else    // skip unrecognized commands and weird lines
            continue;
    }

    // check required flags

    timeflg = qmstepsflg && qmdtflg;

    simflg = icnumflg && kmaxflg && bathtempflg;

    fileflg = inflg && outflg;

    reqflg = timeflg && simflg && fileflg;

    if (!reqflg)
    {
        if (!timeflg)
            throw std::runtime_error("Must specify number of quantum steps and timestep values\n");

        if (!simflg)
            throw std::runtime_error("Must specify number of ICs, kmax, and bath temperature\n"); 

        if (!fileflg)
            throw std::runtime_error("Must specify spectral density input file, and data output file\n");
    }

    // ensure consistency of step_pts and chunk values

    int factor = sim->step_pts/sim->chunks;

    sim->step_pts = factor * sim->chunks;

    // check positive-definite quantities

    if (sim->bath_modes <= 0)
        throw std::runtime_error("Bath oscillator number must be positive\n");

    if (sim->ic_tot <= 0)
        throw std::runtime_error("Number of ICs must be positive\n");

    if (sim->dt <= 0)
        throw std::runtime_error("Timestep must be positive\n");

    if (sim->qm_steps <= 0)
        throw std::runtime_error("Total simulation steps must be positive\n");

    if (sim->kmax <= 0)
        throw std::runtime_error("Kmax segments must be positive\n");

    if (sim->step_pts <= 0)
        throw std::runtime_error("Numer of action integration points must be positive\n");

    if (sim->chunks <= 0)
        throw std::runtime_error("Action chunk number must be positive\n");

    if (sim->rho_steps <= 0)
        throw std::runtime_error("Number of ODE steps for rho(t) must be positive\n");

    if (sim->bath_temp <= 0)
        throw std::runtime_error("Bath temperature must be positive\n");

    // check non-negative quantities

    if (sim->mc_steps < 0)
        throw std::runtime_error("Monte Carlo burn length can't be negative\n");

    // ensure kmax < total steps

    if (sim->kmax > sim->qm_steps)
        throw std::runtime_error("Memory length cannot exceed total simulation time\n");
    
    conf_file.close();
}

/* ----------------------------------------------------------------------
   nextword() behavior:

   find next word in str
   insert 0 at end of word
   ignore leading whitespace
   treat text between single/double quotes as one arg
   matching quote must be followed by whitespace char if not end of string
   strip quotes from returned word
   return ptr to start of word
   return next = ptr after word or NULL if word ended with 0
   return NULL if no word in string
------------------------------------------------------------------------- */

char * nextword(char *str, char **next)
{
  char *start,*stop;

  start = &str[strspn(str," \t\n\v\f\r")];
  if (*start == '\0') return NULL;
  
  if (*start == '"' || *start == '\'') {
    stop = strchr(&start[1],*start);
    if (!stop) throw std::runtime_error("Unbalanced quotes in input line\n");
    if (stop[1] && !isspace(stop[1]))
      throw std::runtime_error("Input line quote not followed by whitespace\n");
    start++;
  } else stop = &start[strcspn(start," \t\n\v\f\r")];
  
  if (*stop == '\0') *next = NULL;
  else *next = stop+1;
  *stop = '\0';

  return start;
}

/*----------------------------------------------------------------------*/

// print_header() -- prints a message (msg) enclosed vertically by
// repeated separator symbols (used for pretty printing)

void print_header(const char * msg, const char * separator, 
    int repeat, FILE * outfile)
{
    for (int i = 0; i < repeat; i++)
        fprintf(outfile, "%s", separator);

    fprintf(outfile, "\n");

    fprintf(outfile, "%s", msg);

    fprintf(outfile, "\n");

    for (int i = 0; i < repeat; i++)
        fprintf(outfile, "%s", separator);

    fprintf(outfile, "\n\n");
}

/* ------------------------------------------------------------------------ */

long get_flines(FILE * fp)
{
    long lines = 0;
    char buf[FLEN];

    while (fgets(buf, FLEN, fp) != NULL)
    {
        // ignore blank lines and comments

        if (buf[0] == '\0' || buf[0] == '#' || buf[0] == '\n')
            continue;   
        else
            lines++;
    }

    // restore original file position
    // could make this more general by
    // using ftell() above in case we start
    // mid-file

    fseek(fp, 0L, SEEK_SET);

    return lines;
}

/* ------------------------------------------------------------------------ */

void get_spec(FILE * fp, double * omega, double * jvals)
{
    char buf[FLEN];
    long index = 0;

    // read in column pairs, skipping blank lines and comments

    // assumes (w,J(w)) ordering for file

    while (fgets(buf, FLEN, fp) != NULL)
    {
        // skip blank lines and comments

        if (buf[0] == '\0' || buf[0] == '#' || buf[0] == '\n')
            continue;
        else
        {
            omega[index] = atof(strtok(buf, " \t\n"));
            jvals[index] = atof(strtok(NULL, " \t\n"));
    
            index++;
        }
    }
}

/* ------------------------------------------------------------------------ */

double bath_setup(double * omega, double * jvals, double * bath_freq, long pts,
    int nmodes)
{
    // first calculate reorg energy via simple integration

    double dw = omega[1] - omega[0];    // assumes uniform spacing
    double sum = 0.0;

    sum += (jvals[0] + jvals[pts-1])/2.0;

    for (long i = 1; i < pts-1; i++)
        sum += jvals[i];

    sum *= dw;

    double w0 = sum/nmodes;

    // report reorg energy to command line

    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    
    double pi = acos(-1.0);
    double reorg = 4.0 * sum / pi;

    if (me == 0)
    {
        //fprintf(stdout, "dw = %.15e\n", dw);
        fprintf(stdout, "Reorg. energy (au): %.7f\n", reorg);
    }

    // discretize frequencies

    // NOTE: notation here is weird to match Tuseeta's
    //  results; I can't seem to replicate them using
    //  normal C syntax, should probably look into this

    for (int j = 0; j < nmodes; j++)
    {
        sum = 0.0;
        long i = -1;
    
        while (sum <= (j+1) && i < pts-2)
        {
            i++;
            sum += jvals[i]*dw/w0;
        }

        bath_freq[j] = omega[i];
    }

    // check modes for 0 vals (may create workaround)

    for (int i = 0; i < nmodes; i++)
    {
        if (bath_freq[i] == 0)
        {
            for (int j = i; j < nmodes; j++)
            {
                if (bath_freq[j] != 0)
                {
                    bath_freq[i] = bath_freq[j]/2.0;    
                    break;
                }
            }

            if (bath_freq[i] == 0)
                throw std::runtime_error("All modes have zero frequency\n");
        }
    }

    return w0;
}

/*----------------------------------------------------------------------*/

void calibrate_mc(double * x_step, double * p_step, double * bath_freq, 
    double * bath_coup, gsl_rng * gen, SimInfo & simData)
{
    const double max_trials = 1000;
    const double base_step = 10.0;
    const double scale = 0.8;
    const double low_thresh = 0.45;
    const double high_thresh = 0.55;
    const int tsteps = 1000;
    double * x_vals = new double [simData.bath_modes];
    double * p_vals = new double [simData.bath_modes];

    // initialize step sizes and (x,p) at minimum of x

    for (int i = 0; i < simData.bath_modes; i++)
    {
        x_step[i] = gsl_rng_uniform(gen) * base_step;
        p_step[i] = gsl_rng_uniform(gen) * base_step;
        x_vals[i] = bath_coup[i] * ( dvr_left/(mass*bath_freq[i]*bath_freq[i]) );
        p_vals[i] = 0.0;
    }

    double x_old, x_new;
    double p_old, p_new;
    int accepted;

    // run MC step tweaking for each mode and phase space dim. separately

    for (int i = 0; i < simData.bath_modes; i++)
    {
        int count = 0;

        while (count <= max_trials)    // keep looping until convergence
        {
            x_old = x_vals[i];
            p_old = p_vals[i];
            accepted = 1;

            for (int j = 0; j < tsteps; j++)
            {
                double step_len = gsl_rng_uniform(gen) * x_step[i];

                // displace x coord.

                if (gsl_rng_uniform_int(gen, 2) == 0)
                    x_new = x_old + step_len;
                else
                    x_new = x_old - step_len;

                // keep p constant for this run

                p_new = p_old;

                // pass in p_old for both configs
                // this makes sure p isn't affected

                double prob = dist(x_old, x_new, p_old, p_old, bath_freq[i], 
                    bath_coup[i]);

                if (prob >= 1.0) // accept
                {
                    // update state
                    x_old = x_new;     
                    p_old = p_new;
                    accepted++;
                }
                else
                {    
                    double xi = gsl_rng_uniform(gen);
    
                    if (prob >= xi) // accept (rescue)
                    {    
                        // update state
                        x_old = x_new;     
                        p_old = p_new;
                        accepted++;  
                    }
                    else // reject
                    {
                        // technically a null operation
                        x_old = x_old;
                        p_old = p_old;
                    }
                }

            } // end x-space MC trials

            double ratio = static_cast<double>(accepted)/tsteps;

            // check ratio for convergence

            if (ratio >= low_thresh && ratio <= high_thresh)    // step is good
                break;
            else if (ratio < low_thresh)                // step too large, decrease
                x_step[i] *= scale;
            else                                 // step too small, increase
                x_step[i] /= scale;    
    

            count++;

        } // end while loop

        if (count >= max_trials)
            throw std::runtime_error("Failed to properly converge MC optimization in x-coord.\n");

    } // end x calibration

    // begin calibrating p steps

    for (int i = 0; i < simData.bath_modes; i++)
    {
        x_vals[i] = bath_coup[i] * ( dvr_left/(mass*bath_freq[i]*bath_freq[i]) );
        p_vals[i] = 0.0;
    }

    // run MC step tweaking for each mode 

    for (int i = 0; i < simData.bath_modes; i++)
    {
        int count = 0;

        while (count <= max_trials)    // keep looping until convergence
        {
            x_old = x_vals[i];
            p_old = p_vals[i];
            accepted = 1;

            for (int j = 0; j < tsteps; j++)
            {
                double step_len = gsl_rng_uniform(gen) * p_step[i];

                // displace p coord.

                if (gsl_rng_uniform_int(gen, 2) == 0)
                    p_new = p_old + step_len;
                else
                    p_new = p_old - step_len;

                // keep x constant for this run

                x_new = x_old;

                // use x_old for both configurations
                // this makes sure x isn't affected

                double prob = dist(x_old, x_old, p_old, p_new, bath_freq[i], 
                    bath_coup[i]);

                if (prob >= 1.0) // accept
                {
                    // update state
                    x_old = x_new;     
                    p_old = p_new;
                    accepted++;
                }
                else
                {    
                    double xi = gsl_rng_uniform(gen);
    
                    if (prob >= xi) // accept (rescue)
                    {    
                        // update state
                        x_old = x_new;     
                        p_old = p_new;
                        accepted++;  
                    }
                    else // reject
                    {
                        // technically a null operation
                        x_old = x_old;
                        p_old = p_old;
                    }
                }

            } // end p-space MC trials

            double ratio = static_cast<double>(accepted)/tsteps;

            // check ratio for convergence

            if (ratio >= low_thresh && ratio <= high_thresh)    // step is good
                break;
            else if (ratio < low_thresh)                // step too large, decrease
                p_step[i] *= scale;
            else                                 // step too small, increase
                p_step[i] /= scale;    
    
            count++;

        } // end while loop

        if (count >= max_trials)
            throw std::runtime_error("Failed to converge MC optimization in p-coord.\n");

    } // end p tweak

    delete [] x_vals;
    delete [] p_vals; 
}

/* ------------------------------------------------------------------------ */

double ic_gen(double * xvals, double * pvals, double * bath_freq, double * bath_coup,
    double * x_step, double * p_step, gsl_rng * gen, SimInfo & simData)
{   
    //double step_max = range/100.0;

    double x_old, x_new;
    double p_old, p_new;

    long accepted = 1;

    for (long i = 1; i < simData.mc_steps; i++)
    { 
        // randomly select index to step, and generate
        // step size in x and p dimension

        int index = gsl_rng_uniform_int(gen, simData.bath_modes);
        double x_len = gsl_rng_uniform(gen) * x_step[index];
        double p_len = gsl_rng_uniform(gen) * p_step[index];

        x_old = xvals[index];
        p_old = pvals[index];

        // displace x coord.

        if (gsl_rng_uniform_int(gen, 2) == 0)
            x_new = x_old + x_len;
        else
            x_new = x_old - x_len;

        // displace p coord.

        if (gsl_rng_uniform_int(gen, 2) == 0)
            p_new = p_old + p_len;
        else
            p_new = p_old - p_len;

        // find probability ratio of new configuration
        // stationary modes divide out, so we only pass info
        // for the currently active mode specified by index

        double prob = dist(x_old, x_new, p_old, p_new, bath_freq[index], 
            bath_coup[index]);

        if (prob >= 1.0) // accept
        {
            // update state
            xvals[index] = x_new;     
            pvals[index] = p_new;
            accepted++;
        }
        else
        {
            double xi = gsl_rng_uniform(gen);
    
            if (prob >= xi) // accept (rescued)
            {
                // update state
                xvals[index] = x_new;     
                pvals[index] = p_new;
                accepted++;  
            }
            else // reject
            {
                // technically a null operation
                xvals[index] = x_old;
                pvals[index] = p_old;
            }
        }

    }

    double ratio = static_cast<double>(accepted)/simData.mc_steps;

    return ratio;
}

/* ------------------------------------------------------------------------ */

double dist(double x_old, double x_new, double p_old, double p_new, 
    double omega, double coup)
{
    // need atomic units     
    double f = hbar*omega*beta;

    // shift to DVR state w/ -1 element in sigma_z basis

    double lambda = dvr_left * coup * ( 1.0/(mass*omega*omega) );

    // shifting distribution for equilibrium

    x_new -= lambda;
    x_old -= lambda;

    // calculate Wigner distribution ratio for these x,p vals

    double delta = (mass*omega/hbar)*(x_new*x_new - x_old*x_old) +
        (1.0/(mass*omega*hbar))*(p_new*p_new - p_old*p_old);

    double pre = tanh(f/2.0);

    double prob = exp(-pre*delta);

    return prob; 
}

/* ------------------------------------------------------------------------ */

void ho_update_exact(Propagator & prop, Mode * mlist, double ref_state, 
        SimInfo & simData)
{
    double del_t = simData.dt/2.0;
    double chunk_dt = simData.dt/chunks;

    for (int mode = 0; mode < simData.bath_modes; mode++)
    {
        double x0, xt;
        double p0, pt;

        double w = mlist[mode].omega;
        double shift = (ref_state * mlist[mode].c)/(mass * w * w);

        // first calculated x(t) at time points
        // for propagator integration

        // clear out any old trajectory info
        // might be more efficient just to overwrite

        mlist[mode].x_t.clear();

        // set up ICs for chunk calc

        x0 = prop.x0_free[mode];
        p0 = prop.p0_free[mode];
        
        for (int i = 0; i < chunks; i++)
        {
            // find x(t) at chunk time points

            xt = (x0 - shift)*cos(w*chunk_dt) + 
                (p0/(mass*w))*sin(w*chunk_dt) + shift;

            pt = p0*cos(w*chunk_dt) - 
                mass*w*(x0 - shift)*sin(w*chunk_dt);

            mlist[mode].x_t.push_back(xt);

            x0 = xt;
            p0 = pt;
        }

        // set up ICs for trajectory

        x0 = prop.x0_free[mode];
        p0 = prop.p0_free[mode];

        // calculate time-evolved x(t), p(t) for
        // first half-step of trajectory

        xt = (x0 - shift)*cos(w*del_t) + (p0/(mass*w))*sin(w*del_t) + shift;

        pt = p0*cos(w*del_t) - mass*w*(x0 - shift)*sin(w*del_t);

        mlist[mode].first_phase = (x0 - shift)*sin(w*del_t) - 
            (p0/(mass*w))*(cos(w*del_t)-1.0) + shift*w*del_t;

        // swap x0, p0 and xt, pt

        x0 = xt;
        p0 = pt;

        // calculate time-evolved x(t), p(t) for
        // second half-step of trajectory

        xt = (x0 - shift)*cos(w*del_t) + (p0/(mass*w))*sin(w*del_t) + shift;

        pt = p0*cos(w*del_t) - mass*w*(x0 - shift)*sin(w*del_t);

        mlist[mode].second_phase = (x0 - shift)*sin(w*del_t) - 
            (p0/(mass*w))*(cos(w*del_t)-1.0) + shift*w*del_t;

        // update current phase space point

        prop.x0_free[mode] = xt;
        prop.p0_free[mode] = pt;

    } // end mode loop
}

/* ------------------------------------------------------------------------- */

// construct system-bath Hamiltonian for
// current timestep

void build_ham(Propagator & prop, Mode * modes, int chunk, SimInfo & simData)
{
    // copy off-diagonal from anharmonic code
    const double off_diag = hbar*tls_freq;

    // store sums for diagonal matrix elements of H
    // left_sum for (0,0); right_sum for (1,1)
    double left_sum;
    double right_sum;

    // energy stores harmonic bath potential energy
    // this should integrate out, but is included here
    // for completeness
    double energy = 0.0;

    // store system and system-bath coupling contributions separately
    complex<double> tls_mat[4];
    complex<double> bath_mat[4];

    // system matrix is just splitting and any asymmetry
    // note that signs should be standard b/c I'm using
    // (-1,+1) instead of (+1,-1)

    tls_mat[0] = simData.asym; //dvr_left*(1.0*asym);
    tls_mat[1] = -1.0*off_diag;
    tls_mat[2] = -1.0*off_diag;
    tls_mat[3] = -1.0*simData.asym; //dvr_right*(1.0*asym);

    // system-bath matrix includes linear coupling plus
    // quadratic offset

    // in this form, it also includes the bath potential energy

    left_sum = 0.0;
    right_sum = 0.0;

    for (int i = 0; i < simData.bath_modes; i++)
    {
        double csquare = modes[i].c*modes[i].c;
        double wsquare = modes[i].omega*modes[i].omega;
        double x = modes[i].x_t[chunk];

        left_sum += -1.0*modes[i].c*x*dvr_left +
            csquare*dvr_left*dvr_left/(2.0*mass*wsquare);

        right_sum += -1.0*modes[i].c*x*dvr_right +
            csquare*dvr_right*dvr_right/(2.0*mass*wsquare);

        energy += 0.5*mass*wsquare*x*x;
    }

    // Removing energy term to see if this is
    // dominating energy gap and causing issues

    bath_mat[0] = left_sum;
    bath_mat[1] = 0.0;
    bath_mat[2] = 0.0;
    bath_mat[3] = right_sum;

    // total hamiltonian is sum of system and system-bath parts

    prop.ham[0] = tls_mat[0] + bath_mat[0];
    prop.ham[1] = tls_mat[1] + bath_mat[1];
    prop.ham[2] = tls_mat[2] + bath_mat[2];
    prop.ham[3] = tls_mat[3] + bath_mat[3];

} // end build_ham()

/* ------------------------------------------------------------------------ */

void qcpi_update_exact(Path & qm_path, Mode * mlist, SimInfo & simData)
{
    double del_t = simData.dt/2.0;
    double dvr_vals[DSTATES] = {dvr_left, dvr_right};

    for (int mode = 0; mode < simData.bath_modes; mode++)
    {
        // set up ICs for first half of path

        double w = mlist[mode].omega;
        double c = mlist[mode].c;

        double x0, xt;
        double p0, pt;

        x0 = qm_path.x0[mode];
        p0 = qm_path.p0[mode];

        unsigned size = qm_path.fwd_path.size();

        unsigned splus = qm_path.fwd_path[size-2];
        unsigned sminus = qm_path.bwd_path[size-2];

        double shift = (dvr_vals[splus] + dvr_vals[sminus])/2.0;

        shift *= c/(mass*w*w);

        // clear out any old trajectory info
        // might be more efficient just to overwrite

        mlist[mode].x_t.clear();

        // calculate time evolution for first
        // half of path

        xt = (x0 - shift)*cos(w*del_t) + (p0/(mass*w))*sin(w*del_t) +
            shift;

        pt = p0*cos(w*del_t) - mass*w*(x0 - shift)*sin(w*del_t);

        mlist[mode].first_phase = (x0 - shift)*sin(w*del_t) -
            (p0/(mass*w))*(cos(w*del_t) - 1.0) + shift*w*del_t;

        // swap x0, p0 with xt, pt

        x0 = xt;
        p0 = pt;

        // find s vals and shift for second half of trajectory

        splus = qm_path.fwd_path[size-1];
        sminus = qm_path.bwd_path[size-1];

        shift = (dvr_vals[splus] + dvr_vals[sminus])/2.0;

        shift *= c/(mass*w*w);

        // calculate time evolution for second
        // half of path

        xt = (x0 - shift)*cos(w*del_t) + (p0/(mass*w))*sin(w*del_t) +
            shift;

        pt = p0*cos(w*del_t) - mass*w*(x0 - shift)*sin(w*del_t);

        mlist[mode].second_phase = (x0 - shift)*sin(w*del_t) -
            (p0/(mass*w))*(cos(w*del_t) - 1.0) + shift*w*del_t;

        // update current phase space point

        qm_path.x0[mode] = xt;
        qm_path.p0[mode] = pt;

    } // end mode loop
}

/* ------------------------------------------------------------------------- */

double action_calc_exact(Path & qm_path, Mode * mlist, Mode * reflist, 
        SimInfo & simData)
{
    double dvr_vals[DSTATES] = {dvr_left, dvr_right};

    // loop over modes and integrate their action contribution
    // as given by S = c*int{del_s(t')*x(t'), 0, t_final}

    double action = 0.0;

    for (int mode = 0; mode < simData.bath_modes; mode++)
    {
        double sum = 0.0;

        // set indices to correct half steps

        unsigned size = qm_path.fwd_path.size();

        unsigned splus = qm_path.fwd_path[size-2];
        unsigned sminus = qm_path.bwd_path[size-2];

        double ds = dvr_vals[splus] - dvr_vals[sminus];
        double pre = (mlist[mode].c * ds)/mlist[mode].omega;

        // find first half of action sum

        sum += pre * (mlist[mode].first_phase - 
            reflist[mode].first_phase);

        // recalculate prefactor

        splus = qm_path.fwd_path[size-1];
        sminus = qm_path.bwd_path[size-1];

        ds = dvr_vals[splus] - dvr_vals[sminus];
        pre = (mlist[mode].c * ds)/mlist[mode].omega;

        // find second half of action sum

        sum += pre * (mlist[mode].second_phase - 
            reflist[mode].second_phase);

        action += sum;

    } // end mode loop

    return action;
}

/* ------------------------------------------------------------------------ */

// propagator evolution

void prop_eqns(double t, complex<double> * y, complex<double> * dydt, 
    void * params)
{
    complex<double> * ham = (complex<double> *) params;
    complex<double> I(0.0, 1.0);

    // this is component version of i*hbar*(dU/dt) = H*U
    // could as write as traditional matrix multiplication

    dydt[0] = -(I/hbar)*(ham[0]*y[0] + ham[1]*y[2]);
    dydt[1] = -(I/hbar)*(ham[0]*y[1] + ham[1]*y[3]);
    dydt[2] = -(I/hbar)*(ham[2]*y[0] + ham[3]*y[2]);
    dydt[3] = -(I/hbar)*(ham[2]*y[1] + ham[3]*y[3]);
}

/* ------------------------------------------------------------------------ */

/* rk4() implements a vanilla fourth-order Runge-Kutta ODE solver, 
although this version has been tweaked to use complex numbers, since
robust complex libraries are hard to find. The derivs fn pointer specifies
the function that will define the ODE equations, and the params array is
included for extra flexibility, as per GSL */

void rk4(complex<double> * y, complex<double> * dydx, int n, double x, double h, 
    complex<double> * yout, void (*derivs)(double, complex<double> *, complex<double> *, void * params), void * params)
{
    int i;
    double xh, h_mid, h_6;
    complex<double> *dym, *dyt, *yt;

    dym = new complex<double>[n];
    dyt = new complex<double>[n];
    yt = new complex<double>[n];

    h_mid = 0.5*h;
    h_6 = h/6.0;
    xh = x+h_mid;

    for (i = 0; i < n; i++)
        yt[i] = y[i] + h_mid*dydx[i];    /* first step */
    
    (*derivs)(xh, yt, dyt, params);        /* second step */
    
    for (i = 0; i < n; i++)
        yt[i] = y[i] + h_mid*dyt[i];    

    (*derivs)(xh, yt, dym, params);        /* third step */

    for (i = 0; i < n; i++)
    {
        yt[i] = y[i] + h*dym[i];
        dym[i] += dyt[i];
    }
    
    (*derivs)(x+h, yt, dyt, params);    /* fourth step */

    for (i = 0; i < n; i++)
        yout[i] = y[i] + h_6*(dydx[i] + dyt[i] + 2.0*dym[i]);

    delete [] dym;
    delete [] dyt; 
    delete [] yt;    
}

/* ------------------------------------------------------------------------ */

/* rkdriver() is an abstraction level that lets different integrators
run using more or less the same input, it initializes some things and
then simply calls the underlying function in a loop; we don't really use
it much, but it was part of the NR approach so it got rolled in here */

void rkdriver(complex<double> * vstart, complex<double> * out, int nvar, double x1,
    double x2, int nstep, void (*derivs)(double, complex<double> *, complex<double> *, void * params), void * params)
{
    int i, k;
    double x, h;
    complex<double> *v, *vout, *dv;

    //Error * ode_error = new Error(MPI_COMM_WORLD);

    v = new complex<double>[nvar];
    vout = new complex<double>[nvar];
    dv = new complex<double>[nvar];

    for (i = 0; i < nvar; i++)
        v[i] = vstart[i];    /* initialize vector */

    x = x1;
    h = (x2-x1)/nstep;

    for (k = 1; k <= nstep; k++)
    {
        (*derivs)(x, v, dv, params);
        rk4(v, dv, nvar, x, h, vout, derivs, params);
        if ((double)(x+h) == x)
        {
            fprintf(stderr, "x: %8.5f\n", x);
            fprintf(stderr, "h: %13.10f\n", h);
            fflush(stderr);

            char err_msg[FLEN]; 

            sprintf(err_msg, "Step size too small in rkdriver\n");

            throw std::runtime_error(err_msg);

            //ode_error->one("Step size too small in rkdriver");

            //fprintf(stderr, "Step size too small in rkdriver\n");
            //exit(EXIT_FAILURE);
        }
        x += h;
        for (i = 0; i < nvar; i++)
            v[i] = vout[i];
    }

    for (i = 0; i < nvar; i++)
        out[i] = vout[i];

    delete [] v;
    delete [] vout;
    delete [] dv;

    //delete ode_error;
}

/* ------------------------------------------------------------------------ */

// Find C = A * B for square complex matrices that are size x size
// Note: takes matrices in linear array form

void mat_mul(complex<double> * C, complex<double> * A, complex<double> * B,
    int size)
{
    int i, j, k;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
        {
            C[i*size + j] = 0.0;
            for (k = 0; k < size; k++)
            {
                C[i*size + j] += A[i*size + k] * B[k*size + j];
            }
        }
    }
}

/* ------------------------------------------------------------------------ */

// map_paths() uses numerical value of the binary string representing each
// system path to generate a mapping of paths to array positions which
// should be unique

void map_paths(map<unsigned long long, unsigned> & pathMap, 
    vector<Path> & pathList)
{
    unsigned long long bin_val;

    for (unsigned path = 0; path < pathList.size(); path++)
    {
        bin_val = get_binary(pathList[path]);

        pathMap[bin_val] = path;
    }
}

/* ------------------------------------------------------------------------ */

// map_single() uses numerical value of the binary string representing each
// system path to generate a mapping of a single path to location in vector

void map_single(map<unsigned long long, unsigned> & pathMap, 
    Path & path, unsigned index)
{
    unsigned long long bin_val = get_binary(path);
    pathMap[bin_val] = index;
}

/* ------------------------------------------------------------------------ */

// get_binary() returns the decimal value of the binary string representing
// a single system path (fwd and bwd)

unsigned long long get_binary(Path & entry)
{
    unsigned long long sum = 0;
    unsigned long long pre = 1;

    // note that ordering below makes fwd path most significant bits
    // also, assumes binary number is in order of normal printing of fwd/bwd
    // i.e. {010,110} -> 010110 -> 22

    if (entry.fwd_path.size() == 0 || entry.bwd_path.size() == 0)
    {
        char err_msg[FLEN];

        sprintf(err_msg, 
            "ERROR: Null fwd/bwd vectors encountered in get_binary()\n");

        throw std::runtime_error(err_msg);
    }

    for (unsigned pos = entry.bwd_path.size() - 1; pos >= 0; pos--)
    {
        sum += entry.bwd_path[pos] * pre;
        pre *= DSTATES;

        if (pos == 0)
            break;
    }

    for (unsigned pos = entry.fwd_path.size() - 1; pos >= 0; pos--)
    {
        sum += entry.fwd_path[pos] * pre;
        pre *= DSTATES;

        if (pos == 0)
            break;
    }

    return sum;
}

/* ------------------------------------------------------------------------ */

// get_binary(vector<unsigned> &, vector<unsigned> &) is an overloaded
// version of the get_binary() function, for use in tensor multiplication

unsigned long long get_binary(vector<unsigned> & fwd, vector<unsigned> & bwd)
{
    unsigned long long sum = 0;
    unsigned long long pre = 1;

    // note that ordering below makes fwd path most significant bits
    // also, assumes binary number is in order of normal printing of fwd/bwd
    // i.e. {010,110} -> 010110 -> 22

    for (unsigned pos = bwd.size() - 1; pos >= 0; pos--)
    {
        sum += bwd[pos] * pre;
        pre *= DSTATES;

        if (pos == 0)
            break;
    }

    for (unsigned pos = fwd.size() - 1; pos >= 0; pos--)
    {
        sum += fwd[pos] * pre;
        pre *= DSTATES;

        if (pos == 0)
            break;
    }

    return sum;
}

/* ------------------------------------------------------------------------ */
