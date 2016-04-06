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

    This code further seeks to improve execution
    time by incorporating filtering ideas into
    the iterative summation. In contrast to
    hbath_filter.cpp, it also deletes filtered
    paths from the path vector, in an attempt to
    save space. 

    This code also employs restart techniques
    as used in iter_qcpi.cpp, in order to run
    safely for long times on large-scale
    computing platforms. */


// NOTE: This program uses the opposite sign convention
//        for the DVR states to other hbath_* programs.
//        However, this should agree with Nancy's convention.

// NOTE: If we take dvr_left = 1.0 and dvr_right = -1.0,
//         then the correctly coordinate-shifted states are 
//         dvr_left = 0.0 and dvr_right = -2.0. Other desired 
//          shifts can be extrapolated from this example.

// NOTE: Need to compile with error.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <list>
#include <stdio.h>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include "error.h"
#include <stdexcept>
#include <sys/stat.h>
#include <gsl/gsl_rng.h>

using namespace std;

// file buffer size
const int FLEN = 1024;

// backup file types
const char state_data_name[] = "coredata";
const char state_data_ext[] = "bin";

// filename prefix to control variation
const char backup_prefix[] = "backup";
const char checkpoint_prefix[] = "checkpoint";
const char restart_prefix[] = "restart";

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

// DVR eigenvals (fixed for now)
const double dvr_left = 1.0;
const double dvr_right = -1.0;

// variables set in config file
double asym; // = 0.5 * kcal_to_hartree;    // system asymmetry
int nmodes; // = 60;                  // number of hbath modes
long steps; // = 50000;               // number of MC steps for IC equil (was 500k).
int ic_tot; // = 30;                  // number of ICs to use

double dt;                              // quantum timestep
int qm_steps; // = 150;                     // number of quantum steps to use
int kmax;                                // memory length

double filter_thresh;                   // filtering threshold

int block_num;                        // number of blocks for MC variance
int step_pts; // = 100;                // number of action integration points per QM timestep
int chunks; // = 5;
int chunksize; // = step_pts/chunks;        // must be integer divisor of step_pts
double rho_dt;
int rho_steps; // = 100;                // number of ODE integration points

double bath_temp; // = 247;                 // Az-B temperature in K
double beta; // = 1.0/(kb*bath_temp);       // reciprocal temperature in a.u.

// RNG flag for repeatability
unsigned long seed; // = 179524;
unsigned long block_seed_spacing;

// Shifting flag for centering W(x,p)
// Set to 1 to turn shifting on, and 0 to turn off
int SHIFT; // = 1;

// Flag to determine numerical/analytical
// trajectory evaluation
int ANALYTICFLAG; // = 0;

// debugging flags
int REPORT; // = 0;
int TFLAG; // = 0;
int RHOPRINT; // = 0;

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
    complex<double> eacp_prod;
    vector<double> x0;
    vector<double> p0;

    // filtering look-ahead
    vector<iter_pair> ic_vec;

    bool active;
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

    bool fixed_ref;          // determines if we used fixed or hopping EACP
    double ref_state;        // what state used for fixed-ref EACP
    Branch branch_state;     // what state to use for iterative branching
    double asym;
    int bath_modes;
    long mc_steps;
    int ic_tot;
    double dt;
    int qm_steps;
    int kmax;
    double filter_thresh;
    int step_pts;
    int chunks;
    int rho_steps;
    int block_num;
    double bath_temp;
    char infile[FLEN];
    char outfile[FLEN];
    char backupname[FLEN];

    // restart parameters

    int ic_index;               // tracks starting IC for restart
    int tstep_index;            // tracks timestep loop number for restart

    bool checkpoint_flg;
    bool plain_dump_flg;
    bool dump_state_flg;
    bool restart_flg;
    bool first_flg;         
    bool break_flg;
    int check_freq;   
    int plain_dump_freq;      
    double time_buffer; 
    double time_budget;  
    double loop_start;          // holds timestamp for loop start
    double loop_time;           // holds time for single loop iteration
    double elapsed_time;        // holds current time spent for execution
    double avg_loop_time;       // running average of loop time
    double alpha; 
};

struct LoadInfo
{
    int ic_index;
    int tstep_index;
};

struct FlagInfo
{
    // debugging flags and RNG seed

    int shift_flag;
    int report_flag;
    int traj_flag;
    int rho_print_flag;
    int analytic_flag;
    unsigned long seed;
    unsigned long block_seed_spacing;
};

struct BathPtr
{
    double * xvals;
    double * pvals;
    double * x_step;
    double * p_step;
};

struct DensityPtr
{
    complex<double> ** rho_proc;
    complex<double> * rho_ic_proc;
    complex<double> ** rho_eacp_proc;

    complex<double> ** rho_curr_checkpoint;
    complex<double> ** rho_full_checkpoint;

    Ref * old_ref_list;
};

// startup and helper functions
void startup(char *, struct SimInfo *, struct FlagInfo *, 
        MPI_Comm);
char * nextword(char *, char **);
void print_header(const char *, const char *, int, FILE *);

// funcs to read file
long get_flines(FILE *);
void get_spec(FILE *, double *, double *);

// func to calc w vals
double bath_setup(double *, double *, double *, long, int);

// funcs to perform MC walk
void calibrate_mc(double *, double *, double *, double *, gsl_rng *);
double ic_gen(double *, double *, double *, double *, double *, double *, gsl_rng *);
double dist(double, double, double, double, double, double);

// propagator integration
void ho_update(Propagator &, Mode *, double);
void ho_update_exact(Propagator &, Mode *, double);
void build_ham(Propagator &, Mode *, int);
void build_ham_exact(Propagator &, Mode *);
void prop_eqns(double, complex<double> *, complex<double> *, void *);
void rk4(complex<double> * y, complex<double> * dydx, int n, double x, double h, 
    complex<double> * yout, void (*derivs)(double, complex<double> *, complex<double> *, void * params), void * params);
void rkdriver(complex<double> * vstart, complex<double> * out, int nvar, double x1,
    double x2, int nstep, void (*derivs)(double, complex<double> *, complex<double> *, void * params), void * params);

// QCPI functions
void qcpi_update(Path &, Mode *);
void qcpi_update_exact(Path &, Mode *);
double action_calc(Path &, Mode *, Mode *);
double action_calc_exact(Path &, Mode *, Mode *);

// simple matrix multiply
void mat_mul(complex<double> *, complex<double> *, complex<double> *, int);

// mapping functions
void map_paths(map<unsigned long long, unsigned> &, 
    vector<Path> &);
void map_single(map<unsigned long long, unsigned> &, 
    Path &, unsigned);
unsigned long long get_binary(Path &);
unsigned long long get_binary(vector<unsigned> &, vector<unsigned> &);

// plaintext checkpoint function

void plaintext_out(complex<double> **, complex<double> **,
    SimInfo &, int, int, MPI_Comm); 

// functions to save state
void save_state(BathPtr &, gsl_rng *, Propagator &, DensityPtr &,
    vector<Path> &, Mode *, SimInfo &, FlagInfo &, char *, int);
void write_bath(BathPtr &, SimInfo &, FILE *);
void write_rng(gsl_rng *, FILE *);
void write_prop(Propagator &, FILE *);
void write_rho(DensityPtr &, SimInfo &, FILE *);
void write_paths(vector<Path> &, FILE *);
void write_modes(Mode *, SimInfo &, FILE *);
void write_prog(SimInfo &, FlagInfo &, FILE *);
bool file_exists(char * filename);

// functions to load state
void load_state(BathPtr &, gsl_rng *, Propagator &, DensityPtr &, 
    vector<Path> &, Mode *, SimInfo &, int);
void read_backup(BathPtr &, gsl_rng *, Propagator &, DensityPtr &, 
    vector<Path> &, Mode *, SimInfo &, const char *, int);
void read_bath(BathPtr &, FILE *);
void read_rng(gsl_rng *, FILE *);
void read_prop(Propagator &, FILE *);
void read_rho(DensityPtr &, SimInfo &, FILE *);
void read_paths(vector<Path> &, FILE *);
void read_modes(Mode *, SimInfo &, FILE *);
void read_prog(SimInfo &, FILE *);

int main(int argc, char * argv[])
{
    // initialize MPI

    MPI_Init(&argc, &argv);
    MPI_Comm w_comm = MPI_COMM_WORLD;

    int me, nprocs;

    MPI_Comm_rank(w_comm, &me);
    MPI_Comm_size(w_comm, &nprocs);

    Error * error = new Error(w_comm);

    // process arguments, format should be:
    //      mpirun -n <proc_num> ./prog_name <config_file> [restart]

    if (argc < 2 || argc > 3)
    {
        char errstr[FLEN];
        sprintf(errstr, "Usage: %s <config_file> [restart]", argv[0]);
        error->all(errstr);
    }

    char config_file[FLEN];   

    sprintf(config_file, "%s", argv[1]);

    // create structs to hold configuration parameters

    struct SimInfo simData;
    struct FlagInfo flagData;

    // assign values to config. vars

    startup(config_file, &simData, &flagData, w_comm);

    // check for restart state

    if (argc == 3)
    {
        if (strcmp(argv[2], "restart") == 0)
            simData.restart_flg = true;
        else
        {
            char errstr[FLEN];
            sprintf(errstr, "Unrecognized command line parameter: %s",
                argv[2]);

            error->all(errstr);
        }
    }

    // set global parameters from startup() output

    asym = simData.asym * kcal_to_hartree;
    nmodes = simData.bath_modes;
    steps = simData.mc_steps;
    ic_tot = simData.ic_tot;

    dt = simData.dt;
    qm_steps = simData.qm_steps;
    kmax = simData.kmax;

    filter_thresh = simData.filter_thresh;

    step_pts = simData.step_pts;
    chunks = simData.chunks;
    chunksize = step_pts/chunks;
    rho_steps = simData.rho_steps;
    block_num = simData.block_num;

    bath_temp = simData.bath_temp;
    beta = 1.0/(kb*bath_temp);

    seed = flagData.seed;
    block_seed_spacing = flagData.block_seed_spacing;

    SHIFT = flagData.shift_flag;
    REPORT = flagData.report_flag;
    TFLAG = flagData.traj_flag;
    ANALYTICFLAG = flagData.analytic_flag;
    RHOPRINT = flagData.rho_print_flag;

    // set some index data manually

    char tmpname[FLEN];

    sprintf(tmpname, "%s", simData.outfile);

    char * tmpfile = strtok(tmpname, ".");

    sprintf(simData.backupname, "%s", tmpfile);

    simData.ic_index = 0;
    simData.tstep_index = kmax;
    simData.avg_loop_time = 0.0;
    simData.first_flg = true;
    simData.break_flg = false;

    // sanity checks

    if (kmax <= 0)
        error->all("kmax must be positive");

    if (dt <= 0)
        error->all("Quantum timestep must be positive");

    if (filter_thresh < 0.0)
        error->all("Filtering threshold cannot be negative");

    if (qm_steps < kmax)
        error->all("Quantum step number smaller than kmax");

    if (ic_tot < nprocs)
        error->all("Too few ICs for processor number");

    if ((nprocs % block_num) != 0)
        error->all("Block number must evenly divide processor number");

    // make sure MC run is long enough

    if (steps < mc_buff * nmodes)
        steps = mc_buff * nmodes;

    // open files for I/O

    FILE * infile;
    FILE * outfile;
    FILE * trajfile;

    infile = fopen(simData.infile, "r");

    if (infile == NULL)
    {
        char errstr[FLEN];
        sprintf(errstr, "Could not open file %s",
            simData.infile);
        error->one(errstr);
    }

    // only open output file on I/O proc

    if (me == 0)
    {
        outfile = fopen(simData.outfile, "w");

        if (outfile == NULL)
        {
            char errstr[FLEN];
            sprintf(errstr, "Could not open file %s",
                simData.outfile);
            error->one(errstr);
        }
    }

    // open up trajfile, if needed

    if (TFLAG)
    {   
        char trajname[FLEN];

        sprintf(trajname, "traj_%s", simData.outfile);

        trajfile = fopen(trajname, "w");

        if (trajfile == NULL)
        {
            char errstr[FLEN];
            sprintf(errstr, "Could not open file %s",
                trajname);
            error->one(errstr);
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

    double * bath_freq = new double [nmodes];
    double * bath_coup = new double [nmodes];

    double w0 = bath_setup(omega, jvals, bath_freq, npoints, nmodes);

    // calculate couplings from frequencies

    double pi = acos(-1.0);

    for (int i = 0; i < nmodes; i++)
        bath_coup[i] = sqrt(2.0*w0/pi) * bath_freq[i];

    // write out frequency, coupling pairs if
    // REPORT flag is turned on

    if (me == 0 && REPORT)
    {
        fprintf(stdout, "\n\n");

        for (int i = 0; i < nmodes; i++)
            fprintf(stdout, "%.10e %.10e\n", bath_freq[i],
                bath_coup[i]);

        fprintf(stdout, "\n");
    }

    // divide up ICs across procs
    // if IC num evenly divides, we just portion out
    // a block; otherwise we give remainder out block-cyclically
    // with blocksize 1
    
    int base_share = ic_tot/nprocs;
    int remain = ic_tot % nprocs;
    int my_ics;

    if (ic_tot < nprocs)
        error->all("Too few ICs for processor group");

    if (remain == 0)
        my_ics = base_share;
    else
    {
        if (me < remain)
            my_ics = base_share+1;
        else
            my_ics = base_share;
    }

    double * xvals = new double [nmodes];
    double * pvals = new double [nmodes];

    double * x_step = new double [nmodes];
    double * p_step = new double [nmodes];

    // split global comm into blocked groups

    MPI_Comm block_comm;
    int block_id = me % block_num;

    MPI_Comm_split(w_comm, block_id, me, &block_comm);

    int block_size;
    int me_block;

    MPI_Comm_rank(block_comm, &me_block);
    MPI_Comm_size(block_comm, &block_size);

    // split procs based on rank in blocks

    MPI_Comm var_comm;

    MPI_Comm_split(w_comm, me_block, me, &var_comm);

    // initialize RNG; can change seed to change trajectory behavior

    gsl_rng * gen = gsl_rng_alloc(gsl_rng_mt19937);
    unsigned long s_val = seed * (me+1) + (block_id * block_seed_spacing); 
   
    // ensure rng seed is not zero
 
    if (s_val == 0)
        s_val = seed + (me+1);

    gsl_rng_set(gen, s_val);

    // run test MC trials to determine optimal step sizes

    calibrate_mc(x_step, p_step, bath_freq, bath_coup, gen);

    // initialize IC arrays

    for (int i = 0; i < nmodes; i++)
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


/*
    complex<double> * old_prop = new complex<double> [DSTATES*DSTATES];
    
    // define propagator as generalized bare TLS

    double w_new = sqrt(asym*asym + tls_freq*tls_freq);

    old_prop[0] = cos(w_new*dt) - I*(asym/w_new)*sin(w_new*dt);
    old_prop[3] = conj(old_prop[0]);
    old_prop[1] = old_prop[2] = I*tls_freq*sin(w_new*dt)/w_new;
*/

    // define ODE timestep

    rho_dt = dt/chunks;

    // create propagator object

    Propagator curr_prop;

    curr_prop.prop = new complex<double> [DSTATES*DSTATES];
    curr_prop.ptemp = new complex<double> [DSTATES*DSTATES];
    curr_prop.ham = new complex<double> [DSTATES*DSTATES];

    // allocate density matrix

    complex<double> ** rho_proc = new complex<double> * [qm_steps];
    complex<double> * rho_ic_proc = new complex<double> [qm_steps];
    complex<double> ** rho_eacp_proc = new complex<double> * [qm_steps];

    // allocated matrices for rho(t) checkpointing

    complex<double> ** rho_curr_checkpoint = new complex<double> * [qm_steps];
    complex<double> ** rho_full_checkpoint = new complex<double> * [qm_steps];

    for (int i = 0; i < qm_steps; i++)
    {
        rho_proc[i] = new complex<double> [DSTATES*DSTATES];
        rho_eacp_proc[i] = new complex<double> [DSTATES*DSTATES];

        rho_curr_checkpoint[i] = new complex<double> [DSTATES*DSTATES];
        rho_full_checkpoint[i] = new complex<double> [DSTATES*DSTATES];
        
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_proc[i][j] = 0.0;
            rho_eacp_proc[i][j] = 0.0;

            rho_curr_checkpoint[i][j] = 0.0;
            rho_full_checkpoint[i][j] = 0.0;
        }
    }

    // initialize path vector

    vector<Path> pathList;
    vector<Path> tempList;

    Path pstart;

    // set up variables to store current and past reference state

    Ref * old_ref_list = new Ref [qm_steps];

    old_ref_list[0] = REF_LEFT;

    // initialize harmonic bath arrays

    Mode * modes = new Mode [nmodes];
    Mode * ref_modes = new Mode [nmodes];

    for (int i = 0; i < nmodes; i++)
    {
        modes[i].omega = bath_freq[i];
        modes[i].c = bath_coup[i];

        ref_modes[i].omega = bath_freq[i];
        ref_modes[i].c = bath_coup[i];
    }

    map<unsigned long long, unsigned> pathMap;

    // set up variable to store current reference state

    double ho_ref_state = dvr_left;

    // check for restart state
    
    BathPtr bath_pointers;
    DensityPtr density_pointers;

    if (simData.restart_flg)
    {
        // set pointers

        bath_pointers.xvals = xvals;
        bath_pointers.pvals = pvals;
        bath_pointers.x_step = x_step;
        bath_pointers.p_step = p_step;

        density_pointers.rho_proc = rho_proc;
        density_pointers.rho_ic_proc = rho_ic_proc;
        density_pointers.rho_eacp_proc = rho_eacp_proc;
        density_pointers.rho_curr_checkpoint = rho_curr_checkpoint;
        density_pointers.rho_full_checkpoint = rho_full_checkpoint;
        density_pointers.old_ref_list = old_ref_list;

        // load restart state from file

        load_state(bath_pointers, gen, curr_prop, density_pointers, 
            pathList, modes, simData, me);

        // reset ref_modes values to ensure consistency
    
        for (int i = 0; i < nmodes; i++)
        {
            ref_modes[i].omega = modes[i].omega;
            ref_modes[i].c = modes[i].c;
        }

    }

    LoadInfo loadData;

    loadData.ic_index = simData.ic_index;
    loadData.tstep_index = simData.tstep_index;

    // Following loops are the core computational ones

    // Outer loop goes over initial conditions (selected by MC runs)

    // Inner loop goes over time points - for times less than
    // kmax, this does a full path sum, and for other points,
    // the iterative tensor propagation scheme is used

    for (int ic_curr = loadData.ic_index; ic_curr < my_ics; ic_curr++)
    {
      // only run t < kmax and build paths if we're not restarting
      
      if (!simData.restart_flg)
      {

        // zero per-proc rho(t)

        for (int i = 0; i < qm_steps; i++)
        {
            rho_ic_proc[i] = 0.0;

            for (int j = 0; j < DSTATES*DSTATES; j++)
                rho_curr_checkpoint[i][j] = 0.0;
        }


        // generate ICs for HOs using MC walk (we use last step's
        // ICs as current step's IC seed)

        double ratio = ic_gen(xvals, pvals, bath_freq, bath_coup, x_step, p_step, gen);

        // write trajectory to separate file for record

        if (TFLAG)
        {
            fprintf(trajfile, "IC set %d (acc. ratio %f):\n\n", ic_curr, ratio);

            for (int i = 0; i < nmodes; i++)
                fprintf(trajfile, "%.10f %.10f\n", xvals[i], pvals[i]);

            fprintf(trajfile, "\n");  
        }       

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
        pstart.eacp_prod = 1.0;
        pstart.x0.assign(xvals, xvals+nmodes);
        pstart.p0.assign(pvals, pvals+nmodes);

        pstart.active = true;

        pathList.push_back(pstart);

        // initialize propagator ICs

        curr_prop.x0_free.assign(xvals, xvals+nmodes);
        curr_prop.p0_free.assign(pvals, pvals+nmodes);

        // loop over first kmax time points

        for (int seg = 0; seg < kmax; seg++)
        {
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
            // note that ho_update clears ref_modes x(t) and p(t) list

            //ho_update(curr_prop, ref_modes, ho_ref_state);

            if (ANALYTICFLAG > 0)
                ho_update_exact(curr_prop, ref_modes, ho_ref_state);
            else
                ho_update(curr_prop, ref_modes, ho_ref_state);

            // chunk trajectory into pieces for greater
            // accuracy in integrating U(t)
/*
            for (int chunk = chunksize-1; chunk < step_pts; chunk += chunksize)
            {
                // construct H(x,p) from bath configuration
            
                build_ham(curr_prop, ref_modes, chunk);

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
*/

            if (ANALYTICFLAG > 0)
            {
                // chunk trajectory into pieces for greater
                // accuracy in integrating U(t)

                for (int chunk_num = 0; chunk_num < chunks; chunk_num++)
                {
                    // construct H(x,p) from bath configuration
            
                    build_ham(curr_prop, ref_modes, chunk_num);

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
            }
            else
            {
                // chunk trajectory into pieces for greater
                // accuracy in integrating U(t)

                for (int chunk = chunksize-1; chunk < step_pts; chunk += chunksize)
                {
                    // construct H(x,p) from bath configuration
            
                    build_ham(curr_prop, ref_modes, chunk);

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

            } // end analytic clause for U(t) integration


            // loop over all paths at this time point

            for (unsigned path = 0; path < pathList.size(); path++)
            { 
                // calculate x(t) and p(t) at integration points
                // along all paths

                //qcpi_update(pathList[path], modes);

                if (ANALYTICFLAG > 0)
                    qcpi_update_exact(pathList[path], modes);
                else
                    qcpi_update(pathList[path], modes);

                // use integration points to find new phase contribution

                //double phi = action_calc(pathList[path], modes, ref_modes);

                double phi;

                if (ANALYTICFLAG > 0)
                    phi = action_calc_exact(pathList[path], modes, ref_modes);
                else
                    phi = action_calc(pathList[path], modes, ref_modes);

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

                pathList[path].eacp_prod *= curr_prop.prop[findex] *
                    conj(curr_prop.prop[bindex]);

                // pull out density matrix at each time point

                unsigned rindex = splus1*DSTATES + sminus1;

                rho_proc[seg][rindex] += pathList[path].product;
                rho_eacp_proc[seg][rindex] += pathList[path].eacp_prod;

                rho_curr_checkpoint[seg][rindex] += pathList[path].product;

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

            // remove paths below thresh

            unsigned del_num = 0;

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                if (abs(pathList[path].product) < filter_thresh)
                {
                    pathList[path].active = false;
                    del_num++;
                }
            }

            tempList.clear();

            // only delete if we find sub-thresh paths

            if (del_num > 0)
            {
                // reserve space to avoid wasted allocations

                tempList.reserve(pathList.size() - del_num);

                for (unsigned path = 0; path < pathList.size(); path++)
                {
                    Path tpath;

                    // copy over non-deleted paths

                    if (pathList[path].active)
                    {
                        tpath = pathList[path];
                        tempList.push_back(tpath);
                    }
                }

                pathList.swap(tempList);
                tempList.clear();
            }

            // !! BEGIN DEBUG !!
/*
            if (me == 0)
            {
                fprintf(stdout, "\nSeg %d, path list size: %zu\n",
                    seg, pathList.size() );
            }
*/
            // !! END DEBUG !!
    
        } // end seg loop (full path phase)

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

      } // end restart if clause

        // map paths to vector location

        map_paths(pathMap, pathList);

        // loop over time points beyond kmax
        // and propagate system iteratively

        for (int seg = loadData.tstep_index; seg < qm_steps; seg++)
        {
            if (simData.dump_state_flg)
            {
                // get loop starting time

                simData.loop_start = MPI_Wtime();
            }

            // choose branch to propagate on stochastically,
            // based on state of system at start of memory span

            unsigned fRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));
            unsigned bRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));

            // using stochastically determined branching
            // and harmonic reference states

            if (simData.fixed_ref)
            {
                ho_ref_state = simData.ref_state;
                
                if (simData.branch_state == BRANCH_LEFT)
                    fRand = bRand = 0;
                else if (simData.branch_state == BRANCH_MID)
                {
                    // following assumes DSTATES = 2

                    fRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));
                    bRand = 1 - fRand;
                }
                else
                    fRand = bRand = 1;
            }
            else
            {
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

/*
                if (xi < rho_ic_proc[seg-1].real() )    
                {
                    fRand = bRand = 0;
                    ho_ref_state = dvr_left;
                }
                else
                {
                    fRand = bRand = 1;
                    ho_ref_state = dvr_right;
                }
*/

            } // end EACP reference choice clause

/*
            if (seg < 2*kmax)
            {
                // update as equil. reference branch

                if (SHIFT)
                {
                    // assumes bath centered around left state

                    fRand = bRand = 0;
                }
                else
                {
                    // averages over (0,1) and (1,0) states, since
                    // in this case bath center is between the two

                    fRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,2));
                    bRand = 1 - fRand;
                }
            }
            else
            {
                // update according to lagged rho(t)

                double xi = gsl_rng_uniform(gen);

                if (xi < rho_ic_proc[seg-kmax].real() )    
                {
                    fRand = bRand = 0;
                }
                else
                {
                    fRand = bRand = 1;
                }
            }
*/

/*
            // TODO: eventually change EACP reference here as well!

            double xi = gsl_rng_uniform(gen);

            if (xi < rho_ic_proc[seg-kmax].real() )
                fRand = bRand = 0;
            else
                fRand = bRand = 1;
*/
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

            //ho_update(curr_prop, ref_modes, ho_ref_state);

            if (ANALYTICFLAG > 0)
                ho_update_exact(curr_prop, ref_modes, ho_ref_state);
            else
                ho_update(curr_prop, ref_modes, ho_ref_state);


/*
            for (int chunk = chunksize-1; chunk < step_pts; chunk += chunksize)
            {
                // construct H(x,p) from bath configuration
            
                build_ham(curr_prop, ref_modes, chunk);

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
*/

            if (ANALYTICFLAG > 0)
            {
                // chunk trajectory into pieces for greater
                // accuracy in integrating U(t)

                for (int chunk_num = 0; chunk_num < chunks; chunk_num++)
                {
                    // construct H(x,p) from bath configuration
            
                    build_ham(curr_prop, ref_modes, chunk_num);

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
            }
            else
            {
                // chunk trajectory into pieces for greater
                // accuracy in integrating U(t)

                for (int chunk = chunksize-1; chunk < step_pts; chunk += chunksize)
                {
                    // construct H(x,p) from bath configuration
            
                    build_ham(curr_prop, ref_modes, chunk);

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

            } // end analytic clause for U(t) integration

            // turn off paths below thresh

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                if (abs(pathList[path].product) < filter_thresh)
                    pathList[path].active = false;
            }

            // !! BEGIN DEBUG !!
/*
            if (me == 0)
            {
                fprintf(stdout, "In step %d, paths after filter: %zu\n",
                    seg, pathList.size() );
            }
*/
            // !! END DEBUG !!

            // do initial loop over paths to set up dependencies

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                // first make sure contributing path is present

                vector<unsigned> fwd_temp, bwd_temp;

                fwd_temp.assign(pathList[path].fwd_path.begin(),
                    pathList[path].fwd_path.end() - 1);

                bwd_temp.assign(pathList[path].bwd_path.begin(),
                    pathList[path].bwd_path.end() - 1);

                fwd_temp.insert(fwd_temp.begin(), fRand);
                bwd_temp.insert(bwd_temp.begin(), bRand);

                unsigned long long index = get_binary(fwd_temp, bwd_temp);

                if (pathMap.count(index) == 0)
                {
                    // IC contributor was filtered, check other candidates

                    bool found_flg = false;

                    for (int i = 0; i < DSTATES; i++)
                    {
                        for (int j = 0; j < DSTATES; j++)
                        {
                            fwd_temp[0] = i;
                            bwd_temp[0] = j;

                            index = get_binary(fwd_temp, bwd_temp);

                            // if we find viable path, get ICs there instead

                            if (pathMap.count(index) > 0)
                            {
                                found_flg = true;

                                unsigned donor_id = pathList[path].fwd_path.back() * DSTATES +
                                    pathList[path].bwd_path.back();

                                unsigned location = pathMap[index];

                                unsigned long long curr_key = 
                                    get_binary(pathList[path].fwd_path, pathList[path].bwd_path);       

                                // Use key rather than path below, since path changes w/ deletion?

                                pathList[location].ic_vec.push_back(iter_pair(donor_id,curr_key));

                            } // end dependency update clause

                            if (found_flg)
                                break;

                        } // end j (bwd) loop
    
                        if (found_flg)
                            break;

                    } // end i (fwd) loop

                    // if we can't find candidate turn off path

                    if (!found_flg)
                        pathList[path].active = false;

                } // end deleted search clause

                else if (!pathList[pathMap[index]].active)
                {
                    // IC contributor was turned off, check other candidates

                    bool found_flg = false;

                    for (int i = 0; i < DSTATES; i++)
                    {
                        for (int j = 0; j < DSTATES; j++)
                        {
                            fwd_temp[0] = i;
                            bwd_temp[0] = j;

                            index = get_binary(fwd_temp, bwd_temp);

                            unsigned target = pathMap[index];

                            // if we find active path, get ICs there instead

                            if (pathList[target].active)
                            {
                                found_flg = true;

                                unsigned donor_id = pathList[path].fwd_path.back() * DSTATES +
                                    pathList[path].bwd_path.back();

                                unsigned long long curr_key = 
                                    get_binary(pathList[path].fwd_path, pathList[path].bwd_path);       

                                // Use key rather than path below, since path changes w/ deletion?

                                pathList[target].ic_vec.push_back(iter_pair(donor_id,curr_key));

                            } // end dependency update clause

                            if (found_flg)
                                break;

                        } // end j (bwd) loop
    
                        if (found_flg)
                            break;

                    } // end i (fwd) loop

                    // if we can't find candidate turn off path

                    if (!found_flg)
                        pathList[path].active = false;

                } // end turned-off search clause

            } // end dependency path loop

            // !! BEGIN DEBUG !!
/*
            if (me == 0)
            {
                fprintf(stdout, "In step %d, paths after dependencies: %zu\n",
                    seg, pathList.size() );
            }
*/
            // !! END DEBUG !!

            // set up tempList to hold our matrix mult.
            // intermediates

            tempList.clear();
            tempList = pathList;

            for (unsigned tp = 0; tp < tempList.size(); tp++)
            {
                tempList[tp].product = 0.0;
                tempList[tp].eacp_prod = 0.0;

                // erase copied ic_vec so we don't double-count

                tempList[tp].ic_vec.clear();
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

              if (pathList[path].active)
              {

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

                        //qcpi_update(temp, modes);

                        if (ANALYTICFLAG > 0)
                            qcpi_update_exact(temp, modes);
                        else
                            qcpi_update(temp, modes);

                        // use integration points to find new phase contribution

                        //double phi = action_calc(temp, modes, ref_modes);

                        double phi;

                        if (ANALYTICFLAG > 0)
                            phi = action_calc_exact(temp, modes, ref_modes);
                        else
                            phi = action_calc(temp, modes, ref_modes);

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

                        if (pathMap.count(target) == 0) // path deleted
                        {
                            // resurrect deleted path

                            Path resPath;
                            
                            resPath.fwd_path.assign(ftemp.begin(), ftemp.end());
                            resPath.bwd_path.assign(btemp.begin(), btemp.end());

                            resPath.product = tensorProd * temp.product;
                            resPath.eacp_prod = tensorEacp * temp.eacp_prod;

                            resPath.active = true;

                            // update ICs since deactivation
    
                            resPath.x0.assign(temp.x0.begin(), temp.x0.end());
                            resPath.p0.assign(temp.p0.begin(), temp.p0.end());

                            // push onto path lists

                            pathList.push_back(resPath);
                            tempList.push_back(resPath);

                            // update map

                            unsigned index = pathList.size() - 1;

                            map_single(pathMap, resPath, index);
                        }
                        else if (!tempList[pathMap[target]].active) // in list, but filtered
                        {
                            unsigned outPath = pathMap[target];

                            // reactivate turned-off path

                            tempList[outPath].active = true;

                            tempList[outPath].product = tensorProd * temp.product;
                            tempList[outPath].eacp_prod = tensorEacp * temp.eacp_prod;

                            // update ICs since deactivation
    
                            tempList[outPath].x0.assign(temp.x0.begin(), temp.x0.end());
                            tempList[outPath].p0.assign(temp.p0.begin(), temp.p0.end());
                        }
                        else
                        {
                            unsigned outPath = pathMap[target];

                            // path is already present in list, so update it

                            tempList[outPath].product += tensorProd * temp.product;
                            tempList[outPath].eacp_prod += tensorEacp * temp.eacp_prod;
        
                            // update ICs if we have correct donor element

                            if (temp.fwd_path[0] == fRand && temp.bwd_path[0] == bRand)
                            {
                                tempList[outPath].x0.assign(temp.x0.begin(), temp.x0.end());
                                tempList[outPath].p0.assign(temp.p0.begin(), temp.p0.end());
                            }

                        } // end sum/reactivation clause

                        // also copy ICs over to elements in ic_vec

                        unsigned long long target_key;

                        for (unsigned ind = 0; ind < temp.ic_vec.size(); ind++) 
                        {
                            // donor_id tracks which extension belongs where
                        
                            unsigned donor_id = fwd * DSTATES + bwd;

                            if (donor_id == temp.ic_vec[ind].first)
                            {
                                unsigned path_loc;

                                target_key = temp.ic_vec[ind].second;

                                path_loc = pathMap[target_key];

                                tempList[path_loc].x0.assign(temp.x0.begin(), 
                                    temp.x0.end());

                                tempList[path_loc].p0.assign(temp.p0.begin(), 
                                    temp.p0.end());
                            }
                        }

                        // pop off previous update (returning length to kmax)

                        temp.fwd_path.pop_back();
                        temp.bwd_path.pop_back();

                    } // end bwd loop

                } // end fwd loop

              } // end filter check loop
                
            } // end path loop (iter. phase)    

            // swap tempList and pathList

            pathList.swap(tempList);

            // just to make sure, clear out ic_vec

            for (unsigned pos = 0; pos < pathList.size(); pos++)
                pathList[pos].ic_vec.clear();


            // !! BEGIN DEBUG !!
/*
            unsigned filtered_paths = 0;
    
            for (unsigned path = 0; path < pathList.size(); path++)
            {
                if (!pathList[path].active)
                    filtered_paths++;
            }

            if (me == 0 && (seg % 10) == 0)
            {
                fprintf(stdout, "In step %d, paths after iteration (of %lu): %lu\n",
                    seg, pathList.size(), pathList.size() - filtered_paths);
            }
*/
            // !! END DEBUG !!


            // pull out current density matrix

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                unsigned size = pathList[path].fwd_path.size();
                unsigned splus1 = pathList[path].fwd_path[size-1];
                unsigned sminus1 = pathList[path].bwd_path[size-1];

                unsigned rindex = splus1*DSTATES + sminus1;

                // only update if path is active

                if (pathList[path].active)
                {
                    rho_proc[seg][rindex] += pathList[path].product;
                    rho_eacp_proc[seg][rindex] += pathList[path].eacp_prod;

                    rho_curr_checkpoint[seg][rindex] += pathList[path].product;

                    if (rindex == 0)
                        rho_ic_proc[seg] += pathList[path].product;
                }

            } // end update loop (iter. phase)

            // reset all paths to active status

            for (unsigned path = 0; path < pathList.size(); path++)
                pathList[path].active = true;

            // check for state-saving directions

            if (simData.dump_state_flg)
            {
                // find loop time

                simData.loop_time = MPI_Wtime() - simData.loop_start;

                // update exponential moving avg

                if (simData.first_flg)
                {
                    // set avg to first time value if undefined

                    simData.avg_loop_time = simData.loop_time;
                    simData.first_flg = false;
                }
                else
                {
                    simData.avg_loop_time = 
                        (simData.alpha * simData.loop_time) +
                            (1.0 - simData.alpha) * simData.avg_loop_time;
                }

                // find total elapsed time to this point

                simData.elapsed_time = MPI_Wtime() - start;

                // check our time budget

                double proj_time = simData.elapsed_time + 
                    simData.avg_loop_time + simData.time_buffer;

                // write restart if we estimate going over time

                if (proj_time > simData.time_budget)
                {                   
                    // store loop indices so we restart correctly

                    simData.ic_index = ic_curr;
                    simData.tstep_index = seg + 1;

                    // write out restart files

                    char type_name[FLEN];

                    sprintf(type_name, "%s_%s", restart_prefix, 
                        simData.backupname);

                    // copy over arrays to structures

                    bath_pointers.xvals = xvals;
                    bath_pointers.pvals = pvals;
                    bath_pointers.x_step = x_step;
                    bath_pointers.p_step = p_step;

                    density_pointers.rho_proc = rho_proc;
                    density_pointers.rho_ic_proc = rho_ic_proc;
                    density_pointers.rho_eacp_proc = rho_eacp_proc;
                    density_pointers.rho_curr_checkpoint = rho_curr_checkpoint;
                    density_pointers.rho_full_checkpoint = rho_full_checkpoint;
                    density_pointers.old_ref_list = old_ref_list;

                    // write out program data

                    save_state(bath_pointers, gen, curr_prop, density_pointers,
                        pathList, modes, simData, flagData, type_name, me);

                    // block until all procs have written to file

                    MPI_Barrier(w_comm);  
    
                    // write confirmation message to outfile

                    if (me == 0)
                    {
                        fprintf(outfile, 
                            "Restart data written successfully\n\n");
                    }

                    // set flag to indicate restart write

                    simData.break_flg = true;

                    // break out of loop
        
                    break;

                } // end save state clause

            } // end dump_state clause

            // break if flag is set

            if (simData.break_flg)
                break;
            
            if (simData.checkpoint_flg)
            {
                // write out data if at checkpoint

                if ((seg % simData.check_freq) == 0)
                {             
                    // increment loop index so we restart correctly

                    simData.ic_index = ic_curr;
                    simData.tstep_index = seg + 1;

                    // write plaintext summary file

                    plaintext_out(rho_curr_checkpoint, rho_full_checkpoint,
                        simData, ic_curr, seg, w_comm); 

                    // write out checkpoint files
            
                    char type_name[FLEN];

                    sprintf(type_name, "%s_%s", checkpoint_prefix, 
                        simData.backupname);

                    // copy over arrays to structures

                    bath_pointers.xvals = xvals;
                    bath_pointers.pvals = pvals;
                    bath_pointers.x_step = x_step;
                    bath_pointers.p_step = p_step;

                    density_pointers.rho_proc = rho_proc;
                    density_pointers.rho_ic_proc = rho_ic_proc;
                    density_pointers.rho_eacp_proc = rho_eacp_proc;
                    density_pointers.rho_curr_checkpoint = rho_curr_checkpoint;
                    density_pointers.rho_full_checkpoint = rho_full_checkpoint;
                    density_pointers.old_ref_list = old_ref_list;

                    // write out program data

                    save_state(bath_pointers, gen, curr_prop, density_pointers,
                        pathList, modes, simData, flagData, type_name, me);
                }

            } // end checkpoint clause

            if (simData.plain_dump_flg)
            {
                if ((seg % simData.plain_dump_freq) == 0)
                {
                    // write plaintext summary file

                    plaintext_out(rho_curr_checkpoint, rho_full_checkpoint,
                        simData, ic_curr, seg, w_comm); 
                }

            } // end plaintext dump clause

        } // end seg loop (iter. phase)

        // break if flag is set

        if (simData.break_flg)
            break;

        // turn off restart flag after first new IC

        if (simData.restart_flg)
        {
            simData.restart_flg = false;
            loadData.tstep_index = simData.kmax;
        }

        // copy current run out to accumulated IC matrix

        for (int i = 0; i < qm_steps; i++)
        {
            for (int j = 0; j < DSTATES*DSTATES; j++)
            {
                rho_full_checkpoint[i][j] += rho_curr_checkpoint[i][j];
            }
        }

    } // end IC loop

    // collect timing data

    double local_time = MPI_Wtime() - start;
    double g_runtime = 0.0;

    MPI_Allreduce(&local_time, &g_runtime, 1, MPI_DOUBLE,
        MPI_MAX, w_comm);

    // collect real and imag parts of rho into separate
    // arrays for MPI communication

    double * rho_real_proc = new double [qm_steps*DSTATES*DSTATES];
    double * rho_imag_proc = new double [qm_steps*DSTATES*DSTATES];

    double * rho_real = new double [qm_steps*DSTATES*DSTATES];
    double * rho_imag = new double [qm_steps*DSTATES*DSTATES];

    for (int i = 0; i < qm_steps; i++)
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

    MPI_Allreduce(rho_real_proc, rho_real, qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    MPI_Allreduce(rho_imag_proc, rho_imag, qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    // scale arrays by Monte Carlo factor

    for (int i = 0; i < qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real[i*DSTATES*DSTATES+j] /= ic_tot;
            rho_imag[i*DSTATES*DSTATES+j] /= ic_tot;
        }
    }

    // collect real and imag parts of EACP rho into separate
    // arrays for MPI communication

    double * eacp_real_proc = new double [qm_steps*DSTATES*DSTATES];
    double * eacp_imag_proc = new double [qm_steps*DSTATES*DSTATES];

    double * eacp_real = new double [qm_steps*DSTATES*DSTATES];
    double * eacp_imag = new double [qm_steps*DSTATES*DSTATES];

    for (int i = 0; i < qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            eacp_real_proc[i*DSTATES*DSTATES+j] = rho_eacp_proc[i][j].real();
            eacp_imag_proc[i*DSTATES*DSTATES+j] = rho_eacp_proc[i][j].imag();

            eacp_real[i*DSTATES*DSTATES+j] = 0.0;
            eacp_imag[i*DSTATES*DSTATES+j] = 0.0;
        }
    }

    // Allreduce the real and imaginary arrays

    MPI_Allreduce(eacp_real_proc, eacp_real, qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    MPI_Allreduce(eacp_imag_proc, eacp_imag, qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    // scale arrays by Monte Carlo factor

    for (int i = 0; i < qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            eacp_real[i*DSTATES*DSTATES+j] /= ic_tot;
            eacp_imag[i*DSTATES*DSTATES+j] /= ic_tot;
        }
    }

    // calculate block averages and variance

    double * rho_real_block = new double [qm_steps*DSTATES*DSTATES];
    double * eacp_real_block = new double [qm_steps*DSTATES*DSTATES];

    double * rho_var = new double [qm_steps];
    double * eacp_var = new double [qm_steps];

    double * rho_var_block = new double [qm_steps];
    double * eacp_var_block = new double [qm_steps];

    // initialize to zero
    
    for (int i = 0; i < qm_steps; i++)
    {
        rho_var[i] = 0.0;
        eacp_var[i] = 0.0;

        rho_var_block[i] = 0.0;
        eacp_var_block[i] = 0.0;

        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real_block[i*DSTATES*DSTATES+j] = 0.0;
            eacp_real_block[i*DSTATES*DSTATES+j] = 0.0;
        }
    }

    // only calculate variance if we have multiple blocks

    if (block_num > 1)
    {
        // Allreduce rho and eacp real values across block

        MPI_Allreduce(rho_real_proc, rho_real_block, qm_steps*DSTATES*DSTATES,
            MPI_DOUBLE, MPI_SUM, block_comm);

        MPI_Allreduce(eacp_real_proc, eacp_real_block, qm_steps*DSTATES*DSTATES,
            MPI_DOUBLE, MPI_SUM, block_comm);

        // get total IC number in block

        int block_ics = 0;

        MPI_Allreduce(&my_ics, &block_ics, 1, MPI_INT, MPI_SUM, block_comm);

        // normalize and calculate variance

        for (int i = 0; i < qm_steps; i++)
        {
            for (int j = 0; j < DSTATES*DSTATES; j++)
            {
                rho_real_block[i*DSTATES*DSTATES+j] /= block_ics;
                eacp_real_block[i*DSTATES*DSTATES+j] /= block_ics;
            }

            int entry = i*DSTATES*DSTATES;

            rho_var_block[i] = rho_real[entry] - rho_real_block[entry];
            rho_var_block[i] *= rho_var_block[i];

            eacp_var_block[i] = eacp_real[entry] - eacp_real_block[entry];
            eacp_var_block[i] *= eacp_var_block[i];
        }

        // reduce variances across parallel procs in blocks

        MPI_Allreduce(rho_var_block, rho_var, qm_steps, MPI_DOUBLE,
            MPI_SUM, var_comm);

        MPI_Allreduce(eacp_var_block, eacp_var, qm_steps, MPI_DOUBLE,
            MPI_SUM, var_comm);

        for (int i = 0; i < qm_steps; i++)
        {
            double temp;

            rho_var[i] /= (block_num - 1);
            temp = rho_var[i];
            rho_var[i] = sqrt(temp);

            eacp_var[i] /= (block_num - 1);
            temp = eacp_var[i];
            eacp_var[i] = sqrt(temp);        
        }    

    } // end variance calc clause

    // output summary

    if (simData.break_flg)
    {
        // we're restarting, so only print partial summary

        if (me == 0)
        {
            int repeat = 50;
        
            for (int i = 0; i < repeat; i++)
                fprintf(outfile, "-");

            fprintf(outfile, "\nTruncated Simulation Summary\n");

            for (int i = 0; i < repeat; i++)
                fprintf(outfile, "-");

            fprintf(outfile, "\n\n");

            fprintf(outfile, "Quantum steps: %d\n", qm_steps);
            fprintf(outfile, "Memory length (kmax): %d\n", kmax);
            fprintf(outfile, "Filtering threshold: %.3e\n", filter_thresh);
            fprintf(outfile, "Step length (a.u): %.5f\n", dt);
            fprintf(outfile, "IC num: %d\n", ic_tot);
            fprintf(outfile, "RNG seed: %lu\n", seed);
            fprintf(outfile, "MC skip: %ld\n", steps);

            if (ANALYTICFLAG > 0)
                fprintf(outfile, "Analytic trajectory integration: on\n");
            else
            {
                fprintf(outfile, "Analytic trajectory integration: off\n");
                fprintf(outfile, "Action integration points (per step): %d\n", step_pts);
            }

            if (simData.fixed_ref)
                fprintf(outfile, "Simulation used EACP reference fixed at point: %.4f\n", 
                    simData.ref_state);
            else
                fprintf(outfile, "Simulation used EACP reference hopping\n");

            fprintf(outfile, "Input spectral density: %s\n", simData.infile);
            fprintf(outfile, "Configuration file: %s\n\n", config_file);

            fprintf(outfile, "Total simulated time (a.u.): %.4f\n", qm_steps*dt);
            fprintf(outfile, "Current simulated time (a.u.): %.4f\n", simData.tstep_index*dt);
            fprintf(outfile, "Processors: %d\n\n", nprocs);

            fprintf(outfile, "Total simulation time: %.3f min\n\n", g_runtime/60.0);

            for (int i = 0; i < repeat; i++)
                fprintf(outfile, "-");

            fprintf(outfile, "\nBath Summary\n");

            for (int i = 0; i < repeat; i++)
                fprintf(outfile, "-");

            fprintf(outfile, "\n\n");

            fprintf(outfile, "Bath modes: %d\n", nmodes);
            fprintf(outfile, "Bath temperature: %.2f\n", bath_temp);
            fprintf(outfile, "Inverse temperature: %.4f\n", beta);
            fprintf(outfile, "Bath mode mass parameter: %.3f\n", mass);
        
            if (SHIFT)
                fprintf(outfile, "Using shifted W(x,p) (minimum at x=lambda)\n\n");
            else
                fprintf(outfile, "Using unshifted W(x,p) (minimum at x=0)\n\n");

            for (int i = 0; i < repeat; i++)
                fprintf(outfile, "-");

            fprintf(outfile, "\nSystem Summary\n");

            for (int i = 0; i < repeat; i++)
                fprintf(outfile, "-");

            fprintf(outfile, "\n\n");

            fprintf(outfile, "Off-diagonal TLS element: %f\n", tls_freq);
            fprintf(outfile, "Asymmetry: %.7e hartree\n", asym);
            fprintf(outfile, "Left DVR state: %.3f\n", dvr_left);
            fprintf(outfile, "Right DVR state: %.3f\n\n", dvr_right);

            fprintf(outfile, "End of restart-dumped data.\n");
            fprintf(outfile, "Use restart command line option to continue simulation.\n\n");

        } // end restart info summary
    }
    else
    {
      // no restart, so print full output

      if (me == 0)
      {
        int repeat = 50;
        
        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nSimulation Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Quantum steps: %d\n", qm_steps);
        fprintf(outfile, "Memory length (kmax): %d\n", kmax);
        fprintf(outfile, "Filtering threshold: %.3e\n", filter_thresh);
        fprintf(outfile, "Step length (a.u): %.5f\n", dt);
        fprintf(outfile, "IC num: %d\n", ic_tot);
        fprintf(outfile, "RNG seed: %lu\n", seed);
        fprintf(outfile, "MC skip: %ld\n", steps);

        if (ANALYTICFLAG > 0)
            fprintf(outfile, "Analytic trajectory integration: on\n");
        else
        {
            fprintf(outfile, "Analytic trajectory integration: off\n");
            fprintf(outfile, "Action integration points (per step): %d\n", step_pts);
        }

        if (simData.fixed_ref)
            fprintf(outfile, "Simulation used EACP reference fixed at point: %.4f\n", 
                simData.ref_state);
        else
            fprintf(outfile, "Simulation used EACP reference hopping\n");

        fprintf(outfile, "Input spectral density: %s\n", simData.infile);
        fprintf(outfile, "Configuration file: %s\n\n", config_file);

        fprintf(outfile, "Total simulated time (a.u.): %.4f\n", qm_steps*dt);

        fprintf(outfile, "Processors: %d\n\n", nprocs);
        fprintf(outfile, "Total simulation time: %.3f min\n\n", g_runtime/60.0);

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nBath Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Bath modes: %d\n", nmodes);
        fprintf(outfile, "Bath temperature: %.2f\n", bath_temp);
        fprintf(outfile, "Inverse temperature: %.4f\n", beta);
        fprintf(outfile, "Bath mode mass parameter: %.3f\n", mass);
        
        if (SHIFT)
            fprintf(outfile, "Using shifted W(x,p) (minimum at x=lambda)\n\n");
        else
            fprintf(outfile, "Using unshifted W(x,p) (minimum at x=0)\n\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nSystem Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Off-diagonal TLS element: %f\n", tls_freq);
        fprintf(outfile, "Asymmetry: %.7e hartree\n", asym);
        fprintf(outfile, "Left DVR state: %.3f\n", dvr_left);
        fprintf(outfile, "Right DVR state: %.3f\n\n", dvr_right);

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nDensity Matrix Values\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        if (RHOPRINT)   // print all entries of matrix
        {    
            // open separate output file

            char * basename = strtok(simData.outfile, ".");
            char rho_full_name[FLEN];

            sprintf(rho_full_name, "%s_full_matrix.dat", basename);

            FILE * full_outfile;

            full_outfile = fopen(rho_full_name, "w");

            if (full_outfile == NULL)
            {
                fprintf(stderr, "Could not open %s for full rho(t) output\n", 
                    rho_full_name);
            }
            else
            {
                // print full rho(t) to new file

                fprintf(full_outfile, 
                    "Columns are flattened rho(t) elements\n");

                fprintf(full_outfile, 
                    "Each of the %d pairs is ordered by Re(rho(t)) Im(rho(t))\n\n", 
                        DSTATES*DSTATES);

                for (int i = 0; i < qm_steps; i++)
                {
                    fprintf(full_outfile, "%7.4f ", (i+1)*dt);

                    for (int j = 0; j < DSTATES*DSTATES; j++)
                    {
                        fprintf(full_outfile, "%13.10f %13.10f ", 
                            rho_real[i*DSTATES*DSTATES+j],
                                rho_imag[i*DSTATES*DSTATES+j]);
                    }

                    fprintf(full_outfile, "\n");
                }

                fclose(full_outfile);

            } // end full rho(t) printing file clause

            // also print baseline results to standard file

            for (int i = 0; i < qm_steps; i++)
            {
                int entry = i*DSTATES*DSTATES;

                fprintf(outfile, "%7.4f %8.5f %6.3f (Tr = %13.10f)\n", (i+1)*dt, 
                    rho_real[entry], rho_var[i], rho_real[entry]+rho_real[entry+3]);    
            }
        }
        else    // only print (0,0) element
        {
            for (int i = 0; i < qm_steps; i++)
            {
                int entry = i*DSTATES*DSTATES;

                fprintf(outfile, "%7.4f %8.5f %6.3f (Tr = %13.10f)\n", (i+1)*dt, 
                    rho_real[entry], rho_var[i], rho_real[entry]+rho_real[entry+3]);
            }

        } // end RHOPRINT clause

        fprintf(outfile, "\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nEACP trajectory (time vs. rho(t), std. dev., trace):\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");
    
        for (int i = 0; i < qm_steps; i++)
        {
            int entry = i*DSTATES*DSTATES;

            fprintf(outfile, "%7.4f %8.5f %6.3f (Tr = %13.10f)\n", (i+1)*dt, 
                eacp_real[entry], eacp_var[i], eacp_real[entry]+eacp_real[entry+3]);
        }

      } // end output conditional

    } // end restart output check

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

    for (int i = 0; i < qm_steps; i++)
    {
        delete [] rho_proc[i];
        delete [] rho_eacp_proc[i];
    }

    delete [] rho_proc;
    delete [] rho_ic_proc;
    delete [] rho_eacp_proc;
    delete [] modes;
    delete [] ref_modes;
    delete [] old_ref_list;
    delete [] rho_real_proc;
    delete [] rho_imag_proc;
    delete [] eacp_real_proc;
    delete [] eacp_imag_proc;
    delete [] rho_real;
    delete [] rho_imag;
    delete [] eacp_real;
    delete [] eacp_imag;

    delete [] rho_real_block;
    delete [] eacp_real_block;
    delete [] rho_var;
    delete [] eacp_var;
    delete [] rho_var_block;
    delete [] eacp_var_block;

    delete error;

    MPI_Finalize();

    return 0;
}

/*----------------------------------------------------------------------*/

// startup() -- read in and process configuration file 

void startup(char * config, struct SimInfo * sim, struct FlagInfo * flag, 
        MPI_Comm comm)
{
    Error * error = new Error(comm);
    int me;

    MPI_Comm_rank(comm, &me);

    ifstream conf_file;
    char buffer[FLEN];
    char * arg1;
    char * arg2;
    char * next;
    char * ptr;

    // set defaults

    sim->fixed_ref = false;         // default to hopping EACP
    sim->ref_state = dvr_left;      // defaults to reactant EACP reference
    sim->branch_state = BRANCH_LEFT; // defaults to left state branch
    sim->asym = 0.5;                // system asymmetry (in kcal/mol)
    sim->bath_modes = 60;           // number of bath oscillators
    sim->mc_steps = 50000;          // default MC burn
    sim->filter_thresh = 0.0;       // default to unfiltered run
    sim->step_pts = 100;            // number of action integration points
    sim->chunks = 5;                // must evenly divide step_pts
    sim->rho_steps = 100;            // points used to integrate U(t)
    sim->block_num = 1;                // number of blocks in final average

    sim->checkpoint_flg = false;
    sim->plain_dump_flg = false;
    sim->dump_state_flg = false;
    sim->restart_flg = false;
    sim->check_freq = 200;          // checkpoint every 200 steps
    sim->plain_dump_freq = 100;     // dump plaintext output every 100 steps 
    sim->time_buffer = 20*60.0;     // 20 minute default buffer  
    sim->alpha = 0.3; 

    // note that original definitions use integer
    // flags, which is why these are ints and not bool

    flag->shift_flag = 1;           // turn on bath shifting
    flag->analytic_flag = 0;        // turn off exact traj. by default
    flag->report_flag = 0;          // turn off bath mode reporting
    flag->traj_flag = 0;            // turn off x0, p0 reporting
    flag->rho_print_flag = 0;       // turn off full rho(t) printing
    flag->seed = 179524;            // GSL RNG seed
    flag->block_seed_spacing = 4782; // change in seed value per block

    // initialize flags to check validity of
    // configuration file input

    bool qmstepsflg = false;
    bool qmdtflg = false;
    bool bathtempflg = false;

    bool icnumflg = false;
    bool kmaxflg = false;

    bool inflg = false;
    bool outflg = false;

    // extra flags for restart commands

    bool time_budgetflg = false;
    bool dump_stateflg = false;

    // set category flags

    bool timeflg;
    bool simflg;
    bool fileflg;

    bool reqflg;

    conf_file.open(config, ios_base::in);
    if (!conf_file.is_open())
        error->one("Could not open configuration file");

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

        else if (strcmp(arg1, "checkpoints") == 0)
        {
            if (strcmp(arg2, "on") == 0)
                sim->checkpoint_flg = true;
            else if (strcmp(arg2, "off") == 0)
                sim->checkpoint_flg = false;
            else
                error->all("Unrecognized checkpointing option");
        }

        else if (strcmp(arg1, "check_freq") == 0)
        {
            sim->check_freq = atoi(arg2);
        }
/* ----------------- */

        else if (strcmp(arg1, "plain_dump") == 0)
        {
            if (strcmp(arg2, "on") == 0)
                sim->plain_dump_flg = true;
            else if (strcmp(arg2, "off") == 0)
                sim->plain_dump_flg = false;
            else
                error->all("Unrecognized plaintext dump option");
        }

        else if (strcmp(arg1, "plain_dump_freq") == 0)
        {
            sim->plain_dump_freq = atoi(arg2);
        }

/* ----------------- */

        else if (strcmp(arg1, "dump_state") == 0)
        {
            if (strcmp(arg2, "on") == 0)
                sim->dump_state_flg = true;
            else if (strcmp(arg2, "off") == 0)
                sim->dump_state_flg = false;
            else
                error->all("Unrecognized state-saving option");

            dump_stateflg = true;
        }

        else if (strcmp(arg1, "time_budget") == 0)
        {
            // time is in hh:mm:ss format (must include all terms)
            // i.e. 30 minutes would be 00:30:00

            int hrs = atoi(strtok(arg2, ":"));
            int mins = atoi(strtok(NULL, ":"));
            int secs = atoi(strtok(NULL, ":"));

            // store wall time in seconds

            sim->time_budget = (hrs*3600.0 + mins*60.0 + secs); 

            time_budgetflg = true;
        }

        else if (strcmp(arg1, "time_buffer") == 0)
        {
            // time is in hh:mm:ss format (must include all terms)
            // i.e. 30 minutes would be 00:30:00

            int hrs = atoi(strtok(arg2, ":"));
            int mins = atoi(strtok(NULL, ":"));
            int secs = atoi(strtok(NULL, ":"));

            // store wall time in seconds

            sim->time_buffer = (hrs*3600.0 + mins*60.0 + secs); 
        }

        else if (strcmp(arg1, "block_num") == 0)
        {
            // set number of blocks in MC average

            sim->block_num = atoi(arg2);

            if (sim->block_num <= 0)
                sim->block_num = 1;
        }

        else if (strcmp(arg1, "fixed_ref") == 0)
        {
            // set EACP as fixed or hopping

            if (strcmp(arg2, "on") == 0)
                sim->fixed_ref = true;
            else if (strcmp(arg2, "off") == 0)
                sim->fixed_ref = false;
            else
                error->all("Unrecognized EACP reference option");
        }

        else if (strcmp(arg1, "ref_state") == 0)
        {
            // set EACP reference (for fixed-reference runs)

            if (strcmp(arg2, "left") == 0)
            {
                sim->ref_state = dvr_left;
                sim->branch_state = BRANCH_LEFT;
            }
            else if (strcmp(arg2, "mid") == 0)
            {
                sim->ref_state = (dvr_left + dvr_right)/2.0;
                sim->branch_state = BRANCH_MID;
            }
            else if (strcmp(arg2, "right") == 0)
            {
                sim->ref_state = dvr_right;
                sim->branch_state = BRANCH_RIGHT;
            }
            else
                error->all("Unrecognized EACP reference state");
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

        else if (strcmp(arg1, "filter_thresh") == 0)
        {
            // set filtering threshold

            sim->filter_thresh = atof(arg2);
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

        else if (strcmp(arg1, "bath_shift") == 0)
        {
            // set SHIFT parameter

            if (strcmp(arg2, "on") == 0)
                flag->shift_flag = 1;
            else if (strcmp(arg2, "off") == 0)
                flag->shift_flag = 0;
            else
                error->all("Unrecognized bath shift option");
        }

        else if (strcmp(arg1, "exact_traj") == 0)
        {
            // set analytic trajectories on or off

            if (strcmp(arg2, "on") == 0)
                flag->analytic_flag = 1;
            else if (strcmp(arg2, "off") == 0)
                flag->analytic_flag = 0;
            else
                error->all("Unrecognized analytical trajectory option");
        }

        else if (strcmp(arg1, "mode_reporting") == 0)
        {
            // set bath mode reporting

            if (strcmp(arg2, "on") == 0)
                flag->report_flag = 1;
            else if (strcmp(arg2, "off") == 0)
                flag->report_flag = 0;
            else
                error->all("Unrecognized mode reporting option");
        }

        else if (strcmp(arg1, "traj_reporting") == 0)
        {
            // set trajectory reporting

            if (strcmp(arg2, "on") == 0)
                flag->traj_flag = 1;
            else if (strcmp(arg2, "off") == 0)
                flag->traj_flag = 0;
            else
                error->all("Unrecognized trajectory reporting option");
        }

        else if (strcmp(arg1, "full_rho_print") == 0)
        {
            // set RHOPRINT parameter

            if (strcmp(arg2, "on") == 0)
                flag->rho_print_flag = 1;
            else if (strcmp(arg2, "off") == 0)
                flag->rho_print_flag = 0;
            else
                error->all("Unrecognized bath shift option");
        }

        else if (strcmp(arg1, "rng_seed") == 0)
        {
            // set RNG seed (ensure seed isn't 0)

            float f_seed = atof(arg2);

            flag->seed = static_cast<unsigned long>(f_seed);

            if (flag->seed == 0)
                flag->seed = 179524;
        }

        else if (strcmp(arg1, "rng_spacing") == 0)
        {
            // set RNG spacing (this can be zero)

            float f_seed = atof(arg2);

            flag->block_seed_spacing = 
                static_cast<unsigned long>(f_seed);
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
            error->all("Must specify number of quantum steps and timestep values");

        if (!simflg)
            error->all("Must specify number of ICs, kmax, and bath temperature");

        if (!fileflg)
            error->all("Must specify spectral density input file, and data output file");
    }

    if (dump_stateflg)
    {
        if (!time_budgetflg)
        {
            error->all("Must specify time budget for state dumping");
        }
    }

    // ensure consistency of step_pts and chunk values

    int factor = sim->step_pts/sim->chunks;

    sim->step_pts = factor * sim->chunks;

    // check positive-definite quantities

    if (sim->bath_modes <= 0)
        error->all("Bath oscillator number must be positive");

    if (sim->ic_tot <= 0)
        error->all("Number of ICs must be positive");

    if (sim->dt <= 0)
        error->all("Timestep must be positive");

    if (sim->qm_steps <= 0)
        error->all("Total simulation steps must be positive");

    if (sim->kmax <= 0)
        error->all("Kmax segments must be positive");

    if (sim->step_pts <= 0)
        error->all("Numer of action integration points must be positive");

    if (sim->chunks <= 0)
        error->all("Action chunk number must be positive");

    if (sim->rho_steps <= 0)
        error->all("Number of ODE steps for rho(t) must be positive");

    if (sim->bath_temp <= 0)
        error->all("Bath temperature must be positive");

    // check non-negative quantities

    if (sim->mc_steps < 0)
        error->all("Monte Carlo burn length can't be negative");

    if (sim->filter_thresh < 0)
        error->all("Filtering threshold can't be negative");

    if (dump_stateflg && sim->time_buffer < 0)
        error->all("Backup dump buffer time can't be negative");

    // ensure kmax < total steps

    if (sim->kmax > sim->qm_steps)
        error->all("Memory length cannot exceed total simulation time");
    
    delete error;
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

  Error * error = new Error(MPI_COMM_WORLD);
  
  start = &str[strspn(str," \t\n\v\f\r")];
  if (*start == '\0') return NULL;
  
  if (*start == '"' || *start == '\'') {
    stop = strchr(&start[1],*start);
    if (!stop) error->all("Unbalanced quotes in input line");
    if (stop[1] && !isspace(stop[1]))
      error->all("Input line quote not followed by whitespace");
    start++;
  } else stop = &start[strcspn(start," \t\n\v\f\r")];
  
  if (*stop == '\0') *next = NULL;
  else *next = stop+1;
  *stop = '\0';

  delete error;

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
    Error * error = new Error(MPI_COMM_WORLD);

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
            {
                char errstr[FLEN];
                sprintf(errstr, "All modes have zero frequency");
                error->one(errstr);
            }
        }
    }

    delete error;

    return w0;
}

/*----------------------------------------------------------------------*/

void calibrate_mc(double * x_step, double * p_step, double * bath_freq, 
    double * bath_coup, gsl_rng * gen)
{
    const double max_trials = 1000;
    const double base_step = 10.0;
    const double scale = 0.8;
    const double low_thresh = 0.45;
    const double high_thresh = 0.55;
    const int tsteps = 1000;
    double * x_vals = new double [nmodes];
    double * p_vals = new double [nmodes];

    Error * error = new Error(MPI_COMM_WORLD);

    // initialize step sizes and (x,p) at minimum of x

    for (int i = 0; i < nmodes; i++)
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

    for (int i = 0; i < nmodes; i++)
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
            error->one("Failed to properly converge MC optimization in x-coord.");

    } // end x calibration

    // begin calibrating p steps

    for (int i = 0; i < nmodes; i++)
    {
        x_vals[i] = bath_coup[i] * ( dvr_left/(mass*bath_freq[i]*bath_freq[i]) );
        p_vals[i] = 0.0;
    }

    // run MC step tweaking for each mode 

    for (int i = 0; i < nmodes; i++)
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
            error->one("Failed to converge MC optimization in p-coord.");

    } // end p tweak

    delete [] x_vals;
    delete [] p_vals; 
}

/* ------------------------------------------------------------------------ */

double ic_gen(double * xvals, double * pvals, double * bath_freq, double * bath_coup,
    double * x_step, double * p_step, gsl_rng * gen)
{   
    //double step_max = range/100.0;

    double x_old, x_new;
    double p_old, p_new;

    long accepted = 1;

    for (long i = 1; i < steps; i++)
    { 
        // randomly select index to step, and generate
        // step size in x and p dimension

        int index = gsl_rng_uniform_int(gen, nmodes);
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

    double ratio = static_cast<double>(accepted)/steps;

    return ratio;
}

/* ------------------------------------------------------------------------ */

double dist(double x_old, double x_new, double p_old, double p_new, 
    double omega, double coup)
{
    // need atomic units     
    double f = hbar*omega*beta;

    if (SHIFT)
    {
        // shift to DVR state w/ -1 element in sigma_z basis
        double lambda = dvr_left * coup * ( 1.0/(mass*omega*omega) );
        //double lambda = dvr_right * coup * ( 1.0/(mass*omega*omega) );

        // shifting distribution for equilibrium

        x_new -= lambda;
        x_old -= lambda;
    }

    // calculate Wigner distribution ratio for these x,p vals

    double delta = (mass*omega/hbar)*(x_new*x_new - x_old*x_old) +
        (1.0/(mass*omega*hbar))*(p_new*p_new - p_old*p_old);

    double pre = tanh(f/2.0);

    double prob = exp(-pre*delta);

    return prob; 
}

/* ------------------------------------------------------------------------ */

// classically propagate unforced H.O. modes

void ho_update(Propagator & prop, Mode * mlist, double ref_state)
{
    double del_t = dt/step_pts;

    for (int mode = 0; mode < nmodes; mode++)
    {
        // set up ICs for trajectory

        double w = mlist[mode].omega;
        //double c = mlist[mode].c;

        double x0, xt;
        double p0, pt;
        double f0, ft;

        x0 = prop.x0_free[mode];
        p0 = prop.p0_free[mode];

        //f0 = -1.0*mass*w*w*x0;

        f0 = -1.0*mass*w*w*x0 + mlist[mode].c * ref_state;

        // clear out any old trajectory info
        // might be more efficient just to overwrite

        mlist[mode].x_t.clear();

        // integrate equations of motion

        for (int step = 0; step < step_pts; step++)
        {
            // Verlet update

            xt = x0 + (p0/mass)*del_t + 0.5*(f0/mass)*del_t*del_t;

            //ft = -1.0*mass*w*w*xt;

            ft = -1.0*mass*w*w*xt + mlist[mode].c * ref_state;

            pt = p0 + 0.5*(f0 + ft)*del_t;

            x0 = xt;
            p0 = pt;
            f0 = ft;
            
            // accumulate trajectory segment in mode object

            mlist[mode].x_t.push_back(xt);          

        } // end traj. loop

        // update current phase space point

        prop.x0_free[mode] = xt;
        prop.p0_free[mode] = pt;

    } // end mode loop

} // end ho_update

/* ------------------------------------------------------------------------- */

void ho_update_exact(Propagator & prop, Mode * mlist, double ref_state)
{
    double del_t = dt/2.0;
    double chunk_dt = dt/chunks;

    for (int mode = 0; mode < nmodes; mode++)
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

void build_ham(Propagator & prop, Mode * modes, int chunk)
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

    tls_mat[0] = asym; //dvr_left*(1.0*asym);
    tls_mat[1] = -1.0*off_diag;
    tls_mat[2] = -1.0*off_diag;
    tls_mat[3] = -1.0*asym; //dvr_right*(1.0*asym);

    // system-bath matrix includes linear coupling plus
    // quadratic offset

    // in this form, it also includes the bath potential energy

    left_sum = 0.0;
    right_sum = 0.0;

    for (int i = 0; i < nmodes; i++)
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

/*
    bath_mat[0] = left_sum + energy;
    bath_mat[1] = 0.0;
    bath_mat[2] = 0.0;
    bath_mat[3] = right_sum + energy;
*/

    // !! BEGIN DEBUG !!

    // Removing energy term to see if this is
    // dominating energy gap and causing issues

    bath_mat[0] = left_sum;
    bath_mat[1] = 0.0;
    bath_mat[2] = 0.0;
    bath_mat[3] = right_sum;

    // !! END DEBUG !!

    // total hamiltonian is sum of system and system-bath parts

    prop.ham[0] = tls_mat[0] + bath_mat[0];
    prop.ham[1] = tls_mat[1] + bath_mat[1];
    prop.ham[2] = tls_mat[2] + bath_mat[2];
    prop.ham[3] = tls_mat[3] + bath_mat[3];

} // end build_ham()

/* ------------------------------------------------------------------------ */

void build_ham_exact(Propagator & prop, Mode * modes)
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

    tls_mat[0] = asym; //dvr_left*(1.0*asym);
    tls_mat[1] = -1.0*off_diag;
    tls_mat[2] = -1.0*off_diag;
    tls_mat[3] = -1.0*asym; //dvr_right*(1.0*asym);

    // system-bath matrix includes linear coupling plus
    // quadratic offset

    // in this form, it also includes the bath potential energy

    left_sum = 0.0;
    right_sum = 0.0;

    for (int i = 0; i < nmodes; i++)
    {
        double csquare = modes[i].c*modes[i].c;
        double wsquare = modes[i].omega*modes[i].omega;
        double x = prop.x0_free[i];

        left_sum += -1.0*modes[i].c*x*dvr_left +
            csquare*dvr_left*dvr_left/(2.0*mass*wsquare);

        right_sum += -1.0*modes[i].c*x*dvr_right +
            csquare*dvr_right*dvr_right/(2.0*mass*wsquare);

        energy += 0.5*mass*wsquare*x*x;
    }

/*
    bath_mat[0] = left_sum + energy;
    bath_mat[1] = 0.0;
    bath_mat[2] = 0.0;
    bath_mat[3] = right_sum + energy;
*/

    // !! BEGIN DEBUG !!

    // Removing energy term to see if this is
    // dominating energy gap and causing issues

    bath_mat[0] = left_sum;
    bath_mat[1] = 0.0;
    bath_mat[2] = 0.0;
    bath_mat[3] = right_sum;

    // !! END DEBUG !!

    // total hamiltonian is sum of system and system-bath parts

    prop.ham[0] = tls_mat[0] + bath_mat[0];
    prop.ham[1] = tls_mat[1] + bath_mat[1];
    prop.ham[2] = tls_mat[2] + bath_mat[2];
    prop.ham[3] = tls_mat[3] + bath_mat[3];
}

/* ------------------------------------------------------------------------ */

void qcpi_update(Path & qm_path, Mode * mlist)
{
    double del_t = dt/step_pts;
    double dvr_vals[DSTATES] = {dvr_left, dvr_right};

    for (int mode = 0; mode < nmodes; mode++)
    {
        // set up ICs for first half of path

        double w = mlist[mode].omega;
        double c = mlist[mode].c;

        double x0, xt;
        double p0, pt;
        double f0, ft;

        x0 = qm_path.x0[mode];
        p0 = qm_path.p0[mode];

        unsigned size = qm_path.fwd_path.size();

        unsigned splus = qm_path.fwd_path[size-2];
        unsigned sminus = qm_path.bwd_path[size-2];

        f0 = -1.0*mass*w*w*x0 + c*0.5*(dvr_vals[splus] +
            dvr_vals[sminus]);

        // clear out any old trajectory info
        // might be more efficient just to overwrite

        mlist[mode].x_t.clear();

        // integrate equations of motion

        for (int step = 0; step < step_pts/2; step++)
        {
            // Verlet update

            xt = x0 + (p0/mass)*del_t + 0.5*(f0/mass)*del_t*del_t;

            ft = -1.0*mass*w*w*xt + c*0.5*(dvr_vals[splus] +
                dvr_vals[sminus]);

            pt = p0 + 0.5*(f0 + ft)*del_t;

            x0 = xt;
            p0 = pt;
            f0 = ft;
            
            // accumulate trajectory segment in mode object

            mlist[mode].x_t.push_back(xt);          

        } // end first half traj. loop

        // loop over second half of trajectory

        splus = qm_path.fwd_path[size-1];
        sminus = qm_path.bwd_path[size-1];

        f0 = -1.0*mass*w*w*x0 + c*0.5*(dvr_vals[splus] +
            dvr_vals[sminus]);
    
        for (int step = step_pts/2; step < step_pts; step++)
        {
            // Verlet update

            xt = x0 + (p0/mass)*del_t + 0.5*(f0/mass)*del_t*del_t;

            ft = -1.0*mass*w*w*xt + c*0.5*(dvr_vals[splus] +
                dvr_vals[sminus]);

            pt = p0 + 0.5*(f0 + ft)*del_t;

            x0 = xt;
            p0 = pt;
            f0 = ft;
            
            // accumulate trajectory segment in mode object

            mlist[mode].x_t.push_back(xt);          

        } // end second half traj. loop

        // update current phase space point

        qm_path.x0[mode] = xt;
        qm_path.p0[mode] = pt;

    } // end mode loop

} // end qcpi_update

/* ------------------------------------------------------------------------- */

void qcpi_update_exact(Path & qm_path, Mode * mlist)
{
    double del_t = dt/2.0;
    double dvr_vals[DSTATES] = {dvr_left, dvr_right};

    for (int mode = 0; mode < nmodes; mode++)
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

// NOTE: Do we need c^2*s^2 correction added here?

double action_calc(Path & qm_path, Mode * mlist, Mode * reflist)
{
    double del_t = dt/step_pts;
    double dvr_vals[DSTATES] = {dvr_left, dvr_right};

    // loop over modes and integrate their action contribution
    // as given by S = c*int{del_s(t')*x(t'), 0, t_final}

    double action = 0.0;

    for (int mode = 0; mode < nmodes; mode++)
    {
        double sum = 0.0;

        // set indices to correct half steps

        unsigned size = qm_path.fwd_path.size();

        unsigned splus = qm_path.fwd_path[size-2];
        unsigned sminus = qm_path.bwd_path[size-2];

        // integrate first half of trajectory

        for (int step = 0; step < step_pts/2; step++)
        {
            double ds = dvr_vals[splus] - dvr_vals[sminus];
            double pre = mlist[mode].c * ds;

            if ( step == 0 || step == (step_pts/2)-1 )
                pre *= 0.5;

            sum += pre * (mlist[mode].x_t[step] - reflist[mode].x_t[step]);

        } // end first half traj. loop

        splus = qm_path.fwd_path[size-1];
        sminus = qm_path.bwd_path[size-1];

        // integrate second half of trajectory

        for (int step = step_pts/2; step < step_pts; step++)
        {
            double ds = dvr_vals[splus] - dvr_vals[sminus];
            double pre = mlist[mode].c * ds;

            if ( step == (step_pts/2) || step == step_pts-1 )
                pre *= 0.5;

            sum += pre * (mlist[mode].x_t[step] - reflist[mode].x_t[step]);

        } // end second half traj. loop

        action += sum * del_t;

    } // end mode loop

    return action;

} // end action_calc

/* ------------------------------------------------------------------------ */

double action_calc_exact(Path & qm_path, Mode * mlist, Mode * reflist)
{
    double dvr_vals[DSTATES] = {dvr_left, dvr_right};

    // loop over modes and integrate their action contribution
    // as given by S = c*int{del_s(t')*x(t'), 0, t_final}

    double action = 0.0;

    for (int mode = 0; mode < nmodes; mode++)
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

// plaintext_out() -- function which writes out progress in finding rho(t)
// up to current checkpoint. This is designed to maximize useful info
// obtained from a run, even in case of catastrophic failure 

void plaintext_out(complex<double> ** rho_curr_checkpoint, 
    complex<double> ** rho_full_checkpoint, SimInfo & simData, 
        int ic_curr, int tstep, MPI_Comm comm)
{
    // get MPI info

    int me, nprocs;

    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &nprocs);

    // open file to write out

    FILE * saveFile;
    char saveName[FLEN];
    int error_flg = 0;

    sprintf(saveName, "%s_plain_out_step_%d.dat", simData.backupname, tstep);

    if (me == 0)
    { 
        if (file_exists(saveName))
        {
            // move file to backup copy if it already exists

            int sys_val;
            char sys_cmd[FLEN];

            sprintf(sys_cmd, "mv %s %s_%s", 
                saveName, backup_prefix, saveName);

            sys_val = system(sys_cmd);

            // warn on backup failure

            if (sys_val)
            {
                fprintf(stdout, "WARNING: Could not backup data file %s\n",
                    saveName);
            }
        }

        saveFile = fopen(saveName, "w");

        // warn if we can't open the save file and exit function

        if (saveFile == NULL)
        {
            fprintf(stdout, "WARNING: Could not open file %s for plain-text dump!\n",
                saveName);

            error_flg = 1;
        }
    }

    // check file status and exit on failure

    int kill_flg;

    MPI_Allreduce(&error_flg, &kill_flg, 1, MPI_INT, MPI_SUM, comm);

    if (kill_flg > 0)
        return;

    // print message on organization for clarity

    if (me == 0)
    {
        fprintf(saveFile, 
            "Columns are flattened rho(t) elements\n");

        fprintf(saveFile, 
            "Each of the %d pairs is ordered by Re(rho(t)) Im(rho(t))\n\n", 
                DSTATES*DSTATES);
    }

    // find total completed ICs across procs

    int finished_ics;

    MPI_Allreduce(&ic_curr, &finished_ics, 1, MPI_INT, MPI_SUM, comm);

    // print total and running rho(t) values to file

    double * rho_real_proc = new double [qm_steps*DSTATES*DSTATES];
    double * rho_imag_proc = new double [qm_steps*DSTATES*DSTATES];

    double * rho_real = new double [qm_steps*DSTATES*DSTATES];
    double * rho_imag = new double [qm_steps*DSTATES*DSTATES];

    // determine which data to print

    if (finished_ics == 0)
    {
        // No completed ICs, print short message
        
        if (me == 0)
            fprintf(saveFile, 
                "No ICs finished, only running summary will be printed\n\n");
    }
    else
    {
        // reduce full IC matrix across procs and normalize

        for (int i = 0; i < qm_steps; i++)
        {
            for (int j = 0; j < DSTATES*DSTATES; j++)
            {
                rho_real_proc[i*DSTATES*DSTATES+j] = 
                    rho_full_checkpoint[i][j].real();

                rho_imag_proc[i*DSTATES*DSTATES+j] = 
                    rho_full_checkpoint[i][j].imag();

                rho_real[i*DSTATES*DSTATES+j] = 0.0;
                rho_imag[i*DSTATES*DSTATES+j] = 0.0;
            }
        }

        // Allreduce the real and imaginary arrays

        MPI_Allreduce(rho_real_proc, rho_real, qm_steps*DSTATES*DSTATES,
            MPI_DOUBLE, MPI_SUM, comm);

        MPI_Allreduce(rho_imag_proc, rho_imag, qm_steps*DSTATES*DSTATES,
            MPI_DOUBLE, MPI_SUM, comm);

        // scale arrays by ICs completed

        for (int i = 0; i < qm_steps; i++)
        {
            for (int j = 0; j < DSTATES*DSTATES; j++)
            {
                rho_real[i*DSTATES*DSTATES+j] /= finished_ics;
                rho_imag[i*DSTATES*DSTATES+j] /= finished_ics;
            }
        }

        // print to output file
        
        if (me == 0)
        {
            fprintf(saveFile, "Density matrix averaged over %d ICs (real and imag):\n\n",
                finished_ics);

            for (int i = 0; i < qm_steps; i++)
            {   
                fprintf(saveFile, "%.4f ", simData.dt*(i+1) );

                for (int j = 0; j < DSTATES*DSTATES; j++)
                {
                    fprintf(saveFile, "%13.10f %13.10f ", rho_real[i*DSTATES*DSTATES+j],
                        rho_imag[i*DSTATES*DSTATES+j]);
                }

                fprintf(saveFile, "\n");

            } // end printing loop (proc 0)

            fprintf(saveFile, "\n");

        } // end printing if clause  

    } // end total rho(t) printing
 
    // locate first zero in current rho(t)
/*
    int min_step = 0;
    double epsilon = 1e-6;

    while ( abs(rho_curr_checkpoint[min_step][0]) > epsilon && 
        min_step < qm_steps )
    {
        min_step++;
    }

    // find global minimum step

    int global_min = qm_steps;

    MPI_Allreduce(&min_step, &global_min, 1, MPI_INT, MPI_MIN, comm);
*/
    // reduce current rho(t) array

    for (int i = 0; i < qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real_proc[i*DSTATES*DSTATES+j] = 
                rho_curr_checkpoint[i][j].real();

            rho_imag_proc[i*DSTATES*DSTATES+j] = 
                rho_curr_checkpoint[i][j].imag();

            rho_real[i*DSTATES*DSTATES+j] = 0.0;
            rho_imag[i*DSTATES*DSTATES+j] = 0.0;
        }
    }

    // Allreduce the real and imaginary arrays

    MPI_Allreduce(rho_real_proc, rho_real, qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, comm);

    MPI_Allreduce(rho_imag_proc, rho_imag, qm_steps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, comm);

    // scale arrays by nprocs to average

    for (int i = 0; i < qm_steps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real[i*DSTATES*DSTATES+j] /= nprocs;
            rho_imag[i*DSTATES*DSTATES+j] /= nprocs;
        }
    }

    // print current rho(t) array to file

    if (me == 0)
    {
        fprintf(saveFile, "Current density matrix averaged over %d procs (real and imag):\n\n",
            nprocs);

        //for (int i = 0; i < global_min; i++)
        for (int i = 0; i < tstep; i++)
        {   
            fprintf(saveFile, "%.4f ", simData.dt*(i+1) );

            for (int j = 0; j < DSTATES*DSTATES; j++)
            {
                fprintf(saveFile, "%13.10f %13.10f ", rho_real[i*DSTATES*DSTATES+j],
                    rho_imag[i*DSTATES*DSTATES+j]);
            }

            fprintf(saveFile, "\n");

        } // end printing loop (proc 0)

        fprintf(saveFile, "\n");

    } // end printing if clause 

    // cleanup

    if (me == 0)
        fclose(saveFile);

    delete [] rho_real_proc;
    delete [] rho_imag_proc;
    
    delete [] rho_real;
    delete [] rho_imag;
}

/* ------------------------------------------------------------------------ */

// save_state() -- general top-level function to control the binary backup
// process for checkpointing and restart file writing

void save_state(BathPtr & bath_pointers, gsl_rng * gen, Propagator & curr_prop, 
    DensityPtr & density_pointers, vector<Path> & pList, Mode * mList, 
        SimInfo & simData, FlagInfo & flagData, char * type_name, int my_rank)
{
    // open binary file to save calculation state

    FILE * saveFile;

    char saveName[FLEN];

    sprintf(saveName, "%s_%s_%d.%s", type_name, 
        state_data_name, my_rank, state_data_ext);

    if (file_exists(saveName))
    {
        // move file to backup copy if it already exists

        int sys_val;
        char sys_cmd[FLEN];

        sprintf(sys_cmd, "mv %s %s_%s", 
            saveName, backup_prefix, saveName);

        sys_val = system(sys_cmd);

        // warn on backup failure

        if (sys_val)
        {
            fprintf(stdout, "WARNING: could not backup data file %s\n",
                saveName);
        }
    }

    saveFile = fopen(saveName, "wb");

    if (saveFile == NULL)
    {
        char err_msg[FLEN];
        sprintf(err_msg, "Could not open state file %s for writing\n",
            saveName);

        throw std::runtime_error(err_msg);
    }

    // copy out bath vectors

    write_bath(bath_pointers, simData, saveFile);

    // copy out RNG state

    write_rng(gen, saveFile);

    // copy out Propagator elements

    write_prop(curr_prop, saveFile);

    // copy out density matrices

    write_rho(density_pointers, simData, saveFile);

    // copy out Path vector elements

    write_paths(pList, saveFile);

    // copy out blocked density matrices

    write_modes(mList, simData, saveFile);

    // copy out program info

    write_prog(simData, flagData, saveFile);
    
    // clean up

    fclose(saveFile);
}

/* ------------------------------------------------------------------------ */

// write_bath() -- helper function to serialize and copy out data on bath
// oscillator phase space and MC step sizes

/*
struct BathPtr
{
    double * xvals;
    double * pvals;
    double * x_step;
    double * p_step;
};
*/

void write_bath(BathPtr & bath_pointers, SimInfo & simData, FILE * saveFile)
{
    // write out size of all arrays

    unsigned len = simData.bath_modes;

    fwrite(&len, sizeof(unsigned), 1, saveFile);

    // write xvals array

    fwrite(bath_pointers.xvals, sizeof(double), len, saveFile);

    // write pvals array

    fwrite(bath_pointers.pvals, sizeof(double), len, saveFile);

    // write x_step array

    fwrite(bath_pointers.x_step, sizeof(double), len, saveFile);

    // write p_step array

    fwrite(bath_pointers.p_step, sizeof(double), len, saveFile);
}

/* ------------------------------------------------------------------------ */

// write_rng() -- helper function to write out RNG state using GSL library
// functions

void write_rng(gsl_rng * gen, FILE * saveFile)
{
    gsl_rng_fwrite(saveFile, gen);
}

/* ------------------------------------------------------------------------ */

// write_prop() -- helper function to serialize and copy out data from
// Propagator object to binary backup file. Note that not all fields are
// written, since some are used for temporary data.

/*
struct Propagator
{
    complex<double> * prop;
    complex<double> * ham;
    complex<double> * ptemp;
    vector<double> x0_free;
    vector<double> p0_free;
};
*/

void write_prop(Propagator & curr_prop, FILE * saveFile)
{
    // print propagator length (to avoid cross-run assumptions on DSTATES)
        
    unsigned propLen = static_cast<unsigned>(DSTATES*DSTATES);

    fwrite(&propLen, sizeof(unsigned), 1, saveFile);

    // write out current propagator array

    fwrite(curr_prop.prop, sizeof(complex<double>), propLen, saveFile);

    // write size of x0, p0 arrays

    unsigned ic_len = curr_prop.x0_free.size();

    fwrite(&ic_len, sizeof(unsigned), 1, saveFile);

    // serialize and write out x0 array

    double * serial_ics = new double [ic_len];

    for (unsigned i = 0; i < ic_len; i++)
        serial_ics[i] = curr_prop.x0_free[i];

    fwrite(serial_ics, sizeof(double), ic_len, saveFile);

    // serialize and write out p0 array

    for (unsigned i = 0; i < ic_len; i++)
        serial_ics[i] = curr_prop.p0_free[i];

    fwrite(serial_ics, sizeof(double), ic_len, saveFile);

    delete [] serial_ics;
}

/* ------------------------------------------------------------------------ */

// write_rho() -- helper function to serialize and copy data from single
// DensityMatrices element to binary backup file

/* 
struct DensityPtr
{
    complex<double> ** rho_proc;
    complex<double> * rho_ic_proc;
    complex<double> ** rho_eacp_proc;

    complex<double> ** rho_curr_checkpoint;
    complex<double> ** rho_full_checkpoint;
};

*/

void write_rho(DensityPtr & density_pointers, SimInfo & simData, 
    FILE * saveFile)
{
    // print first array size

    int xLen = qm_steps;

    fwrite(&xLen, sizeof(int), 1, saveFile);

    // print second array size

    int yLen = DSTATES * DSTATES;

    fwrite(&yLen, sizeof(int), 1, saveFile);

    // loop over array dimensions and write out QCPI matrix

    complex<double> * rho_single = new complex<double> [yLen];

    for (int step = 0; step < xLen; step++)
    {
        // copy line out

        for (int elem = 0; elem < yLen; elem++)
            rho_single[elem] = density_pointers.rho_proc[step][elem];

        fwrite(rho_single, sizeof(complex<double>), yLen, saveFile);
    }

    // can write rho_ic_proc directly

    fwrite(density_pointers.rho_ic_proc, sizeof(complex<double>), 
        xLen, saveFile);

    // loop over array dimensions and write out EACP matrix

    for (int step = 0; step < xLen; step++)
    {
        // copy line out

        for (int elem = 0; elem < yLen; elem++)
            rho_single[elem] = density_pointers.rho_eacp_proc[step][elem];

        fwrite(rho_single, sizeof(complex<double>), yLen, saveFile);
    }

    // loop over array dimensions and write out current checkpoint data

    for (int step = 0; step < xLen; step++)
    {
        // copy line out

        for (int elem = 0; elem < yLen; elem++)
            rho_single[elem] = density_pointers.rho_curr_checkpoint[step][elem];

        fwrite(rho_single, sizeof(complex<double>), yLen, saveFile);
    }

    // loop over array dimensions and write out accumulated checkpoint data

    for (int step = 0; step < xLen; step++)
    {
        // copy line out

        for (int elem = 0; elem < yLen; elem++)
            rho_single[elem] = density_pointers.rho_full_checkpoint[step][elem];

        fwrite(rho_single, sizeof(complex<double>), yLen, saveFile);
    }

    // write out list of old reference states

    fwrite(density_pointers.old_ref_list, sizeof(Ref), qm_steps, saveFile);

    delete [] rho_single;
}

/* ------------------------------------------------------------------------ */

// write_paths() -- helper function to serialize and copy out data from
// Path vector to backup file. Note that we don't write out ic_vec entries or 
// path active status because these have standard values at the end of every 
// segment loop when we go to write out a restart or checkpoint file

/*
struct Path
{
    vector<unsigned> fwd_path;
    vector<unsigned> bwd_path;
    complex<double> product;
    complex<double> eacp_prod;
    vector<double> x0;
    vector<double> p0;

    // filtering look-ahead
    vector<iter_pair> ic_vec;

    bool active;
};
*/

void write_paths(vector<Path> & pList, FILE * saveFile)
{
    // print path list and fwd/bwd length

    unsigned len = pList.size();

    fwrite(&len, sizeof(unsigned), 1, saveFile);

    unsigned path_len = pList[0].fwd_path.size();

    fwrite(&path_len, sizeof(unsigned), 1, saveFile);

    // print bath IC length

    unsigned bath_len = pList[0].x0.size();

    fwrite(&bath_len, sizeof(unsigned), 1, saveFile);

    unsigned * rho_path = new unsigned [path_len];

    double * bath_vals = new double [bath_len];

    // loop over path list to print elements one-by-one

    for (unsigned path = 0; path < len; path++)
    {
        // serialize and write fwd path

        for (unsigned i = 0; i < path_len; i++)
            rho_path[i] = pList[path].fwd_path[i];

        fwrite(rho_path, sizeof(unsigned), path_len, saveFile);

        // serialize and write bwd path

        for (unsigned i = 0; i < path_len; i++)
            rho_path[i] = pList[path].bwd_path[i];

        fwrite(rho_path, sizeof(unsigned), path_len, saveFile);

        // write out rho and eacp products

        fwrite(&(pList[path].product), sizeof(complex<double>), 1,
            saveFile);

        fwrite(&(pList[path].eacp_prod), sizeof(complex<double>), 1,
            saveFile);

        // write out ICs (as x0 and p0)

        for (unsigned i = 0; i < bath_len; i++)
            bath_vals[i] = pList[path].x0[i];

        fwrite(bath_vals, sizeof(double), bath_len, saveFile);

        for (unsigned i = 0; i < bath_len; i++)
            bath_vals[i] = pList[path].p0[i];

        fwrite(bath_vals, sizeof(double), bath_len, saveFile);

    } // end Path vector writing

    delete [] rho_path;
    delete [] bath_vals;
}

/* ------------------------------------------------------------------------ */

// write_modes() -- helper function to serialize and copy data from
// bath mode listing to backup file

/*
struct Mode
{
    vector<double> x_t;
    
    double c;
    double omega;
};
*/

void write_modes(Mode * mList, SimInfo & simData, FILE * saveFile)
{
    // write out total length of mode array

    unsigned bath_modes = static_cast<unsigned>(simData.bath_modes);

    fwrite(&bath_modes, sizeof(unsigned), 1, saveFile);

    // write out x(t) list size

    unsigned list_len = mList[0].x_t.size();

    fwrite(&list_len, sizeof(unsigned), 1, saveFile);

    // allocate array to serialize x(t) vector

    double * xt_array = new double [list_len];

    // loop over modes and write out relevant data

    for (unsigned mode = 0; mode < bath_modes; mode++)
    {
        // serialize x(t) vector

        for (unsigned i = 0; i < list_len; i++)
            xt_array[i] = mList[mode].x_t[i];

        fwrite(xt_array, sizeof(double), list_len, saveFile);

        // write out frequency and coupling info

        fwrite(&(mList[mode].omega), sizeof(double), 1, saveFile);

        fwrite(&(mList[mode].c), sizeof(double), 1, saveFile);
    }

    delete [] xt_array;
}

/* ------------------------------------------------------------------------ */

// write_prog() -- helper function to copy out data from ProgramInfo
// structure to binary backup file (mostly error info and loop indices)

/*
struct SimInfo
{
    int ic_index;               // tracks starting IC for restart
    int tstep_index;            // tracks timestep loop number for restart

    char infile[FLEN];

    double asym;
    int ic_tot;
    double dt;

    int block_num;
};

struct FlagInfo
{
    // debugging flags and RNG seed

    int shift_flag;
    int analytic_flag;
};
*/

void write_prog(SimInfo & simData, FlagInfo & flagData, FILE * saveFile)
{
    // write out loop indices

    fwrite(&(simData.ic_index), sizeof(int), 1, saveFile);

    fwrite(&(simData.tstep_index), sizeof(int), 1, saveFile);

    // write out name of J(w)/w file as future consistency check

    unsigned fname_size = strlen(simData.infile) + 1;

    fwrite(&fname_size, sizeof(unsigned), 1, saveFile);

    fwrite(simData.infile, sizeof(char), fname_size, saveFile);

    // write out parameters we can't check via array sizes
    // for future consistency guarantees

    fwrite(&(simData.asym), sizeof(double), 1, saveFile);

    fwrite(&(simData.ic_tot), sizeof(int), 1, saveFile);

    fwrite(&(simData.dt), sizeof(double), 1, saveFile);

    fwrite(&(simData.block_num), sizeof(int), 1, saveFile);

    // write out important flag values

    fwrite(&(flagData.shift_flag), sizeof(int), 1, saveFile);
    
    fwrite(&(flagData.analytic_flag), sizeof(int), 1, saveFile);
}

/* ------------------------------------------------------------------------ */

// load_state() -- top-level function to control the order in which we
// scan through files to find most current system state information. Order
// is restart -> checkpoint -> backup checkpoint. The implementation of
// lower-level functions to read data should also throw errors if the files
// are corrupted or incomplete, so we don't need a separate check for this.

void load_state(BathPtr & bath_pointers, gsl_rng * gen, Propagator & new_prop, 
    DensityPtr & density_pointers, vector<Path> & newList, Mode * new_modes, 
        SimInfo & simData, int my_rank)
{
    char curr_name[FLEN];
    bool checkflg = false;
    bool backupflg = false;
    bool failflg = false;

    char filename[FLEN];

    sprintf(filename, "%s", simData.backupname);

    // begin by looking for restart files

    try
    {
        sprintf(curr_name, "%s_%s", restart_prefix, filename);

        read_backup(bath_pointers, gen, new_prop, density_pointers, 
            newList, new_modes, simData, curr_name, my_rank);
    }
    catch(const std::runtime_error & err)
    {
        // if we get an error, set a different flag

        fprintf(stdout, "Proc %d threw exception: %s", 
            my_rank, err.what());

        checkflg = true;
    }

    // second choice is checkpoint files

    if (checkflg)
    {
        try
        {
            sprintf(curr_name, "%s_%s", checkpoint_prefix, filename);

            read_backup(bath_pointers, gen, new_prop, density_pointers, 
                newList, new_modes, simData, curr_name, my_rank);
        }
        catch(const std::runtime_error & err)
        {
            // if we get error, set backup flag

            fprintf(stdout, "Proc %d threw exception: %s", 
                my_rank, err.what());

            backupflg = true;
        }
    }

    // last option is backup checkpoint files

    if (backupflg)
    {
        try
        {
            sprintf(curr_name, "%s_%s_%s", backup_prefix, 
                checkpoint_prefix, filename);

            read_backup(bath_pointers, gen, new_prop, density_pointers, 
                newList, new_modes, simData, curr_name, my_rank);
        }
        catch(const std::runtime_error & err)
        {
            // nothing else to try, fail and exit

            fprintf(stdout, "Proc %d threw exception: %s", 
                my_rank, err.what());
            
            failflg = true;
        }
    }

    if (failflg)
    {
        char err_msg[FLEN]; 
        sprintf(err_msg, 
            "ERROR: Could not find working fileset to recover state from\n");

        throw std::runtime_error(err_msg);
    }
}

/* ------------------------------------------------------------------------ */

// read_backup() -- general top-level function that reads binary backup files 
// created by save_state()

void read_backup(BathPtr & bath_pointers, gsl_rng * gen, Propagator & new_prop, 
    DensityPtr & density_pointers, vector<Path> & newList, Mode * new_modes, 
        SimInfo & simData, const char * curr_name, int my_rank)
{   
    // open binary file to read in new program state

    FILE * saveFile;

    char saveName[FLEN];

    sprintf(saveName, "%s_%s_%d.%s", curr_name, 
        state_data_name, my_rank, state_data_ext);

    saveFile = fopen(saveName, "rb");

    if (saveFile == NULL)
    {
        char err_msg[FLEN];
        sprintf(err_msg, "Could not open backup file %s for reading\n",
            saveName);

        throw std::runtime_error(err_msg);
    }

    // read in bath arrays
    
    read_bath(bath_pointers, saveFile);

    // read in RNG state
    
    read_rng(gen, saveFile);

    // read in Propagator elements to new object
    
    read_prop(new_prop, saveFile);

    // read in density matrices 

    read_rho(density_pointers, simData, saveFile);

    // read in Path vector elements to new array

    read_paths(newList, saveFile);

    // read in bath mode data

    read_modes(new_modes, simData, saveFile);

    // read in program data

    read_prog(simData, saveFile);

    // clean up

    fclose(saveFile);
}

/* ------------------------------------------------------------------------ */

// read_bath() -- helper function to read in bath trajectory and MC step
// data from binary save file

void read_bath(BathPtr & bath_pointers, FILE * saveFile)
{
    size_t bytes_read;
    char err_msg[FLEN];

    unsigned old_modes;

    bytes_read = fread(&old_modes, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {  
        sprintf(err_msg, "Error reading vector size\n");

        throw std::runtime_error(err_msg);
    }
    else if ( old_modes != static_cast<unsigned>(nmodes) )
    {
        sprintf(err_msg, "FATAL ERROR: previous nmodes value %u differs from current %u\n",
            old_modes, nmodes);

        throw std::runtime_error(err_msg);
    }

    // NOTE: Following should read results directly into correct
    // memory for all scopes to use. Should check that this happens
    // correctly, however!

    // read in xvals array

    bytes_read = fread(bath_pointers.xvals, sizeof(double), old_modes, saveFile);

    if (bytes_read != old_modes)
    {  
        sprintf(err_msg, "Error reading in bath xvals array\n");

        throw std::runtime_error(err_msg);
    }

    // read in pvals array

    bytes_read = fread(bath_pointers.pvals, sizeof(double), old_modes, saveFile);

    if (bytes_read != old_modes)
    {  
        sprintf(err_msg, "Error reading in bath pvals array\n");

        throw std::runtime_error(err_msg);
    }

    // read in x_step array

    bytes_read = fread(bath_pointers.x_step, sizeof(double), old_modes, saveFile);

    if (bytes_read != old_modes)
    {  
        sprintf(err_msg, "Error reading in bath x_step array\n");

        throw std::runtime_error(err_msg);
    }

    // read in p_step array

    bytes_read = fread(bath_pointers.p_step, sizeof(double), old_modes, saveFile);

    if (bytes_read != old_modes)
    {  
        sprintf(err_msg, "Error reading in bath p_step array\n");

        throw std::runtime_error(err_msg);
    }
}

/* ------------------------------------------------------------------------ */

// read_rng() -- helper function that uses GSL library calls to read in a
// saved rng state from a previous run. Assumption is that same generator is used
// between runs, otherwise reload is meaningless.

void read_rng(gsl_rng * gen, FILE * saveFile)
{
    gsl_rng_fread(saveFile, gen);
}

/* ------------------------------------------------------------------------ */

// read_prop() -- helper function to read in propagator data from binary
// backup file. Note that not all fields are read, since some are for
// temporary variables.

void read_prop(Propagator & new_prop, FILE * saveFile)
{
    size_t bytes_read;
    char err_msg[FLEN];

    // read in propagator array length
        
    unsigned propLen;

    bytes_read = fread(&propLen, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading propagator size\n");

        throw std::runtime_error(err_msg);
    }
    else if (propLen != DSTATES*DSTATES)
    {
        sprintf(err_msg, "FATAL ERROR: Old propagator length of %u not equal to current size of %u\n",
            propLen, DSTATES*DSTATES);

        throw std::runtime_error(err_msg);
    }

    // read in stored propagator array

    bytes_read = fread(new_prop.prop, sizeof(complex<double>), 
        propLen, saveFile);

    if (bytes_read != propLen)
    {
        sprintf(err_msg, "Error reading propagator array\n");

        throw std::runtime_error(err_msg);
    }

    // read in size of x0, p0 arrays

    unsigned ic_len;
    
    bytes_read = fread(&ic_len, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading propagator IC array size\n");

        throw std::runtime_error(err_msg);
    }

    // read x0 into array and copy to vector

    double * serial_ics = new double [ic_len];

    bytes_read = fread(serial_ics, sizeof(double), ic_len, saveFile);

    if (bytes_read != ic_len)
    {
        sprintf(err_msg, "Error reading propagator x0 array\n");

        throw std::runtime_error(err_msg);
    }

    //new_prop.x0_free.clear();
    //new_prop.x0_free.reserve(ic_len);

    for (unsigned i = 0; i < ic_len; i++)
        new_prop.x0_free.push_back(serial_ics[i]);

    // read p0 into array and copy to vector

    bytes_read = fread(serial_ics, sizeof(double), ic_len, saveFile);

    if (bytes_read != ic_len)
    {
        sprintf(err_msg, "Error reading propagator p0 array\n");

        throw std::runtime_error(err_msg);
    }

    //new_prop.p0_free.clear();
    //new_prop.p0_free.reserve(ic_len);

    for (unsigned i = 0; i < ic_len; i++)
        new_prop.p0_free.push_back(serial_ics[i]);

    delete [] serial_ics;
}

/* ------------------------------------------------------------------------ */

// read_rho() -- helper function to read out QCPI and EACP rho(t) data
// from binary backup file into current IC density matrix 

void read_rho(DensityPtr & density_pointers, SimInfo & simData, FILE * saveFile)
{
    size_t bytes_read;
    char err_msg[FLEN];

    // read in first array size

    int xLen;

    bytes_read = fread(&xLen, sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading rho(t) x-dimension size\n");

        throw std::runtime_error(err_msg);
    }
    else if (xLen != qm_steps)
    {
        sprintf(err_msg, 
            "Backup file step num. %u different from current value %u\n",
                xLen, qm_steps);

        throw std::runtime_error(err_msg);
    }

    // read in second array size

    int yLen;

    bytes_read = fread(&yLen, sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading rho(t) y-dimension size\n");

        throw std::runtime_error(err_msg);
    }
    else if (yLen != (DSTATES*DSTATES) )
    {
        sprintf(err_msg, 
            "Backup file DVR state num. %d different from current value %d\n",
                yLen, DSTATES*DSTATES);

        throw std::runtime_error(err_msg);
    }

    // define type to match bytes_read (to avoid warning below)

    unsigned arrayLen = static_cast<unsigned>(yLen);

    // loop over array dimensions and read in QCPI matrix

    complex<double> * rho_single = new complex<double> [yLen];

    for (int step = 0; step < xLen; step++)
    {
        // read in single rho(t) line

        bytes_read = fread(rho_single, sizeof(complex<double>), 
            yLen, saveFile);

        if (bytes_read != arrayLen)
        {
            sprintf(err_msg, "Error reading QCPI rho(t) block\n");

            throw std::runtime_error(err_msg);
        }

        for (int elem = 0; elem < yLen; elem++)
            density_pointers.rho_proc[step][elem] = rho_single[elem];            
    }

    // read in rho_ic_proc directly

    bytes_read = fread(density_pointers.rho_ic_proc, sizeof(complex<double>), 
        xLen, saveFile);

    if ( bytes_read != static_cast<unsigned>(xLen) )
    {
         sprintf(err_msg, "Error reading current IC rho(t) array\n");

         throw std::runtime_error(err_msg);
    }

    // loop over array dimensions and read in EACP matrix

    for (int step = 0; step < xLen; step++)
    {
        // read in single rho(t) line

        bytes_read = fread(rho_single, sizeof(complex<double>), 
            yLen, saveFile);

        if (bytes_read != arrayLen)
        {
            char err_msg[FLEN];
            sprintf(err_msg, "Error reading EACP rho(t) block\n");

            throw std::runtime_error(err_msg);
        }

        for (int elem = 0; elem < yLen; elem++)
            density_pointers.rho_eacp_proc[step][elem] = rho_single[elem];
    }

    // loop over array dimensions and read in current checkpoint matrix

    for (int step = 0; step < xLen; step++)
    {
        // read in single rho(t) line

        bytes_read = fread(rho_single, sizeof(complex<double>), 
            yLen, saveFile);

        if (bytes_read != arrayLen)
        {
            char err_msg[FLEN];
            sprintf(err_msg, "Error reading current checkpoint rho(t) block\n");

            throw std::runtime_error(err_msg);
        }

        for (int elem = 0; elem < yLen; elem++)
            density_pointers.rho_curr_checkpoint[step][elem] = rho_single[elem];
    }

    // loop over array dimensions and read in accumulated checkpoint matrix

    for (int step = 0; step < xLen; step++)
    {
        // read in single rho(t) line

        bytes_read = fread(rho_single, sizeof(complex<double>), 
            yLen, saveFile);

        if (bytes_read != arrayLen)
        {
            char err_msg[FLEN];
            sprintf(err_msg, "Error reading accumulated checkpoint rho(t) block\n");

            throw std::runtime_error(err_msg);
        }

        for (int elem = 0; elem < yLen; elem++)
            density_pointers.rho_full_checkpoint[step][elem] = rho_single[elem];
    }

    // read in list of previous reference states

    bytes_read = fread(density_pointers.old_ref_list, sizeof(Ref), qm_steps, saveFile);

    if (bytes_read != static_cast<unsigned>(qm_steps))
    {
        char err_msg[FLEN];
        sprintf(err_msg, "Error reading previous reference list\n");
        
        throw std::runtime_error(err_msg);
    }

    delete [] rho_single;
}

/* ------------------------------------------------------------------------ */

// read_paths() -- helper function read in serialized Path vector info and
// translate this back to a new Path array in memory

// Note that ic_vector can be left empty at this point, and all activity flags
// are set to true, because this is the boundary condition present before each
// write, and by extension, immediately after resuming segment loop on restart

void read_paths(vector<Path> & newList, FILE * saveFile)
{
    size_t bytes_read;
    char err_msg[FLEN];

    // read path list lengths

    unsigned len;

    bytes_read = fread(&len, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading Path vector length\n");

        throw std::runtime_error(err_msg);
    }

    unsigned path_len;

    bytes_read = fread(&path_len, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading fwd/bwd path length\n");

        throw std::runtime_error(err_msg);
    }
    else if ( path_len != static_cast<unsigned>(kmax) )
    {
        sprintf(err_msg, "FATAL ERROR: Old fwd/bwd length %u not compatible with current kmax %d\n",
            path_len, kmax);

        throw std::runtime_error(err_msg);
    }

    // read in length of bath IC list

    unsigned bath_len;

    bytes_read = fread(&bath_len, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading bath IC length for Path vector\n");

        throw std::runtime_error(err_msg);
    }

    unsigned * rho_path = new unsigned [path_len];

    double * bath_vals = new double [bath_len];

    // loop over path list to set elements

    Path * tempPath;

    for (unsigned path = 0; path < len; path++)
    {
        tempPath = new Path;

        // read in fwd path and assign

        bytes_read = fread(rho_path, sizeof(unsigned), path_len, saveFile);

        if (bytes_read != path_len)
        {
            sprintf(err_msg, "Error reading fwd path array\n");

            throw std::runtime_error(err_msg);
        }

        //tempPath->fwd_path.reserve(path_len);

        for (unsigned i = 0; i < path_len; i++)
            tempPath->fwd_path.push_back(rho_path[i]);   

        // read in bwd path and assign

        bytes_read = fread(rho_path, sizeof(unsigned), path_len, saveFile);

        if (bytes_read != path_len)
        {
            sprintf(err_msg, "Error reading bwd path array\n");

            throw std::runtime_error(err_msg);
        }

        //tempPath->bwd_path.reserve(path_len);

        for (unsigned i = 0; i < path_len; i++)
            tempPath->bwd_path.push_back(rho_path[i]);

        // read in rho and eacp products

        bytes_read = fread(&(tempPath->product), sizeof(complex<double>), 1,
            saveFile);

        if (bytes_read != 1)
        {
            sprintf(err_msg, "Error reading Path vector QCPI weight\n");

            throw std::runtime_error(err_msg);
        }

        bytes_read = fread(&(tempPath->eacp_prod), sizeof(complex<double>), 1,
            saveFile);

        if (bytes_read != 1)
        {
            sprintf(err_msg, "Error reading Path vector EACP weight\n");

            throw std::runtime_error(err_msg);
        }

        // read in x0 ICs

        bytes_read = fread(bath_vals, sizeof(double), bath_len, saveFile);

        if (bytes_read != bath_len)
        {
            sprintf(err_msg, "Error reading Path vector x0 vals\n");

            throw std::runtime_error(err_msg);
        }

        //tempPath->x0.reserve(bath_len);

        for (unsigned i = 0; i < bath_len; i++)
            tempPath->x0.push_back(bath_vals[i]);

        // read in p0 ICs

        bytes_read = fread(bath_vals, sizeof(double), bath_len, saveFile);

        if (bytes_read != bath_len)
        {
            sprintf(err_msg, "Error reading Path vector p0 vals\n");

            throw std::runtime_error(err_msg);
        }

        //tempPath->p0.reserve(bath_len);

        for (unsigned i = 0; i < bath_len; i++)
            tempPath->p0.push_back(bath_vals[i]);

        // set bool activity state

        tempPath->active = true;

        // push restored path element onto array

        newList.push_back(*tempPath);

        // delete temporary structure to avoid memory leak

        delete tempPath;

    } // end Path vector reading

    delete [] rho_path;
    delete [] bath_vals;
}

/* ------------------------------------------------------------------------ */

// read_modes() -- helper function to read in mode information tracked as
// part of classical trajectory propogation and action calculation

void read_modes(Mode * new_modes, SimInfo & simData, FILE * saveFile)
{
    size_t bytes_read;
    char err_msg[FLEN];

    // read in mode number

    unsigned bath_modes;

    bytes_read = fread(&bath_modes, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading bath mode number\n");

        throw std::runtime_error(err_msg);
    }
    else if ( bath_modes != static_cast<unsigned>(nmodes) )
    {
        sprintf(err_msg, 
            "Backup file mode number %u different from current value %d\n",
                bath_modes, nmodes);

        throw std::runtime_error(err_msg);
    }
    
    // read in x(t) list size

    unsigned list_len;

    bytes_read = fread(&list_len, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Error reading bath x(t) vector length\n");

        throw std::runtime_error(err_msg);
    }

    // allocate memory to read in serial x(t) vector

    double * xt_array = new double [list_len];

    // loop over modes and write out relevant data

    for (unsigned mode = 0; mode < bath_modes; mode++)
    {
        // read in x(t) values

        bytes_read = fread(xt_array, sizeof(double), list_len, saveFile);

        if (bytes_read != list_len)
        {
            sprintf(err_msg, "Error reading bath x(t) vector\n");

            throw std::runtime_error(err_msg);
        }

        //new_modes[mode].x_t.reserve(list_len);

        for (unsigned i = 0; i < list_len; i++)
            new_modes[mode].x_t.push_back(xt_array[i]);

        // read in frequency and coupling info

        bytes_read = fread(&(new_modes[mode].omega), sizeof(double), 1, saveFile);

        if (bytes_read != 1)
        {
            sprintf(err_msg, "Error reading in mode frequency\n");

            throw std::runtime_error(err_msg);
        }
        
        bytes_read = fread(&(new_modes[mode].c), sizeof(double), 1, saveFile);

        if (bytes_read != 1)
        {
            sprintf(err_msg, "Error reading in mode coupling\n");

            throw std::runtime_error(err_msg);
        }

    } // end mode loop

    delete [] xt_array;
}

/* ------------------------------------------------------------------------ */

// read_prog() -- helper function to read in error and loop index info
// into ProgramInfo structure from binary restart file

void read_prog(SimInfo & simData, FILE * saveFile)
{
    size_t bytes_read;
    char err_msg[FLEN];

    // read in loop indices

    bytes_read = fread(&(simData.ic_index), sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in IC loop index\n");

        throw std::runtime_error(err_msg);
    }

    bytes_read = fread(&(simData.tstep_index), sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in time segment index\n");

        throw std::runtime_error(err_msg);
    }

    // read in name of J(w)/w file as consistency check

    unsigned fname_size;
    char filename[FLEN];

    bytes_read = fread(&fname_size, sizeof(unsigned), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in input filename size\n");

        throw std::runtime_error(err_msg);
    }

    bytes_read = fread(filename, sizeof(char), fname_size, saveFile);

    if (bytes_read != fname_size)
    {
        sprintf(err_msg, "Could not read in input filename\n");

        throw std::runtime_error(err_msg);
    }
    else if (strcmp(filename, simData.infile) != 0)
    {
        sprintf(err_msg, "FATAL ERROR: Old spectral density file %s incompatible with new file %s\n",
            filename, simData.infile);

        throw std::runtime_error(err_msg);
    }

    // read in non-array parameters as consistency check

    // NOTE: Should double-valued comparisons use epsilon 
    // error vals so we have wiggle room?

    double old_asym;
    int old_ics;
    double old_dt;

    // read in old asymmetry

    bytes_read = fread(&old_asym, sizeof(double), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in asymmetry value\n");

        throw std::runtime_error(err_msg);
    }
    else if (old_asym != simData.asym)
    {
        sprintf(err_msg, "FATAL ERROR: Old asymmetry %.7e incompatible with new value %.7e\n",
            old_asym, simData.asym);

        throw std::runtime_error(err_msg);
    }

    // read in old IC number

    bytes_read = fread(&old_ics, sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in total IC number\n");

        throw std::runtime_error(err_msg);
    }
    else if (old_ics != simData.ic_tot)
    {
        sprintf(err_msg, "FATAL ERROR: Old IC number %d incompatible with new value %d\n",
            old_ics, simData.ic_tot);

        throw std::runtime_error(err_msg);
    }

    // read in old dt value

    bytes_read = fread(&old_dt, sizeof(double), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in old dt value\n");

        throw std::runtime_error(err_msg);
    }
    else if (old_dt != simData.dt)
    {
        sprintf(err_msg, "FATAL ERROR: Old timestep %.4f incompatible with new value %.4f\n",
            old_dt, simData.dt);

        throw std::runtime_error(err_msg);
    }

    // read in old block number

    int old_blocks;

    bytes_read = fread(&old_blocks, sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in old block number value\n");

        throw std::runtime_error(err_msg);
    }
    else if (old_blocks != simData.block_num)
    {
        sprintf(err_msg, "FATAL ERROR: Old block number %d incompatible with new value %d\n",
            old_blocks, simData.block_num);

        throw std::runtime_error(err_msg);
    }

    // read in SHIFT flag value

    int temp_flag;

    bytes_read = fread(&temp_flag, sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in bath shift flag\n");

        throw std::runtime_error(err_msg);
    }

    SHIFT = temp_flag;

    // read in ANALYTICFLAG value

    bytes_read = fread(&temp_flag, sizeof(int), 1, saveFile);

    if (bytes_read != 1)
    {
        sprintf(err_msg, "Could not read in analytic trajectory flag\n");

        throw std::runtime_error(err_msg);
    }

    ANALYTICFLAG = temp_flag;
}

/* ------------------------------------------------------------------------ */

// file_exists() -- simple function that uses POSIX filesystem I/O methods
// to check for file existence and returns this information as bool state

bool file_exists(char * filename)
{
    struct stat fileInfo;

    return (stat(filename, &fileInfo) == 0);
}

/* ------------------------------------------------------------------------ */
