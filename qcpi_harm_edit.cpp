/*  qcpi_harmonic.cpp - a program to read in
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

#include "harmonic.h"
#include "initial_bath.h"
#include "sim_info.h"

using namespace qcpiConstNS;

// propagator integration
void ho_update_exact(Propagator &, Mode *, double, SimInfo &);
void build_ham(Propagator &, Mode *, int, SimInfo &);

// Need to remove NR ODE functions and reimplement
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

    std::string config_file = argv[1];   

    // create structs to hold configuration parameters

    SimInfo simData;

    const char * delimiters = " \t\n";
    boost::char_separator<char> sep(delimiters);
    std::string emptyString = "";
    Tokenizer tok(emptyString, sep);

    // assign values to config. vars

    simData.startup(config_file, tok);

    if (simData.icTotal < nprocs)
        throw std::runtime_error("Too few ICs for processor number\n");

    // open output file for I/O on single proc, opening early to 
    // prevent I/O errors from terminating run

    FILE * outfile;

    if (me == 0)
    {
        outfile = fopen(simData.outputName.c_str(), "w");

        if (outfile == NULL)
        {
            throw std::runtime_error("Could not open file " + 
                    simData.outputName + "\n");
        }
    }

    // begin timing calculation

    double start = MPI_Wtime();

    // divide up ICs across procs
    
    if ((simData.icTotal % nprocs) != 0)
        throw std::runtime_error("Initial conditions must evenly distribute over procs\n");

    if (simData.icTotal < nprocs)
        throw std::runtime_error("Too few ICs for processor group\n");

    int my_ics = simData.icTotal/nprocs;

    // initialize RNG

    gsl_rng * gen = gsl_rng_alloc(gsl_rng_mt19937);
    unsigned long s_val = simData.seed * (me+1);
   
    // ensure rng seed is not zero
 
    if (s_val == 0)
        s_val = simData.seed + (me+1);

    gsl_rng_set(gen, s_val);

    // prepare the harmonic bath modes

    InitialBath bath(simData.bathModes);

    bath.bath_setup(simData, tok, gen, me);

    // create propagator object

    Propagator curr_prop;

    curr_prop.prop = new complex<double> [DSTATES*DSTATES];
    curr_prop.ptemp = new complex<double> [DSTATES*DSTATES];
    curr_prop.ham = new complex<double> [DSTATES*DSTATES];

    // EDIT NOTES: (consider using small object/struct here)

    // allocate density matrix

    complex<double> ** rho_proc = new complex<double> * [simData.qmSteps];
    complex<double> * rho_ic_proc = new complex<double> [simData.qmSteps];

    for (int i = 0; i < simData.qmSteps; i++)
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

    Ref * old_ref_list = new Ref [simData.qmSteps];

    old_ref_list[0] = REF_LEFT;

    // initialize harmonic bath arrays

    Mode * modes = new Mode [simData.bathModes];
    Mode * ref_modes = new Mode [simData.bathModes];

    for (int i = 0; i < simData.bathModes; i++)
    {
        modes[i].omega = ref_modes[i].omega = bath.bathFreq[i];
        modes[i].c = ref_modes[i].c = bath.bathCoup[i];
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

        for (int i = 0; i < simData.qmSteps; i++)
            rho_ic_proc[i] = 0.0;

        // generate ICs for HOs using MC walk (we use last step's
        // ICs as current step's IC seed)

        bath.ic_gen(gen, simData);

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
        pstart.x0.assign(bath.xVals.begin(), bath.xVals.end());
        pstart.p0.assign(bath.pVals.begin(), bath.pVals.end());

        pathList.push_back(pstart);

        // initialize propagator ICs

        curr_prop.x0_free.assign(bath.xVals.begin(), bath.xVals.end());
        curr_prop.p0_free.assign(bath.pVals.begin(), bath.pVals.end());

        // loop over first kmax time points

        for (int seg = 0; seg < simData.kmax; seg++)
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

            for (int chunk_num = 0; chunk_num < simData.chunks; chunk_num++)
            {
                // construct H(x,p) from bath configuration
            
                build_ham(curr_prop, ref_modes, chunk_num, simData);

                // integrate TDSE for U(t) w/ piece-wise constant
                // Hamiltonian approx.

                rkdriver(curr_prop.prop, curr_prop.ptemp, DSTATES*DSTATES, 
                    0.0, simData.rhoDelta, simData.rhoSteps, prop_eqns, curr_prop.ham);

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

        for (int seg = simData.kmax; seg < simData.qmSteps; seg++)
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

                if (old_ref_list[seg-simData.kmax] == REF_LEFT)
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

            for (int chunk_num = 0; chunk_num < simData.chunks; chunk_num++)
            {
                // construct H(x,p) from bath configuration
            
                build_ham(curr_prop, ref_modes, chunk_num, simData);

                // integrate TDSE for U(t) w/ piece-wise constant
                // Hamiltonian approx.

                rkdriver(curr_prop.prop, curr_prop.ptemp, DSTATES*DSTATES, 
                    0.0, simData.rhoDelta, simData.rhoSteps, prop_eqns, curr_prop.ham);

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

    double * rho_real_proc = new double [simData.qmSteps*DSTATES*DSTATES];
    double * rho_imag_proc = new double [simData.qmSteps*DSTATES*DSTATES];

    double * rho_real = new double [simData.qmSteps*DSTATES*DSTATES];
    double * rho_imag = new double [simData.qmSteps*DSTATES*DSTATES];

    for (int i = 0; i < simData.qmSteps; i++)
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

    MPI_Allreduce(rho_real_proc, rho_real, simData.qmSteps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    MPI_Allreduce(rho_imag_proc, rho_imag, simData.qmSteps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    // scale arrays by Monte Carlo factor

    for (int i = 0; i < simData.qmSteps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real[i*DSTATES*DSTATES+j] /= simData.icTotal;
            rho_imag[i*DSTATES*DSTATES+j] /= simData.icTotal;
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

        fprintf(outfile, "Processors: %d\n", nprocs);
        fprintf(outfile, "Total simulation time: %.3f min\n", g_runtime/60.0);
        fprintf(outfile, "Configuration file: %s\n\n", config_file.c_str());

        simData.print(outfile, repeat);

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nDensity Matrix Values\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

       
            for (int i = 0; i < simData.qmSteps; i++)
            {
                int entry = i*DSTATES*DSTATES;

                fprintf(outfile, "%7.4f %8.5f %6.3f (Tr = %13.10f)\n", (i+1)*simData.dt, 
                    rho_real[entry], 0.0, rho_real[entry]+rho_real[entry+3]);
            }


        fprintf(outfile, "\n");

      } // end output conditional

    // cleanup


    delete [] curr_prop.prop;
    delete [] curr_prop.ptemp;
    delete [] curr_prop.ham;

    for (int i = 0; i < simData.qmSteps; i++)
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

/* ------------------------------------------------------------------------ */

void ho_update_exact(Propagator & prop, Mode * mlist, double ref_state, 
        SimInfo & simData)
{
    double del_t = simData.dt/2.0;
    double chunk_dt = simData.dt/simData.chunks;

    for (int mode = 0; mode < simData.bathModes; mode++)
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
        
        for (int i = 0; i < simData.chunks; i++)
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

    for (int i = 0; i < simData.bathModes; i++)
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

    for (int mode = 0; mode < simData.bathModes; mode++)
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

    for (int mode = 0; mode < simData.bathModes; mode++)
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
