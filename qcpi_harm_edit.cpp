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
#include "propagator.h"

using namespace qcpiConstNS;

// QCPI functions
void qcpi_update_exact(Path &, std::vector<Mode> &, SimInfo &);
double action_calc_exact(Path &, std::vector<Mode> &, std::vector<Mode> &, 
    SimInfo &);

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

    Propagator curr_prop(simData.qmSteps);

    // EDIT NOTES: (consider using small object/struct here)

    // allocate density matrix
    
    cvector rho_proc;

    rho_proc.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);

    // initialize path vector

    vector<Path> pathList;
    vector<Path> tempList;

    Path pstart;

    pstart.fwd_path.push_back(0);   // left-localized
    pstart.bwd_path.push_back(0);   // left-localized
    pstart.product = 1.0;

    // initialize harmonic bath arrays

    std::vector<Mode> modes;
    std::vector<Mode> ref_modes;

    Mode currMode;

    for (int i = 0; i < simData.bathModes; i++)
    {
        currMode.omega = bath.bathFreq[i];
        currMode.c = bath.bathCoup[i];

        modes.push_back(currMode);
        ref_modes.push_back(currMode);
    }

    map<unsigned long long, unsigned> pathMap;

    int my_ics = simData.icTotal/nprocs;

    // Following loops are the core computational ones

    // Outer loop goes over initial conditions (selected by MC runs)

    // Inner loop goes over time points - for times less than
    // kmax, this does a full path sum, and for other points,
    // the iterative tensor propagation scheme is used

    for (int ic_curr = 0; ic_curr < my_ics; ic_curr++)
    {
        // zero per-proc rho(t)

        curr_prop.qiAmp.assign(simData.qmSteps, 0.0);

        // generate ICs for HOs using MC walk (we use last step's
        // ICs as current step's IC seed)

        bath.ic_gen(gen, simData);

        // for time points < kmax

        // loop over kmax segments of fwd/bwd paths

        pathList.clear();
        pathMap.clear();
        tempList.clear();

        pstart.x0.assign(bath.xVals.begin(), bath.xVals.end());
        pstart.p0.assign(bath.pVals.begin(), bath.pVals.end());

        pathList.push_back(pstart);

        // initialize propagator ICs

        curr_prop.xRef.assign(bath.xVals.begin(), bath.xVals.end());
        curr_prop.pRef.assign(bath.pVals.begin(), bath.pVals.end());

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
                        Path incr_path(temp);
                        incr_path.fwd_path.push_back(fwd);
                        incr_path.bwd_path.push_back(bwd);

                        tempList.push_back(incr_path);
                    }
                }
            }

            pathList.clear();
            tempList.swap(pathList);
            tempList.clear();

            // run unforced trajectory and integrate U(t)

            // select reference state for next timestep

            curr_prop.pick_ref(seg, gen);

            curr_prop.update(ref_modes, simData);

            // loop over all paths at this time point

            for (unsigned path = 0; path < pathList.size(); path++)
            { 
                // calculate x(t) and p(t) at integration points
                // along all paths

                qcpi_update_exact(pathList[path], modes, simData);

                // use integration points to find new phase contribution

                double phi = 0.0;

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

                rho_proc[seg*DSTATES*DSTATES + rindex] += pathList[path].product;
                
                if (rindex == 0)
                    curr_prop.qiAmp[seg] += pathList[path].product;

            } // end path loop (full path phase)

            tempList.clear();

        } // end seg loop (full path phase)

        // EDIT NOTES: (be more explicit about how this works?)

        // slide paths forward one step, i.e. (010) -> (10)
        // note that our system IC means we only
        // have 1/4 of all paths, or this would // need to be handled with full T matrix as below

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

            curr_prop.pick_ref(seg, gen);

            // choose branching kmax steps back

            if (curr_prop.oldRefs[seg-simData.kmax] == REF_LEFT)
                fRand = bRand = 0;
            else
                fRand = bRand = 1;

            // integrate unforced equations and find U(t)

            curr_prop.update(ref_modes, simData);

            // set up tempList to hold our matrix mult.
            // intermediates

            tempList.clear();
            tempList = pathList;

            for (unsigned tp = 0; tp < tempList.size(); tp++)
                tempList[tp].product = 0.0;

            // loop over paths to construct tensor contributions
            // in this loop, path variable indexes the input path array
            // since this contains data used to extend path one step

            // fix size of path so any additions aren't double-counted            

            unsigned currSize = pathList.size();
    
            for (unsigned path = 0; path < currSize; path++)
            {
                complex<double> tensorProd;

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

                rho_proc[seg*DSTATES*DSTATES + rindex] += pathList[path].product;

                if (rindex == 0)
                    curr_prop.qiAmp[seg] += pathList[path].product;

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

    std::vector<double> rho_real_proc;
    std::vector<double> rho_imag_proc;

    std::vector<double> rho_real;
    std::vector<double> rho_imag;

    rho_real_proc.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);
    rho_imag_proc.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);
    rho_real.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);
    rho_imag.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);

    for (int i = 0; i < simData.qmSteps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rho_real_proc[i*DSTATES*DSTATES+j] = rho_proc[i*DSTATES*DSTATES+j].real();
            rho_imag_proc[i*DSTATES*DSTATES+j] = rho_proc[i*DSTATES*DSTATES+j].imag();
        }
    }

    // Allreduce the real and imaginary arrays

    MPI_Allreduce(&rho_real_proc[0], &rho_real[0], simData.qmSteps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, w_comm);

    MPI_Allreduce(&rho_imag_proc[0], &rho_imag[0], simData.qmSteps*DSTATES*DSTATES,
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

    MPI_Finalize();

    return 0;
}

/* ------------------------------------------------------------------------ */

void qcpi_update_exact(Path & qm_path, std::vector<Mode> & mlist, 
        SimInfo & simData)
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

double action_calc_exact(Path & qm_path, std::vector<Mode> & mlist, 
        std::vector<Mode> & reflist, SimInfo & simData)
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
