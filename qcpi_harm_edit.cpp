/* This file implements the core QCPI algorithm, creating and
 * invoking other classes to perform the necessary functions.
 * It also defines a few functions of its own, for various
 * path-specific updates
 */

#include "harmonic.h"
#include "initial_bath.h"
#include "sim_info.h"
#include "propagator.h"

using namespace qcpiConstNS;

// QCPI per-path update functions

void qcpi_update(Path &, std::vector<Mode> &, SimInfo &);
double action_calc(Path &, std::vector<Mode> &, std::vector<Mode> &, 
    SimInfo &);

// mapping functions

void map_paths(map<unsigned long long, unsigned> &, 
    vector<Path> &);
unsigned long long string_to_nary(Path &);
unsigned long long string_to_nary(vector<unsigned> &, vector<unsigned> &);

// utility functions

void sum_paths(std::vector<Path> & pathList, cvector & rho, 
        Propagator & currProp, int step);
void global_mc_reduce(cvector & rhoLocal, cvector & rhoGlobal, 
        SimInfo & simData, MPI_Comm worldComm);
void print_results(FILE * outfile, SimInfo & simData, cvector & globalRho,
        std::string configFile, double globalRuntime, int nprocs);

int main(int argc, char * argv[])
{
    // initialize MPI

    MPI_Init(&argc, &argv);
    MPI_Comm worldComm = MPI_COMM_WORLD;

    int me, nprocs;

    MPI_Comm_rank(worldComm, &me);
    MPI_Comm_size(worldComm, &nprocs);

    // process command line args, format should be:
    //      mpirun -n <proc_num> ./prog_name <configFile>

    if (argc < 2)
    {
        std::string programName = argv[0];
        throw std::runtime_error("Usage: " + programName + " <configFile>\n");
    }

    std::string configFile = argv[1];   

    // assign simulation parameters from config file

    SimInfo simData;

    const char * delimiters = " \t\n";
    boost::char_separator<char> sep(delimiters);
    std::string emptyString = "";
    Tokenizer tok(emptyString, sep);

    simData.startup(configFile, tok);

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

    // begin timing this run

    double start = MPI_Wtime();

    // divide up ICs across procs
    
    if ((simData.icTotal % nprocs) != 0)
        throw std::runtime_error("Initial conditions must evenly distribute over procs\n");

    if (simData.icTotal < nprocs)
        throw std::runtime_error("Too few ICs for processor group\n");

    // initialize RNG

    gsl_rng * gen = gsl_rng_alloc(gsl_rng_mt19937);
    unsigned long seedVal = simData.seed * (me+1);
   
    // ensure rng seed is not zero
 
    if (seedVal == 0)
        seedVal = simData.seed + (me+1);

    gsl_rng_set(gen, seedVal);

    // prepare the harmonic bath modes

    InitialBath bath(simData.bathModes);

    bath.bath_setup(simData, tok, gen, me);

    // create propagator object

    Propagator currProp(simData.qmSteps);

    // allocate density matrix
    
    cvector rhoProc;

    rhoProc.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);

    // initialize path vector and create default path

    vector<Path> pathList;
    vector<Path> tempList;

    Path pstart;

    // path choice assumes donor-localized rho(0)

    pstart.fwdPath.push_back(0);  
    pstart.bwdPath.push_back(0); 
    pstart.product = 1.0;

    // initialize harmonic bath arrays

    std::vector<Mode> modes;
    std::vector<Mode> refModes;

    Mode currMode;

    for (int i = 0; i < simData.bathModes; i++)
    {
        currMode.omega = bath.bathFreq[i];
        currMode.c = bath.bathCoup[i];

        modes.push_back(currMode);
        refModes.push_back(currMode);
    }

    map<unsigned long long, unsigned> pathMap;

    int myICNum = simData.icTotal/nprocs;

    // Following loops are the core computational ones

    // Outer loop goes over initial conditions (selected by MC runs)

    // Inner loop goes over time points - for times less than
    // kmax, this does a full path sum, and for other points,
    // the iterative tensor propagation scheme is used

    for (int currIC = 0; currIC < myICNum; currIC++)
    {
        // zero per-proc rho amplitude

        currProp.qiAmp.assign(simData.qmSteps, 0.0);

        // generate ICs from MC walk 

        bath.ic_gen(gen, simData);

        // initialize bath coordinates and path vector

        pathList.clear();
        pathMap.clear();
        tempList.clear();

        pstart.x0.assign(bath.xVals.begin(), bath.xVals.end());
        pstart.p0.assign(bath.pVals.begin(), bath.pVals.end());

        pathList.push_back(pstart);

        // initialize propagator ICs

        currProp.xRef.assign(bath.xVals.begin(), bath.xVals.end());
        currProp.pRef.assign(bath.pVals.begin(), bath.pVals.end());

        // loop over first kmax time points, direct path summation

        for (int seg = 0; seg < simData.kmax; seg++)
        {
            // grow path list vector to include all branches

            for (unsigned path = 0; path < pathList.size(); path++)
            {
                Path temp = pathList[path];
        
                for (int fwd = 0; fwd < DSTATES; fwd++)
                {
                    for (int bwd = 0; bwd < DSTATES; bwd++)
                    {
                        Path incrPath(temp);
                        incrPath.fwdPath.push_back(fwd);
                        incrPath.bwdPath.push_back(bwd);

                        tempList.push_back(incrPath);
                    }
                }
            }

            pathList.clear();
            tempList.swap(pathList);
            tempList.clear();

            // select reference state for next timestep

            currProp.pick_ref(seg, gen);

            // run unforced trajectory and integrate U(t)

            currProp.update(refModes, simData);

            // update oscillators associated with each path and find actions

            for (unsigned path = 0; path < pathList.size(); path++)
            { 
                // calculate x(t) and p(t) at integration points

                qcpi_update(pathList[path], modes, simData);

                // use integration points to find new phase contribution

                double phi = 0.0;

                phi = action_calc(pathList[path], modes, refModes, simData);

                // calculate rho contribution for this path

                complex<double> kernelAmp = 
                    currProp.get_kernel_prod(pathList[path]);

                pathList[path].product *= kernelAmp * exp(I*phi);

            } 

            sum_paths(pathList, rhoProc, currProp, seg);

            tempList.clear();

        } 

        // slide paths forward one step, i.e. (010) -> (10)
        // this removes memory of localized rho(0)

        vector<unsigned> tPath;

        for (unsigned path = 0; path < pathList.size(); path++)
        {
            tPath.assign(pathList[path].fwdPath.begin()+1, 
                pathList[path].fwdPath.end());

            pathList[path].fwdPath.swap(tPath);

            tPath.assign(pathList[path].bwdPath.begin()+1, 
                pathList[path].bwdPath.end());

            pathList[path].bwdPath.swap(tPath);
        }

        // map paths to vector location

        map_paths(pathMap, pathList);

        // loop over time points beyond kmax and propagate iteratively

        for (int seg = simData.kmax; seg < simData.qmSteps; seg++)
        {
            unsigned fRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));
            unsigned bRand = static_cast<unsigned>(gsl_rng_uniform_int(gen,DSTATES));

            // choose reference state for this step

            currProp.pick_ref(seg, gen);

            // choose branching consistent with previous dynamics (DCSH)

            if (currProp.oldRefs[seg-simData.kmax] == REF_LEFT)
                fRand = bRand = 0;
            else
                fRand = bRand = 1;

            // integrate unforced equations and find U(t)

            currProp.update(refModes, simData);

            // set up tempList to hold iteration intermediates

            tempList.clear();
            tempList = pathList;

            for (unsigned tp = 0; tp < tempList.size(); tp++)
                tempList[tp].product = 0.0;

            // loop over paths to construct tensor contributions

            unsigned currSize = pathList.size();
    
            for (unsigned path = 0; path < currSize; path++)
            {
                complex<double> tensorProd;

                // loop over all pairs of fwd/bwd system states
                // to generate next path extension

                for (int fwd = 0; fwd < DSTATES; fwd++)
                {
                    for (int bwd = 0; bwd < DSTATES; bwd++)
                    {
                        // temporarily update paths for calculations
                        // path length should now be (kmax+1)

                        Path temp;
                        temp = pathList[path];

                        temp.fwdPath.push_back(fwd);
                        temp.bwdPath.push_back(bwd);

                        // calculate x(t) and p(t) at integration points

                        qcpi_update(temp, modes, simData);

                        // use integration points to find new phase contribution

                        double phi;

                        phi = action_calc(temp, modes, refModes, simData);

                        // evaluate tensor element

                        complex<double> kernelAmp = 
                            currProp.get_kernel_prod(temp);

                        tensorProd = kernelAmp * exp(I*phi);

                        // add tensor result to correct path via index calcs
                        // note that this index comes from the last kmax 
                        // entries of our current kmax+1 list

                        vector<unsigned> ftemp, btemp;

                        ftemp.assign(temp.fwdPath.begin()+1, 
                            temp.fwdPath.end() );

                        btemp.assign(temp.bwdPath.begin()+1, 
                            temp.bwdPath.end() );

                        // find path dictated by tensor update in array

                        unsigned long long target = string_to_nary(ftemp, btemp);

                        unsigned outPath = pathMap[target];

                        // add tensor product to correct element

                        tempList[outPath].product += tensorProd * temp.product;
        
                        // update ICs if we have correct path element

                        if (temp.fwdPath[0] == fRand && temp.bwdPath[0] == bRand)
                        {
                            tempList[outPath].x0.assign(temp.x0.begin(), temp.x0.end());
                            tempList[outPath].p0.assign(temp.p0.begin(), temp.p0.end());
                        }

                        // pop off previous update (returning length to kmax)

                        temp.fwdPath.pop_back();
                        temp.bwdPath.pop_back();

                    } // end bwd loop

                } // end fwd loop

            } // end path loop (iter. phase)    

            // swap tempList and pathList

            pathList.swap(tempList);

            // pull out current density matrix

            sum_paths(pathList, rhoProc, currProp, seg);

        } // end seg loop (iter. phase)

    } // end IC loop

    // collect timing data

    double localRuntime = MPI_Wtime() - start;
    double globalRuntime = 0.0;

    MPI_Allreduce(&localRuntime, &globalRuntime, 1, MPI_DOUBLE,
        MPI_MAX, worldComm);

    // reduce rho(t) across processors

    cvector globalRho;

    globalRho.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);

    global_mc_reduce(rhoProc, globalRho, simData, worldComm);

    // output summary

    if (me == 0)
        print_results(outfile, simData, globalRho, configFile,
            globalRuntime, nprocs);

    // cleanup

    if (me == 0)
        fclose(outfile);

    MPI_Finalize();

    return 0;
}

/* ------------------------------------------------------------------------ */

// qcpi_update() evolves the harmonic oscillator modes based on the system
// states encoded in the current path list

void qcpi_update(Path & qmPath, std::vector<Mode> & mlist, 
        SimInfo & simData)
{
    double delta = simData.dt/2.0;
    double dvrVals[DSTATES] = {dvrLeft, dvrRight};

    for (int mode = 0; mode < simData.bathModes; mode++)
    {
        // set up ICs for first half of path

        double w = mlist[mode].omega;
        double c = mlist[mode].c;

        double x0, xt;
        double p0, pt;

        x0 = qmPath.x0[mode];
        p0 = qmPath.p0[mode];

        unsigned size = qmPath.fwdPath.size();

        unsigned splus = qmPath.fwdPath[size-2];
        unsigned sminus = qmPath.bwdPath[size-2];

        double shift = (dvrVals[splus] + dvrVals[sminus])/2.0;

        shift *= c/(mass*w*w);

        mlist[mode].xt.clear();

        // calculate time evolution for first
        // half of path

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) +
            shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].phase1 = (x0 - shift)*sin(w*delta) -
            (p0/(mass*w))*(cos(w*delta) - 1.0) + shift*w*delta;

        // swap x0, p0 with xt, pt

        x0 = xt;
        p0 = pt;

        // find s vals and shift for second half of trajectory

        splus = qmPath.fwdPath[size-1];
        sminus = qmPath.bwdPath[size-1];

        shift = (dvrVals[splus] + dvrVals[sminus])/2.0;

        shift *= c/(mass*w*w);

        // calculate time evolution for second
        // half of path

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) +
            shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].phase2 = (x0 - shift)*sin(w*delta) -
            (p0/(mass*w))*(cos(w*delta) - 1.0) + shift*w*delta;

        // update current phase space point

        qmPath.x0[mode] = xt;
        qmPath.p0[mode] = pt;

    } // end mode loop
}

/* ------------------------------------------------------------------------- */

// action_calc() computes the residual action contributed by each path to the
// summation after accounting for the action of the bath reference trajectory

double action_calc(Path & qmPath, std::vector<Mode> & mlist, 
        std::vector<Mode> & reflist, SimInfo & simData)
{
    double dvrVals[DSTATES] = {dvrLeft, dvrRight};

    // loop over modes and integrate their action contribution

    double action = 0.0;

    for (int mode = 0; mode < simData.bathModes; mode++)
    {
        double sum = 0.0;

        // set indices to correct half steps

        unsigned size = qmPath.fwdPath.size();

        unsigned splus = qmPath.fwdPath[size-2];
        unsigned sminus = qmPath.bwdPath[size-2];

        double ds = dvrVals[splus] - dvrVals[sminus];
        double pre = (mlist[mode].c * ds)/mlist[mode].omega;

        // find first half of action sum

        sum += pre * (mlist[mode].phase1 - 
            reflist[mode].phase1);

        // recalculate prefactor

        splus = qmPath.fwdPath[size-1];
        sminus = qmPath.bwdPath[size-1];

        ds = dvrVals[splus] - dvrVals[sminus];
        pre = (mlist[mode].c * ds)/mlist[mode].omega;

        // find second half of action sum

        sum += pre * (mlist[mode].phase2 - 
            reflist[mode].phase2);

        action += sum;

    } // end mode loop

    return action;
}

/* ------------------------------------------------------------------------ */

// sum_paths() is a helper function which adds the contribution of each
// path amplitude to the total rho(t) value at each time point

void sum_paths(std::vector<Path> & pathList, cvector & rho, 
        Propagator & currProp, int step)
{
    if (pathList.begin() == pathList.end())
        return;

    unsigned size = pathList[0].fwdPath.size();

    for (unsigned path = 0; path < pathList.size(); path++)
    {
        unsigned splus1 = pathList[path].fwdPath[size-1];
        unsigned sminus1 = pathList[path].bwdPath[size-1];

        unsigned rindex = splus1*DSTATES + sminus1;

        rho[step*DSTATES*DSTATES + rindex] += pathList[path].product;

        if (rindex == 0)
            currProp.qiAmp[step] += pathList[path].product;
    }

}

/* ------------------------------------------------------------------------ */

// map_paths() uses numerical value of the binary string representing each
// system path to generate a mapping of paths to array positions which
// should be unique

void map_paths(map<unsigned long long, unsigned> & pathMap, 
    vector<Path> & pathList)
{
    unsigned long long binVal;

    for (unsigned path = 0; path < pathList.size(); path++)
    {
        binVal = string_to_nary(pathList[path]);

        pathMap[binVal] = path;
    }
}

/* ------------------------------------------------------------------------ */

// string_to_nary() returns the decimal value of the binary string representing
// a single system path (fwd and bwd)

unsigned long long string_to_nary(Path & entry)
{
    unsigned long long sum = 0;
    unsigned long long pre = 1;

    // note that ordering below makes fwd path most significant bits
    // also, assumes binary number is in order of normal printing of fwd/bwd
    // i.e. {010,110} -> 010110 -> 22

    if (entry.fwdPath.size() == 0 || entry.bwdPath.size() == 0)
    {
        std::string err_msg = 
            "ERROR: Null fwd/bwd vectors encountered in string_to_nary()\n";
        throw std::runtime_error(err_msg);
    }

    // backward path forms least significant bit set

    for (unsigned pos = entry.bwdPath.size() - 1; pos >= 0; pos--)
    {
        sum += entry.bwdPath[pos] * pre;
        pre *= DSTATES;

        if (pos == 0)
            break;
    }

    // forward path corresponds to most significant bits

    for (unsigned pos = entry.fwdPath.size() - 1; pos >= 0; pos--)
    {
        sum += entry.fwdPath[pos] * pre;
        pre *= DSTATES;

        if (pos == 0)
            break;
    }

    return sum;
}

/* ------------------------------------------------------------------------ */

// string_to_nary(vector<unsigned> &, vector<unsigned> &) is an overloaded
// version of the string_to_nary() function, for use in tensor multiplication

unsigned long long string_to_nary(vector<unsigned> & fwd, vector<unsigned> & bwd)
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

// global_mc_reduce() is a helper function which handles the MPI calls needed
// to add up the Monte Carlo contributions to rho(t) from each processor and
// normalize them. It is somewhat complicated by the lack of a natural complex
// type for MPI reduction.

void global_mc_reduce(cvector & rhoLocal, cvector & rhoGlobal, 
        SimInfo & simData, MPI_Comm worldComm)
{
    std::vector<double> rhoRealProc;
    std::vector<double> rhoImagProc;

    std::vector<double> rhoReal;
    std::vector<double> rhoImag;

    rhoRealProc.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);
    rhoImagProc.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);
    rhoReal.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);
    rhoImag.assign(DSTATES*DSTATES*simData.qmSteps, 0.0);

    for (int i = 0; i < simData.qmSteps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rhoRealProc[i*DSTATES*DSTATES+j] = 
                rhoLocal[i*DSTATES*DSTATES+j].real();

            rhoImagProc[i*DSTATES*DSTATES+j] = 
                rhoLocal[i*DSTATES*DSTATES+j].imag();
        }
    }

    // Allreduce the real and imaginary arrays

    MPI_Allreduce(&rhoRealProc[0], &rhoReal[0], simData.qmSteps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, worldComm);

    MPI_Allreduce(&rhoImagProc[0], &rhoImag[0], simData.qmSteps*DSTATES*DSTATES,
        MPI_DOUBLE, MPI_SUM, worldComm);

    // scale arrays by Monte Carlo factor

    for (int i = 0; i < simData.qmSteps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            rhoReal[i*DSTATES*DSTATES+j] /= simData.icTotal;
            rhoImag[i*DSTATES*DSTATES+j] /= simData.icTotal;
        }
    }

    // reassign to complex output

    for (int i = 0; i < simData.qmSteps; i++)
    {
        for (int j = 0; j < DSTATES*DSTATES; j++)
        {
            int index = i*DSTATES*DSTATES + j;

            rhoGlobal[index] = rhoReal[index] + I*rhoImag[index];
        }
    }

}

/* ------------------------------------------------------------------------ */

// print_results() is a helper function which summarizes the simulation and
// presents the final rho(t) values for plotting and analysis in simple
// column-wise format

void print_results(FILE * outfile, SimInfo & simData, cvector & globalRho,
        std::string configFile, double globalRuntime, int nprocs)
{
        int repeat = 50;
        
        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nSimulation Summary\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

        fprintf(outfile, "Processors: %d\n", nprocs);
        fprintf(outfile, "Total simulation time: %.3f min\n", globalRuntime/60.0);
        fprintf(outfile, "Configuration file: %s\n\n", configFile.c_str());

        simData.print_vars(outfile, repeat);

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\nDensity Matrix Values\n");

        for (int i = 0; i < repeat; i++)
            fprintf(outfile, "-");

        fprintf(outfile, "\n\n");

       
            for (int i = 0; i < simData.qmSteps; i++)
            {
                int entry = i*DSTATES*DSTATES;
                double trace = globalRho[entry].real()+globalRho[entry+3].real(); 

                fprintf(outfile, "%7.4f %8.5f %6.3f (Tr = %13.10f)\n", (i+1)*simData.dt, 
                    globalRho[entry].real(), 0.0, trace);
            }


        fprintf(outfile, "\n");

}

/* ------------------------------------------------------------------------ */
