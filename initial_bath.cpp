/* Implementation file for the InitialBath class, which stores 
 * information used to generate samples of the bath phase space
 * for QCPI averaging.
 */

#include "initial_bath.h"

using namespace qcpiConstNS;

/* ------------------------------------------------------------------------- */

InitialBath::InitialBath(unsigned modes)
{
    numModes = modes;

    xStep.assign(numModes, 0.0);
    pStep.assign(numModes, 0.0);

    xVals.assign(numModes, 0.0);
    pVals.assign(numModes, 0.0);

    bathFreq.assign(numModes, 0.0);
    bathCoup.assign(numModes, 0.0);
}

/* ------------------------------------------------------------------------- */

// read_spec() parses a discrete spectral density file to initialize the
// frequencies and couplings of the bath oscillators. The file is assumed to
// be in the format of whitespace separated frequency and J(w)/w values.

void InitialBath::read_spec(std::string specName, std::vector<double> & omega, 
    std::vector<double> & jvals, Tokenizer & tok)
{
    ifstream specFile;
    std::string buffer;
    std::string entry;
    const char commentChar = '#';

    specFile.open(specName.c_str(), std::ios_base::in);

    if (!specFile.is_open())
        throw std::runtime_error("Could not open spectral density file\n");

    // read in file line-by-line

    Tokenizer::iterator iter;

    while (getline(specFile, buffer))
    {   
        tok.assign(buffer);

        iter = tok.begin();

        // skip empty lines and comments

        if (tok.begin() == tok.end())
            continue;

        entry = *iter;

        if (entry[0] == commentChar)
            continue;

        // first column is frequency

        omega.push_back(boost::lexical_cast<double>(entry));

        // check for bad entries

        ++iter;

        if (iter == tok.end())
            throw std::runtime_error("Incomplete line in spectral density file\n");

        entry = *iter;

        // second column is J(w)/w

        jvals.push_back(boost::lexical_cast<double>(entry)); 

    } 

    specFile.close();
}

/* ------------------------------------------------------------------------- */

// process_step() performs basic Monte Carlo updates on the bath positions and
// momenta, moving through phase space one dimension at a time, as specified
// by the index variable.

int InitialBath::process_step(unsigned index, double beta, gsl_rng * gen)
{
    double prob = dist(index, beta);
    int accepted = 0;

    if (prob >= 1.0) 
    {
        // accept step and update state

        xOld = xNew;     
        pOld = pNew;
        
        xPick = xNew;
        pPick = pNew;

        accepted++;
    }
    else
    {    
        double xi = gsl_rng_uniform(gen);
    
        if (prob >= xi) 
        {    
            // accept on coin flip and update state

            xOld = xNew;     
            pOld = pNew;

            xPick = xNew;
            pPick = pNew;

            accepted++;  
        }
        else
        {
            // reject step

            xPick = xOld;
            pPick = pOld;
        }
    }

    return accepted;
}

/* ------------------------------------------------------------------------- */

// dist() calculates the ratio of Wigner distribution values for a trial
// step in the Monte Carlo walk

double InitialBath::dist(unsigned index, double beta)
{
    double omega = bathFreq[index];
    double coup = bathCoup[index];

    double f = hbar*omega*beta;

    // shift the bath coordinates to reflect system-bath equilibrium

    double lambda = dvrLeft * coup * ( 1.0/(mass*omega*omega) );

    double xNewShift = xNew - lambda;
    double xOldShift = xOld - lambda;

    // calculate Wigner distribution ratio for these x,p vals

    double delta = 
        (mass*omega/hbar)*(xNewShift*xNewShift - xOldShift*xOldShift) + 
            (1.0/(mass*omega*hbar))*(pNew*pNew - pOld*pOld);

    double pre = tanh(f/2.0);

    return exp(-pre*delta);
}

/* ------------------------------------------------------------------------- */

// bath_setup() initializes the bath frequencies and coupling constants based
// on a logarithmic discretization of the input spectral density (assumed to
// be in the form of J(w)/w)

void InitialBath::bath_setup(SimInfo & simData, Tokenizer & tok, gsl_rng * gen, 
        int myRank)
{
    // read in spectral density from file

    std::vector<double> omega;
    std::vector<double> jvals;

    read_spec(simData.inputName, omega, jvals, tok);

    // first calculate reorg energy via simple integration
    // assumes uniform spacing in frequency

    double dw = omega[1] - omega[0];    
    double sum = 0.0;

    sum += (jvals.front() + jvals.back())/2.0;

    for (unsigned i = 1; i < jvals.size()-1; i++)
        sum += jvals[i];

    sum *= dw;

    double w0 = sum/numModes;

    // report reorg energy to command line as simple error check

    double pi = acos(-1.0);
    double reorg = 4.0 * sum / pi;

    if (myRank == 0)
        fprintf(stdout, "Reorg. energy (au): %.7f\n", reorg);

    // discretize frequencies so that normalized integral up to
    // w[i] equals i

    for (unsigned j = 0; j < numModes; j++)
    {
        sum = 0.0;
        long i = -1;
    
        while (sum <= (j+1) && i < static_cast<long>(jvals.size()-2))
        {
            i++;
            sum += jvals[i]*dw/w0;
        }

        bathFreq[j] = omega[i];
    }

    // check modes for 0 vals (in case input file is sparse)

    for (unsigned i = 0; i < numModes; i++)
    {
        if (bathFreq[i] == 0)
        {
            for (unsigned j = i; j < numModes; j++)
            {
                if (bathFreq[j] != 0)
                {
                    bathFreq[i] = bathFreq[j]/2.0;    
                    break;
                }
            }

            if (bathFreq[i] == 0)
                throw std::runtime_error("All modes have zero frequency\n");
        }
    }

    // calculate couplings from bath frequencies

    for (unsigned i = 0; i < numModes; i++)
        bathCoup[i] = sqrt(2.0*w0/pi) * bathFreq[i];

    // determine optimal step sizes for MC walk

    calibrate_mc(gen, simData);
}

/* ------------------------------------------------------------------------- */

// calibrate_mc() is a helper function which determines optimal phase space
// MC steps for each bath oscillator by trial and error

void InitialBath::calibrate_mc(gsl_rng * gen, SimInfo & simData)
{
    // set maximum trial number to bound initialization time

    const double maxTrials = 1000;

    const double baseStep = 10.0;
    const double scale = 0.8;

    // establish bounds of acceptance ratio on optimization

    const double lowThresh = 0.45;
    const double highThresh = 0.55;

    const int iterCount = 1000;
    std::vector<double> xCurr;
    std::vector<double> pCurr;

    xCurr.assign(numModes, 0.0);
    pCurr.assign(numModes, 0.0);

    // initialize step sizes and (x,p) at minimum of coupled system-bath

    for (unsigned i = 0; i < numModes; i++)
    {
        xStep[i] = gsl_rng_uniform(gen) * baseStep;
        pStep[i] = gsl_rng_uniform(gen) * baseStep;

        double pos = 
            bathCoup[i]*( dvrLeft/(mass*bathFreq[i]*bathFreq[i]) );

        xCurr[i] = pos;
        pCurr[i] = 0.0;
    }

    int accepted;

    // run MC trials, modifying steps in each dimension independently

    for (unsigned i = 0; i < numModes; i++)
    {
        int count = 0;

        while (count <= maxTrials)    
        {
            xOld = xCurr[i];
            pOld = pCurr[i];
            accepted = 1;

            for (int j = 0; j < iterCount; j++)
            {
                double stepLen = gsl_rng_uniform(gen) * xStep[i];

                // displace x coord.

                xNew = xOld + pow(-1.0, gsl_rng_uniform_int(gen,2))*stepLen;

                // keep p constant for this run

                pNew = pOld;

                int incr = process_step(i, simData.beta, gen);

                accepted += incr;

            } // end x-space MC trials

            double ratio = static_cast<double>(accepted)/iterCount;

            // check ratio for convergence

            if (ratio >= lowThresh && ratio <= highThresh)    
                break;

            // step is too large

            else if (ratio < lowThresh)                
                xStep[i] *= scale;

            // step is too small

            else                                 
                xStep[i] /= scale;    
    
            count++;

        } // end while loop

        if (count >= maxTrials)
            throw std::runtime_error("Failed to properly converge MC optimization in x-coord.\n");

    } // end x calibration

    // begin calibrating p steps

    for (unsigned i = 0; i < numModes; i++)
    {
        // re-initialize bath coordinates
        
        xCurr[i] = bathCoup[i] * ( dvrLeft/(mass*bathFreq[i]*bathFreq[i]) );
        pCurr[i] = 0.0;
    }

    // run MC trials, modifying steps in each dimension independently

    for (unsigned i = 0; i < numModes; i++)
    {
        int count = 0;

        while (count <= maxTrials)    
        {
            xOld = xCurr[i];
            pOld = pCurr[i];
            accepted = 1;

            for (int j = 0; j < iterCount; j++)
            {
                double stepLen = gsl_rng_uniform(gen) * pStep[i];

                // displace p coord.

                pNew = pOld + pow(-1.0, gsl_rng_uniform_int(gen,2))*stepLen;

                // keep x constant for this run

                xNew = xOld;

                int incr = process_step(i, simData.beta, gen);

                accepted += incr;

            } // end p-space MC trials

            double ratio = static_cast<double>(accepted)/iterCount;

            // check ratio for convergence

            if (ratio >= lowThresh && ratio <= highThresh)   
                break;

            // step is too large

            else if (ratio < lowThresh)               
                pStep[i] *= scale;

            // step is too small

            else                                 
                pStep[i] /= scale;    
    
            count++;

        } // end while loop

        if (count >= maxTrials)
            throw std::runtime_error("Failed to converge MC optimization in p-coord.\n");

    } // end p calibration

    // initialize IC arrays for future runs

    for (int i = 0; i < simData.bathModes; i++)
    {
        double pos = bathCoup[i] * ( dvrLeft/(mass*bathFreq[i]*bathFreq[i]) );

        if (gsl_rng_uniform_int(gen, 2) == 0)
            xVals[i] = pos + gsl_rng_uniform(gen) * xStep[i];
        else
            xVals[i] = pos - gsl_rng_uniform(gen) * xStep[i];

        if (gsl_rng_uniform_int(gen, 2) == 0)
            pVals[i] = gsl_rng_uniform(gen) * pStep[i];
        else
            pVals[i] = -1.0 * gsl_rng_uniform(gen) * pStep[i];
    }

}

/* ------------------------------------------------------------------------- */

// ic_gen() performs a Monte Carlo walk using the optimized step sizes to
// generate sample points in the bath phase space for the QCPI algorithm

void InitialBath::ic_gen(gsl_rng * gen, SimInfo & simData)
{
    for (long i = 1; i < simData.mcSteps; i++)
    { 
        // randomly select index to step, and generate
        // step size in x and p dimension

        int index = gsl_rng_uniform_int(gen, numModes);
        double xLen = gsl_rng_uniform(gen) * xStep[index];
        double pLen = gsl_rng_uniform(gen) * pStep[index];

        xOld = xVals[index];
        pOld = pVals[index];

        // displace x coord.

        if (gsl_rng_uniform_int(gen, 2) == 0)
            xNew = xOld + xLen;
        else
            xNew = xOld - xLen;

        // displace p coord.

        if (gsl_rng_uniform_int(gen, 2) == 0)
            pNew = pOld + pLen;
        else
            pNew = pOld - pLen;

        process_step(index, simData.beta, gen);

        xVals[index] = xPick;
        pVals[index] = pPick;
    }

}

/* ------------------------------------------------------------------------- */
