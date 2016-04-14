/* Implementation file for the InitialBath class, which stores 
 * information used to generate samples of the bath phase space
 * for QCPI averaging.
 */

#include "initial_bath.h"

void InitialBath::read_spec(std::string specName, std::vector<double> & omega, 
    std::vector<double> & jvals, Tokenizer & tok)
{
    ifstream spec_file;
    std::string buffer;
    std::string entry;
    const char comment_char = '#';

    spec_file.open(specName.c_str(), std::ios_base::in);

    if (!spec_file.is_open())
        throw std::runtime_error("Could not open spectral density file\n");

    // read in file line-by-line

    Tokenizer::iterator iter;

    while (getline(spec_file, buffer))
    {   
        tok.assign(buffer);

        iter = tok.begin();

        // skip empty lines and comments

        if (tok.begin() == tok.end())
            continue;

        entry = *iter;

        if (entry[0] == comment_char)
            continue;

        omega.push_back(boost::lexical_cast<double>(entry));

        // assign arguments

        ++iter;

        if (iter == tok.end())
            throw std::runtime_error("Incomplete line in spectral density file\n");

        entry = *iter;

        jvals.push_back(boost::lexical_cast<double>(entry)); 

    } // end specden reading

    spec_file.close();
}

/* ------------------------------------------------------------------------- */

int InitialBath::process_step(unsigned index, double beta, gsl_rng * gen)
{
    double prob = dist(index, beta);
    int accepted = 0;

    if (prob >= 1.0) // accept
    {
        // update state

        xOld = xNew;     
        pOld = pNew;
        
        xPick = xNew;
        pPick = pNew;

        accepted++;
    }
    else
    {    
        double xi = gsl_rng_uniform(gen);
    
        if (prob >= xi) // accept (rescue)
        {    
            // update state

            xOld = xNew;     
            pOld = pNew;

            xPick = xNew;
            pPick = pNew;

            accepted++;  
        }
        else
        {
            xPick = xOld;
            pPick = pOld;
        }
    }

    return accepted;
}

/* ------------------------------------------------------------------------- */

double InitialBath::dist(unsigned index, double beta)
{
    // need atomic units    

    double omega = bathFreq[index];
    double coup = bathCoup[index];

    double f = hbar*omega*beta;

    // shift to DVR state w/ -1 element in sigma_z basis

    double lambda = dvr_left * coup * ( 1.0/(mass*omega*omega) );

    // shifting distribution for equilibrium

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

void InitialBath::bath_setup(std::string specName, int numModes, 
        Tokenizer & tok, int myRank)
{
    // read in spectral density from file

    std::vector<double> omega;
    std::vector<double> jvals;

    read_spec(specName, omega, jvals, tok);

    // first calculate reorg energy via simple integration

    double dw = omega[1] - omega[0];    // assumes uniform spacing
    double sum = 0.0;

    sum += (jvals.front() + jvals.back())/2.0;

    for (unsigned i = 1; i < jvals.size()-1; i++)
        sum += jvals[i];

    sum *= dw;

    double w0 = sum/numModes;

    // report reorg energy to command line

    double pi = acos(-1.0);
    double reorg = 4.0 * sum / pi;

    if (myRank == 0)
        fprintf(stdout, "Reorg. energy (au): %.7f\n", reorg);

    // discretize frequencies

    // NOTE: notation here is weird to match Tuseeta's
    //  results; I can't seem to replicate them using
    //  normal C syntax, should probably look into this

    for (int j = 0; j < numModes; j++)
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

    // check modes for 0 vals (may create workaround)

    for (int i = 0; i < bathFreq.size(); i++)
    {
        if (bathFreq[i] == 0)
        {
            for (int j = i; j < bathFreq.size(); j++)
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

    for (int i = 0; i < numModes; i++)
        bathCoup[i] = sqrt(2.0*w0/pi) * bathFreq[i];
}

/* ------------------------------------------------------------------------- */

void InitialBath::calibrate_mc(gsl_rng * gen, SimInfo & simData)
{
    const double maxTrials = 1000;
    const double baseStep = 10.0;
    const double scale = 0.8;
    const double lowThresh = 0.45;
    const double highThresh = 0.55;
    const int iterCount = 1000;
    std::vector<double> xCurr;
    std::vector<double> pCurr;

    // initialize step sizes and (x,p) at minimum of x

    for (int i = 0; i < bathFreq.size(); i++)
    {
        xStep.push_back(gsl_rng_uniform(gen) * baseStep);
        pStep.push_back(gsl_rng_uniform(gen) * baseStep);

        double pos = 
            bathCoup[i]*( dvr_left/(mass*bathFreq[i]*bathFreq[i]) );

        xCurr.push_back(pos);
        pCurr.push_back(0.0);
    }

    double xOld, xNew;
    double pOld, pNew;
    int accepted;

    // run MC step tweaking for each mode and phase space dim. separately

    for (int i = 0; i < bathFreq.size(); i++)
    {
        int count = 0;

        while (count <= maxTrials)    // keep looping until convergence
        {
            xOld = xCurr[i];
            pOld = pCurr[i];
            accepted = 1;

            for (int j = 0; j < iterCount; j++)
            {
                double stepLen = gsl_rng_uniform(gen) * xStep[i];

                // displace x coord.

                xNew = xOld + pow(-1.0, gsl_rng_uniform_int(gen,2))*stepLen;
/*
                if (gsl_rng_uniform_int(gen, 2) == 0)
                    xNew = xOld + stepLen;
                else
                    xNew = xOld - stepLen;
*/
                // keep p constant for this run

                pNew = pOld;

                int incr = process_step(i, simData.beta, gen);

                accepted += incr;

                // pass in pOld for both configs
                // this makes sure p isn't affected
/*
                double prob = dist(xOld, xNew, pOld, pOld, bathFreq[i], 
                    bathCoup[i], simData.beta);

                if (prob >= 1.0) // accept
                {
                    // update state
                    xOld = xNew;     
                    pOld = pNew;
                    accepted++;
                }
                else
                {    
                    double xi = gsl_rng_uniform(gen);
    
                    if (prob >= xi) // accept (rescue)
                    {    
                        // update state
                        xOld = xNew;     
                        pOld = pNew;
                        accepted++;  
                    }
                    else // reject
                    {
                        // technically a null operation
                        xOld = xOld;
                        pOld = pOld;
                    }
                }
*/
            } // end x-space MC trials

            double ratio = static_cast<double>(accepted)/iterCount;

            // check ratio for convergence

            if (ratio >= lowThresh && ratio <= highThresh)    // step is good
                break;
            else if (ratio < lowThresh)                // step too large, decrease
                xStep[i] *= scale;
            else                                 // step too small, increase
                xStep[i] /= scale;    
    
            count++;

        } // end while loop

        if (count >= maxTrials)
            throw std::runtime_error("Failed to properly converge MC optimization in x-coord.\n");

    } // end x calibration

    // begin calibrating p steps

    for (int i = 0; i < bathFreq.size(); i++)
    {
        // re-initialize bath coordinates
        
        xCurr[i] = bathCoup[i] * ( dvr_left/(mass*bathFreq[i]*bathFreq[i]) );
        pCurr[i] = 0.0;
    }

    // run MC step tweaking for each mode 

    for (int i = 0; i < bathFreq.size(); i++)
    {
        int count = 0;

        while (count <= maxTrials)    // keep looping until convergence
        {
            xOld = xCurr[i];
            pOld = pCurr[i];
            accepted = 1;

            for (int j = 0; j < iterCount; j++)
            {
                double stepLen = gsl_rng_uniform(gen) * pStep[i];

                // displace p coord.

                pNew = pOld + pow(-1.0, gsl_rng_uniform_int(gen,2))*stepLen;
/*
                if (gsl_rng_uniform_int(gen, 2) == 0)
                    pNew = pOld + stepLen;
                else
                    pNew = pOld - stepLen;
*/
                // keep x constant for this run

                xNew = xOld;

                int incr = process_step(i, simData.beta, gen);

                accepted += incr;

               // use xOld for both configurations
                // this makes sure x isn't affected
/*
                double prob = dist(xOld, xOld, pOld, pNew, bathFreq[i], 
                    bathCoup[i], simData.beta);

                if (prob >= 1.0) // accept
                {
                    // update state
                    xOld = xNew;     
                    pOld = pNew;
                    accepted++;
                }
                else
                {    
                    double xi = gsl_rng_uniform(gen);
    
                    if (prob >= xi) // accept (rescue)
                    {    
                        // update state
                        xOld = xNew;     
                        pOld = pNew;
                        accepted++;  
                    }
                    else // reject
                    {
                        // technically a null operation
                        xOld = xOld;
                        pOld = pOld;
                    }
                }
*/
            } // end p-space MC trials

            double ratio = static_cast<double>(accepted)/iterCount;

            // check ratio for convergence

            if (ratio >= lowThresh && ratio <= highThresh)    // step is good
                break;
            else if (ratio < lowThresh)                // step too large, decrease
                pStep[i] *= scale;
            else                                 // step too small, increase
                pStep[i] /= scale;    
    
            count++;

        } // end while loop

        if (count >= maxTrials)
            throw std::runtime_error("Failed to converge MC optimization in p-coord.\n");

    } // end p calibration

    // initialize IC arrays

    for (int i = 0; i < simData.bath_modes; i++)
    {
        double pos = bathCoup[i] * ( dvr_left/(mass*bathFreq[i]*bathFreq[i]) );

        if (gsl_rng_uniform_int(gen, 2) == 0)
            xVals[i] =  pos + gsl_rng_uniform(gen) * xStep[i];
        else
            xVals[i] = pos - gsl_rng_uniform(gen) * xStep[i];

        if (gsl_rng_uniform_int(gen, 2) == 0)
            pVals[i] = gsl_rng_uniform(gen) * pStep[i];
        else
            pVals[i] = -gsl_rng_uniform(gen) * pStep[i];
    }

}

/* ------------------------------------------------------------------------- */

void InitialBath::ic_gen(gsl_rng * gen, SimInfo & simData)
{
    for (long i = 1; i < simData.mc_steps; i++)
    { 
        // randomly select index to step, and generate
        // step size in x and p dimension

        int index = gsl_rng_uniform_int( gen, bathFreq.size() );
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
