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

void InitialBath::calibrate_mc(gsl_rng * gen, SimInfo & simData)
{

}

/* ------------------------------------------------------------------------- */

double InitialBath::dist(double xOld, double xNew, double pOld, double pNew, 
            double omega, double coup, double beta)
{

}

/* ------------------------------------------------------------------------- */

void InitialBath::bath_setup(std::string specName, int numModes, 
        Tokenizer & tok)
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

    for (int i = 0; i < numModes; i++)
    {
        if (bathFreq[i] == 0)
        {
            for (int j = i; j < numModes; j++)
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

void InitialBath::ic_gen(gsl_rng * gen, SimInfo & simData)
{

}

/* ------------------------------------------------------------------------- */

