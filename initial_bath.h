/* Declaration of the InitialBath class, which aids in the generation
 * of samples from the harmonic bath phase space
 */

#ifndef INITIAL_BATH_H
#define INITIAL_BATH_H

#include "harmonic.h"
#include "sim_info.h"

class InitialBath
{
    private:
        double xOld, xNew;
        double pOld, pNew;
        double xPick, pPick;

        std::vector<double> xStep;
        std::vector<double> pStep;

        void read_spec(std::string specName, std::vector<double> & omega, 
            std::vector<double> & jvals, Tokenizer & tok);
        int process_step(unsigned index, double beta, gsl_rng * gen);
        double dist(unsigned index, double beta);

    public:
        std::vector<double> bathFreq;
        std::vector<double> bathCoup;
        std::vector<double> xVals;
        std::vector<double> pVals;

        void bath_setup(std::string specName, int numModes, Tokenizer & tok, 
                int myRank);
        void calibrate_mc(gsl_rng * gen, SimInfo & simData);
        void ic_gen(gsl_rng * gen, SimInfo & simData);
};

#endif
