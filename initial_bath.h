/* Declaration of the InitialBath class, which handles the setup and
 * generation of harmonic bath initial conditions for each Monte
 * Carlo sample taken during simulation.
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

        unsigned numModes;

        // steps sizes in each bath dimension

        std::vector<double> xStep;
        std::vector<double> pStep;

        // helper functions

        void read_spec(std::string specName, std::vector<double> & omega, 
            std::vector<double> & jvals, Tokenizer & tok);
        void calibrate_mc(gsl_rng * gen, SimInfo & simData);
        int process_step(unsigned index, double beta, gsl_rng * gen);
        double dist(unsigned index, double beta);

    public:

        // bath characteristic data for each mode

        std::vector<double> bathFreq;
        std::vector<double> bathCoup;

        // positions and momenta to initialize MC trials

        std::vector<double> xVals;
        std::vector<double> pVals;

        // main interface to initialize and sample bath state

        explicit InitialBath(unsigned modes);
        void bath_setup(SimInfo & simData, Tokenizer & tok, 
            gsl_rng * gen, int myRank);
        void ic_gen(gsl_rng * gen, SimInfo & simData);
};

#endif
