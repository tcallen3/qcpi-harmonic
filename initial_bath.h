/* Declaration of the InitialBath class, which aids in the generation
 * of samples from the harmonic bath phase space
 */

#ifndef INITIAL_BATH_H
#define INITIAL_BATH_H

#include "qcpi_harmonic.h"

class InitialBath
{
    private:
        vector<double> xStep;
        vector<double> pStep;

        void read_spec(std::string specName, std::vector<double> & omega, 
            std::vector<double> & jvals, Tokenizer & tok);
        void calibrate_mc(gsl_rng * gen, SimInfo & simData);
        double dist(double xOld, double xNew, double pOld, double pNew, 
            double omega, double coup, double beta);

    public:
        vector<double> bathFreq;
        vector<double> bathCoup;
        vector<double> xvals;
        vector<double> pvals;

        void bath_setup(std::string specName, int numModes, Tokenizer & tok);
        void ic_gen(gsl_rng * gen, SimInfo & simData);
};

#endif
