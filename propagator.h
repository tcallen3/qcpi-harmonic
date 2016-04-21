/* Declaration file for the Propagator class, designed to handle
 * the reference-based harmonic bath updates which constitute the
 * bulk of the quantum information in QCPI
 */

#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "harmonic.h"
#include "sim_info.h"

class Propagator
{
    private:
        int matLen;
        cvector ham;
        cvector ptemp;
        double refState;

        void ho_update_exact(std::vector<Mode> & mlist, SimInfo & simData);
        void build_ham(std::vector<Mode> & modes, int chunk, SimInfo & simData);
        void prop_eqns(cvector & y, cvector & dydt);
        void rk4(cvector & y, cvector & dydx, int n, 
            double x, double h, cvector & yout);
        void rkdriver(int nvar, double x1, double x2, int nstep);

    public:
        cvector prop;
        std::vector<Ref> oldRefs;
        std::vector<complex<double> > qiAmp;
        vector<double> xRef;
        vector<double> pRef;

        explicit Propagator(int qmSteps);
        void update(std::vector<Mode> & mlist, SimInfo & simData);
        void pick_ref(int seg, gsl_rng * gen);
        complex<double> get_kernel_prod(const Path & path);
};

#endif
