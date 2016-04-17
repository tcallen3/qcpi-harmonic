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
    public:
        cvector prop;
        cvector ham;
        cvector ptemp;
        vector<double> x0_free;
        vector<double> p0_free;

        Propagator();
        void ho_update_exact(Mode * mlist, double ref_state, 
            SimInfo & simData);
        void build_ham(Mode * modes, int chunk, SimInfo & simData);
        void prop_eqns(double t, complex<double> * y, complex<double> * dydt);
        void rk4(complex<double> * y, complex<double> * dydx, int n, 
            double x, double h, complex<double> * yout);
        void rkdriver(int nvar, double x1, double x2, int nstep);
};

#endif
