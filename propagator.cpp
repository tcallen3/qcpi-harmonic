/* Implementation file for the Propagator class. These functions
 * handle updates to the reference bath, as well as integration
 * of the TDSE to generate U(t) at each timestep.
 */

#include "propagator.h"

using namespace qcpiConstNS;

/* ------------------------------------------------------------------------- */

Propagator::Propagator(int qmSteps)
{
    refState = REF_LEFT;
    oldRefs.assign(qmSteps, REF_LEFT);

    matLen = DSTATES;

    prop.assign(matLen*matLen, 0.0);
    ham.assign(matLen*matLen, 0.0);
    ptemp.assign(matLen*matLen, 0.0);
}

/* ------------------------------------------------------------------------- */

// pick_ref() selects the reference state for the next QCPI iteration based
// on the local density matrix amplitude, in accord with the DCSH algorithm

void Propagator::pick_ref(int seg, gsl_rng * gen)
{
    double xi = gsl_rng_uniform(gen);
    double rhoVal;

    if ((seg-1) < 0)
        rhoVal = 1.0;
    else
        rhoVal = qiAmp[seg-1].real();

    if (xi < rhoVal)
    {
        refState = dvrLeft;
        oldRefs[seg] = REF_LEFT;
    }
    else    
    {
        refState = dvrRight;
        oldRefs[seg] = REF_RIGHT;
    }

}

/* ------------------------------------------------------------------------- */

// get_kernel_prod() finds the current propagator product required in the
// iteration of rho(t) from rho(t-dt), based on the forward and backward paths

complex<double> Propagator::get_kernel_prod(const Path & path)
{
    unsigned size = path.fwdPath.size();
    
    unsigned splus0 = path.fwdPath[size-2];
    unsigned splus1 = path.fwdPath[size-1];

    unsigned findex = splus1*DSTATES + splus0;

    unsigned sminus0 = path.bwdPath[size-2];
    unsigned sminus1 = path.bwdPath[size-1];
                
    unsigned bindex = sminus1*DSTATES + sminus0;

    return prop[findex]*conj(prop[bindex]);
}

/* ------------------------------------------------------------------------- */

// update() evolves the harmonic bath and recalculates U(t) for each new
// iteration, based on the reference state chosen for this step

void Propagator::update(std::vector<Mode> & refModes, SimInfo & simData)
{
    // reset U(t) to identity as inital matrix

    for (int i = 0; i < matLen; i++)
    {
        for (int j = 0; j < matLen; j++)
        {
            if (i == j)
                prop[i*matLen+j] = 1.0;
            else
                prop[i*matLen+j] = 0.0;
        }
    }

    // update bath oscillators using reference state

    bath_update(refModes, simData);

    // sample energies along quantum time step and integrate U(t)
    // using piece-wise constant hamiltonian

    for (int chunkNum = 0; chunkNum < simData.chunks; chunkNum++)
    {
        // construct H(x,p) from bath configuration
            
        build_hamiltonian(refModes, chunkNum, simData);

        // integrate TDSE for U(t) w/ piece-wise constant H(x,p)

        ode_solve(0.0, simData.rhoDelta, simData.rhoSteps);

        prop.swap(ptemp);
    } 

}

/* ------------------------------------------------------------------------- */

// bath_update() evolves all the bath oscillators according to the selected
// system reference state, using closed form solutions for forced harmonic
// oscillators

void Propagator::bath_update(std::vector<Mode> & mlist, SimInfo & simData)
{
    double delta = simData.dt/2.0;
    double chunkDelta = simData.dt/simData.chunks;

    for (int mode = 0; mode < simData.bathModes; mode++)
    {
        double x0, xt;
        double p0, pt;

        double w = mlist[mode].omega;
        double c = mlist[mode].c;

        double shift = (refState * c)/(mass * w * w);

        mlist[mode].xt.clear();

        // set up ICs for each subinterval

        x0 = xRef[mode];
        p0 = pRef[mode];
        
        for (int i = 0; i < simData.chunks; i++)
        {
            // find coordinates at interval time points

            xt = (x0 - shift)*cos(w*chunkDelta) + 
                (p0/(mass*w))*sin(w*chunkDelta) + shift;

            pt = p0*cos(w*chunkDelta) - 
                mass*w*(x0 - shift)*sin(w*chunkDelta);

            mlist[mode].xt.push_back(xt);

            x0 = xt;
            p0 = pt;
        }

        // set up ICs for trajectory step from 0 to dt

        x0 = xRef[mode];
        p0 = pRef[mode];

        // calculate time-evolved x(t), p(t) for first half-step
        // (using half-steps to match phases with full QCPI update functions)

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) + shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].phase1 = (x0 - shift)*sin(w*delta) - 
            (p0/(mass*w))*(cos(w*delta)-1.0) + shift*w*delta;

        // swap x0, p0 and xt, pt

        x0 = xt;
        p0 = pt;

        // calculate time-evolved x(t), p(t) for second half-step

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) + shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].phase2 = (x0 - shift)*sin(w*delta) - 
            (p0/(mass*w))*(cos(w*delta)-1.0) + shift*w*delta;

        // update oscillator coordinates

        xRef[mode] = xt;
        pRef[mode] = pt;

    } 
}

/* ------------------------------------------------------------------------- */

// build_hamiltonian() constructs the piece-wise constant Hamiltonians used
// to propagate U(t) during each QCPI time step, based on the system-bath 
// energetics along the reference trajectory

void Propagator::build_hamiltonian(std::vector<Mode> & modes, int chunk, 
        SimInfo & simData)
{
    const double offDiag = hbar*tlsFreq;

    std::vector<complex<double> > systemMat;
    std::vector<complex<double> > bathMat;

    systemMat.assign(matLen*matLen, 0.0);
    bathMat.assign(matLen*matLen, 0.0);

    std::vector<double> dvrVals;

    dvrVals.push_back(dvrLeft);
    dvrVals.push_back(dvrRight);

    // system matrix corresponds to simple asymmetric TLS

    for (int i = 0; i < matLen; i++)
    {
        for (int j = 0; j < matLen; j++)
        {
            if (i == j)
                systemMat[i*matLen+j] = dvrVals[i]*simData.asym;
            else
                systemMat[i*matLen+j] = -1.0*offDiag;
        }
    }

    // system-bath interaction includes bi-linear coupling 

    std::vector<double> energies;

    energies.assign(matLen, 0.0);

    for (int i = 0; i < simData.bathModes; i++)
    {
        double csquare = modes[i].c*modes[i].c;
        double wsquare = modes[i].omega*modes[i].omega;
        double x = modes[i].xt[chunk];

        for (int index = 0; index < matLen; index++)
        {
            energies[index] += -1.0*modes[i].c*x*dvrVals[index] +
                csquare*dvrVals[index]*dvrVals[index]/(2.0*mass*wsquare);
        }
    }

    // these interactions are all diagonal

    for (int i = 0; i < matLen; i++)
        bathMat[i*matLen+i] = energies[i];

    // total hamiltonian is sum of system and system-bath parts

    for (int i = 0; i < matLen; i++)
    {
        for (int j = 0; j < matLen; j++)
        {
            int index = i*matLen+j;

            ham[index] = systemMat[index] + bathMat[index];
        }
    }

} 

/* ------------------------------------------------------------------------ */

// prop_eqns() encodes the time-dependent Schroedinger equation for the
// ODE integration functions

void Propagator::prop_eqns(cvector & y, cvector & dydt)
{
    dydt.assign(matLen*matLen, 0.0);

    for (int i = 0; i < matLen; i++)
    {
        for (int j = 0; j < matLen; j++)
        {
            for (int k = 0; k < matLen; k++)
            {
                dydt[i*matLen+j] += (-I/hbar)*ham[i*matLen+k]*y[k*matLen+j];
            }
        }
    }

}

/* ------------------------------------------------------------------------ */

// ode_step() handles 4th order Runge-Kutta integration of U(t) at each
// step of the sub-divided dt value

void Propagator::ode_step(cvector & yin, double dt, cvector & yout)
{
    double dt2, dt6;
    cvector yt, k1, k2, k3, k4;

    yt.assign(yin.size(), 0.0);

    k1.assign(yin.size(), 0.0);
    k4 = k3 = k2 = k1;

    dt2 = 0.5*dt;
    dt6 = dt/6.0;

    // generate k_i derivatives for RK4 algorithm

    // find k1 from initial time point

    prop_eqns(yin, k1);

    // find k2 from midpoint projection

    for (unsigned i = 0; i < yin.size(); i++)
        yt[i] = yin[i] + dt2*k1[i];
    
    prop_eqns(yt, k2);
   
    // find k3 from midpoint projection of k2

    for (unsigned i = 0; i < yin.size(); i++)
        yt[i] = yin[i] + dt2*k2[i];    

    prop_eqns(yt, k3);

    // find k4 from endpoint projection of k3

    for (unsigned i = 0; i < yin.size(); i++)
        yt[i] = yin[i] + dt*k3[i];
    
    prop_eqns(yt, k4);

    // sum k_i contributions to get result at t+dt

    for (unsigned i = 0; i < yin.size(); i++)
        yout[i] = yin[i] + dt6*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);

}

/* ------------------------------------------------------------------------ */

// ode_solve() handles setup and iteration of the 4th order Runge-Kutta
// solver used to evaluate U(t) during each time step

void Propagator::ode_solve(double tstart, double tend, int nsteps)
{
    double dt;
    cvector vecIn, vecOut;

    vecIn.assign(prop.begin(), prop.end());
    vecOut.assign(vecIn.size(), 0.0);

    dt = (tend - tstart)/nsteps;

    for (int step = 0; step < nsteps; step++)
    {
        ode_step(vecIn, dt, vecOut);

        vecIn.assign(vecOut.begin(), vecOut.end());
    }

    ptemp.assign(vecOut.begin(), vecOut.end());
}

/* ------------------------------------------------------------------------ */
