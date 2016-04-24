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

void Propagator::update(std::vector<Mode> & refModes, SimInfo & simData)
{
    // run unforced trajectory and integrate U(t)

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

    // first find unforced (x,p)
    // note that ho_update_exact clears ref_modes x(t) and p(t) list

    ho_update_exact(refModes, simData);

    // chunk trajectory into pieces for greater
    // accuracy in integrating U(t)

    for (int chunkNum = 0; chunkNum < simData.chunks; chunkNum++)
    {
        // construct H(x,p) from bath configuration
            
        build_ham(refModes, chunkNum, simData);

        // integrate TDSE for U(t) w/ piece-wise constant
        // Hamiltonian approx.

        ode_solve(0.0, simData.rhoDelta, simData.rhoSteps);

        // swap out true and temp pointers

        prop.swap(ptemp);
    } 

}

/* ------------------------------------------------------------------------- */

void Propagator::ho_update_exact(std::vector<Mode> & mlist, SimInfo & simData)
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

        // first calculated x(t) at time points
        // for propagator integration

        // clear out any old trajectory info
        // might be more efficient just to overwrite

        mlist[mode].xt.clear();

        // set up ICs for chunk calc

        x0 = xRef[mode];
        p0 = pRef[mode];
        
        for (int i = 0; i < simData.chunks; i++)
        {
            // find x(t) at chunk time points

            xt = (x0 - shift)*cos(w*chunkDelta) + 
                (p0/(mass*w))*sin(w*chunkDelta) + shift;

            pt = p0*cos(w*chunkDelta) - 
                mass*w*(x0 - shift)*sin(w*chunkDelta);

            mlist[mode].xt.push_back(xt);

            x0 = xt;
            p0 = pt;
        }

        // set up ICs for trajectory

        x0 = xRef[mode];
        p0 = pRef[mode];

        // calculate time-evolved x(t), p(t) for
        // first half-step of trajectory

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) + shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].phase1 = (x0 - shift)*sin(w*delta) - 
            (p0/(mass*w))*(cos(w*delta)-1.0) + shift*w*delta;

        // swap x0, p0 and xt, pt

        x0 = xt;
        p0 = pt;

        // calculate time-evolved x(t), p(t) for
        // second half-step of trajectory

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) + shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].phase2 = (x0 - shift)*sin(w*delta) - 
            (p0/(mass*w))*(cos(w*delta)-1.0) + shift*w*delta;

        // update current phase space point

        xRef[mode] = xt;
        pRef[mode] = pt;

    } // end mode loop
}

/* ------------------------------------------------------------------------- */

// construct system-bath Hamiltonian for
// current timestep

void Propagator::build_ham(std::vector<Mode> & modes, int chunk, SimInfo & simData)
{
    // copy off-diagonal from anharmonic code
    const double offDiag = hbar*tlsFreq;

    // store system and system-bath coupling contributions separately
    std::vector<complex<double> > systemMat;
    std::vector<complex<double> > bathMat;

    systemMat.assign(matLen*matLen, 0.0);
    bathMat.assign(matLen*matLen, 0.0);

    std::vector<double> dvrVals;

    dvrVals.push_back(dvrLeft);
    dvrVals.push_back(dvrRight);

    // system matrix is just splitting and any asymmetry
    // note that signs should be standard b/c I'm using
    // (-1,+1) instead of (+1,-1)

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

    // system-bath matrix includes bi-linear coupling 

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

} // end build_ham()

/* ------------------------------------------------------------------------ */

// propagator evolution

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

/* rk4() implements a vanilla fourth-order Runge-Kutta ODE solver, 
although this version has been tweaked to use complex numbers, since
robust complex libraries are hard to find. The derivs fn pointer specifies
the function that will define the ODE equations, and the params array is
included for extra flexibility, as per GSL */

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

/* rkdriver() is an abstraction level that lets different integrators
run using more or less the same input, it initializes some things and
then simply calls the underlying function in a loop; we don't really use
it much, but it was part of the NR approach so it got rolled in here */

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
