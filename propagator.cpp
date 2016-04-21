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
        refState = dvr_left;
        oldRefs[seg] = REF_LEFT;
    }
    else    
    {
        refState = dvr_right;
        oldRefs[seg] = REF_RIGHT;
    }

}

/* ------------------------------------------------------------------------- */

complex<double> Propagator::get_kernel_prod(const Path & path)
{
    unsigned size = path.fwd_path.size();
    
    unsigned splus0 = path.fwd_path[size-2];
    unsigned splus1 = path.fwd_path[size-1];

    unsigned findex = splus1*DSTATES + splus0;

    unsigned sminus0 = path.bwd_path[size-2];
    unsigned sminus1 = path.bwd_path[size-1];
                
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

        rkdriver(matLen*matLen, 0.0, simData.rhoDelta, simData.rhoSteps);

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

        mlist[mode].x_t.clear();

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

            mlist[mode].x_t.push_back(xt);

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

        mlist[mode].first_phase = (x0 - shift)*sin(w*delta) - 
            (p0/(mass*w))*(cos(w*delta)-1.0) + shift*w*delta;

        // swap x0, p0 and xt, pt

        x0 = xt;
        p0 = pt;

        // calculate time-evolved x(t), p(t) for
        // second half-step of trajectory

        xt = (x0 - shift)*cos(w*delta) + (p0/(mass*w))*sin(w*delta) + shift;

        pt = p0*cos(w*delta) - mass*w*(x0 - shift)*sin(w*delta);

        mlist[mode].second_phase = (x0 - shift)*sin(w*delta) - 
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
    const double offDiag = hbar*tls_freq;

    // store system and system-bath coupling contributions separately
    std::vector<complex<double> > tls_mat;
    std::vector<complex<double> > bath_mat;

    tls_mat.assign(matLen*matLen, 0.0);
    bath_mat.assign(matLen*matLen, 0.0);

    std::vector<double> dvrVals;

    dvrVals.push_back(dvr_left);
    dvrVals.push_back(dvr_right);

    // system matrix is just splitting and any asymmetry
    // note that signs should be standard b/c I'm using
    // (-1,+1) instead of (+1,-1)

    for (int i = 0; i < matLen; i++)
    {
        for (int j = 0; j < matLen; j++)
        {
            if (i == j)
                tls_mat[i*matLen+j] = dvrVals[i]*simData.asym;
            else
                tls_mat[i*matLen+j] = -1.0*offDiag;
        }
    }

    // system-bath matrix includes bi-linear coupling 

    std::vector<double> energies;

    energies.assign(matLen, 0.0);

    for (int i = 0; i < simData.bathModes; i++)
    {
        double csquare = modes[i].c*modes[i].c;
        double wsquare = modes[i].omega*modes[i].omega;
        double x = modes[i].x_t[chunk];

        for (int index = 0; index < matLen; index++)
        {
            energies[index] += -1.0*modes[i].c*x*dvrVals[index] +
                csquare*dvrVals[index]*dvrVals[index]/(2.0*mass*wsquare);
        }
    }

    for (int i = 0; i < matLen; i++)
        bath_mat[i*matLen+i] = energies[i];

    // total hamiltonian is sum of system and system-bath parts

    for (int i = 0; i < matLen; i++)
    {
        for (int j = 0; j < matLen; j++)
        {
            int index = i*matLen+j;

            ham[index] = tls_mat[index] + bath_mat[index];
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

void Propagator::rk4(cvector & y, cvector & dydx, int n, double h, 
    cvector & yout)
{
    double h_mid, h_6;
    cvector yt, dyt, dym;

    yt.assign(matLen*matLen, 0.0);
    dyt.assign(matLen*matLen, 0.0);
    dym.assign(matLen*matLen, 0.0);

    h_mid = 0.5*h;
    h_6 = h/6.0;

    for (unsigned i = 0; i < yt.size(); i++)
        yt[i] = y[i] + h_mid*dydx[i];    /* first step */
    
    prop_eqns(yt, dyt);        /* second step */
    
    for (unsigned i = 0; i < yt.size(); i++)
        yt[i] = y[i] + h_mid*dyt[i];    

    prop_eqns(yt, dym);        /* third step */

    for (unsigned i = 0; i < yt.size(); i++)
    {
        yt[i] = y[i] + h*dym[i];
        dym[i] += dyt[i];
    }
    
    prop_eqns(yt, dyt);    /* fourth step */

    for (unsigned i = 0; i < dyt.size(); i++)
        yout[i] = y[i] + h_6*(dydx[i] + dyt[i] + 2.0*dym[i]);

}

/* ------------------------------------------------------------------------ */

/* rkdriver() is an abstraction level that lets different integrators
run using more or less the same input, it initializes some things and
then simply calls the underlying function in a loop; we don't really use
it much, but it was part of the NR approach so it got rolled in here */

void Propagator::rkdriver(int nvar, double x1, double x2, int nstep)
{
    double h;
    cvector v, vout, dv;

    v.assign(prop.begin(), prop.end());
    vout.assign(matLen*matLen, 0.0);
    dv.assign(matLen*matLen, 0.0);

    h = (x2-x1)/nstep;

    for (int k = 1; k <= nstep; k++)
    {
        prop_eqns(v, dv);
        rk4(v, dv, nvar, h, vout);

        v.assign(vout.begin(), vout.end());
    }

    ptemp.assign(vout.begin(), vout.end());
}

/* ------------------------------------------------------------------------ */
