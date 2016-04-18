/* Implementation file for the Propagator class. These functions
 * handle updates to the reference bath, as well as integration
 * of the TDSE to generate U(t) at each timestep.
 */

#include "propagator.h"

using namespace qcpiConstNS;

/* ------------------------------------------------------------------------- */

Propagator::Propagator()
{
    prop.assign(DSTATES*DSTATES, 0.0);
    ham.assign(DSTATES*DSTATES, 0.0);
    ptemp.assign(DSTATES*DSTATES, 0.0);
}

/* ------------------------------------------------------------------------- */

void Propagator::update(Mode * refModes, double refState, SimInfo & simData)
{
    // run unforced trajectory and integrate U(t)

    for (int i = 0; i < DSTATES; i++)
    {
        for (int j = 0; j < DSTATES; j++)
        {
            if (i == j)
                prop[i*DSTATES+j] = 1.0;
            else
                prop[i*DSTATES+j] = 0.0;
        }
    }

    // first find unforced (x,p)
    // note that ho_update_exact clears ref_modes x(t) and p(t) list

    ho_update_exact(refModes, refState, simData);

    // chunk trajectory into pieces for greater
    // accuracy in integrating U(t)

    for (int chunkNum = 0; chunkNum < simData.chunks; chunkNum++)
    {
        // construct H(x,p) from bath configuration
            
        build_ham(refModes, chunkNum, simData);

        // integrate TDSE for U(t) w/ piece-wise constant
        // Hamiltonian approx.

        rkdriver(DSTATES*DSTATES, 0.0, simData.rhoDelta, simData.rhoSteps);

        // swap out true and temp pointers

        prop.swap(ptemp);
    } 

}

/* ------------------------------------------------------------------------- */

void Propagator::ho_update_exact(Mode * mlist, double ref_state, 
        SimInfo & simData)
{
    double del_t = simData.dt/2.0;
    double chunk_dt = simData.dt/simData.chunks;

    for (int mode = 0; mode < simData.bathModes; mode++)
    {
        double x0, xt;
        double p0, pt;

        double w = mlist[mode].omega;
        double shift = (ref_state * mlist[mode].c)/(mass * w * w);

        // first calculated x(t) at time points
        // for propagator integration

        // clear out any old trajectory info
        // might be more efficient just to overwrite

        mlist[mode].x_t.clear();

        // set up ICs for chunk calc

        x0 = x0_free[mode];
        p0 = p0_free[mode];
        
        for (int i = 0; i < simData.chunks; i++)
        {
            // find x(t) at chunk time points

            xt = (x0 - shift)*cos(w*chunk_dt) + 
                (p0/(mass*w))*sin(w*chunk_dt) + shift;

            pt = p0*cos(w*chunk_dt) - 
                mass*w*(x0 - shift)*sin(w*chunk_dt);

            mlist[mode].x_t.push_back(xt);

            x0 = xt;
            p0 = pt;
        }

        // set up ICs for trajectory

        x0 = x0_free[mode];
        p0 = p0_free[mode];

        // calculate time-evolved x(t), p(t) for
        // first half-step of trajectory

        xt = (x0 - shift)*cos(w*del_t) + (p0/(mass*w))*sin(w*del_t) + shift;

        pt = p0*cos(w*del_t) - mass*w*(x0 - shift)*sin(w*del_t);

        mlist[mode].first_phase = (x0 - shift)*sin(w*del_t) - 
            (p0/(mass*w))*(cos(w*del_t)-1.0) + shift*w*del_t;

        // swap x0, p0 and xt, pt

        x0 = xt;
        p0 = pt;

        // calculate time-evolved x(t), p(t) for
        // second half-step of trajectory

        xt = (x0 - shift)*cos(w*del_t) + (p0/(mass*w))*sin(w*del_t) + shift;

        pt = p0*cos(w*del_t) - mass*w*(x0 - shift)*sin(w*del_t);

        mlist[mode].second_phase = (x0 - shift)*sin(w*del_t) - 
            (p0/(mass*w))*(cos(w*del_t)-1.0) + shift*w*del_t;

        // update current phase space point

        x0_free[mode] = xt;
        p0_free[mode] = pt;

    } // end mode loop
}

/* ------------------------------------------------------------------------- */

// construct system-bath Hamiltonian for
// current timestep

void Propagator::build_ham(Mode * modes, int chunk, SimInfo & simData)
{
    // copy off-diagonal from anharmonic code
    const double off_diag = hbar*tls_freq;

    // store sums for diagonal matrix elements of H
    // left_sum for (0,0); right_sum for (1,1)
    double left_sum;
    double right_sum;

    // energy stores harmonic bath potential energy
    // this should integrate out, but is included here
    // for completeness
    double energy = 0.0;

    // store system and system-bath coupling contributions separately
    complex<double> tls_mat[4];
    complex<double> bath_mat[4];

    // system matrix is just splitting and any asymmetry
    // note that signs should be standard b/c I'm using
    // (-1,+1) instead of (+1,-1)

    tls_mat[0] = simData.asym; //dvr_left*(1.0*asym);
    tls_mat[1] = -1.0*off_diag;
    tls_mat[2] = -1.0*off_diag;
    tls_mat[3] = -1.0*simData.asym; //dvr_right*(1.0*asym);

    // system-bath matrix includes linear coupling plus
    // quadratic offset

    // in this form, it also includes the bath potential energy

    left_sum = 0.0;
    right_sum = 0.0;

    for (int i = 0; i < simData.bathModes; i++)
    {
        double csquare = modes[i].c*modes[i].c;
        double wsquare = modes[i].omega*modes[i].omega;
        double x = modes[i].x_t[chunk];

        left_sum += -1.0*modes[i].c*x*dvr_left +
            csquare*dvr_left*dvr_left/(2.0*mass*wsquare);

        right_sum += -1.0*modes[i].c*x*dvr_right +
            csquare*dvr_right*dvr_right/(2.0*mass*wsquare);

        energy += 0.5*mass*wsquare*x*x;
    }

    // Removing energy term to see if this is
    // dominating energy gap and causing issues

    bath_mat[0] = left_sum;
    bath_mat[1] = 0.0;
    bath_mat[2] = 0.0;
    bath_mat[3] = right_sum;

    // total hamiltonian is sum of system and system-bath parts

    ham[0] = tls_mat[0] + bath_mat[0];
    ham[1] = tls_mat[1] + bath_mat[1];
    ham[2] = tls_mat[2] + bath_mat[2];
    ham[3] = tls_mat[3] + bath_mat[3];

} // end build_ham()

/* ------------------------------------------------------------------------ */

// propagator evolution

void Propagator::prop_eqns(double t, complex<double> * y, complex<double> * dydt)
{
    // this is component version of i*hbar*(dU/dt) = H*U
    // could as write as traditional matrix multiplication

    dydt[0] = -(I/hbar)*(ham[0]*y[0] + ham[1]*y[2]);
    dydt[1] = -(I/hbar)*(ham[0]*y[1] + ham[1]*y[3]);
    dydt[2] = -(I/hbar)*(ham[2]*y[0] + ham[3]*y[2]);
    dydt[3] = -(I/hbar)*(ham[2]*y[1] + ham[3]*y[3]);
}

/* ------------------------------------------------------------------------ */

/* rk4() implements a vanilla fourth-order Runge-Kutta ODE solver, 
although this version has been tweaked to use complex numbers, since
robust complex libraries are hard to find. The derivs fn pointer specifies
the function that will define the ODE equations, and the params array is
included for extra flexibility, as per GSL */

void Propagator::rk4(complex<double> * y, complex<double> * dydx, int n, double x, double h, 
    complex<double> * yout)
{
    int i;
    double xh, h_mid, h_6;
    complex<double> *dym, *dyt, *yt;

    dym = new complex<double>[n];
    dyt = new complex<double>[n];
    yt = new complex<double>[n];

    h_mid = 0.5*h;
    h_6 = h/6.0;
    xh = x+h_mid;

    for (i = 0; i < n; i++)
        yt[i] = y[i] + h_mid*dydx[i];    /* first step */
    
    prop_eqns(xh, yt, dyt);        /* second step */
    
    for (i = 0; i < n; i++)
        yt[i] = y[i] + h_mid*dyt[i];    

    prop_eqns(xh, yt, dym);        /* third step */

    for (i = 0; i < n; i++)
    {
        yt[i] = y[i] + h*dym[i];
        dym[i] += dyt[i];
    }
    
    prop_eqns(x+h, yt, dyt);    /* fourth step */

    for (i = 0; i < n; i++)
        yout[i] = y[i] + h_6*(dydx[i] + dyt[i] + 2.0*dym[i]);

    delete [] dym;
    delete [] dyt; 
    delete [] yt;    
}

/* ------------------------------------------------------------------------ */

/* rkdriver() is an abstraction level that lets different integrators
run using more or less the same input, it initializes some things and
then simply calls the underlying function in a loop; we don't really use
it much, but it was part of the NR approach so it got rolled in here */

void Propagator::rkdriver(int nvar, double x1, double x2, int nstep)
{
    int i, k;
    double x, h;
    complex<double> *v, *vout, *dv;

    v = new complex<double>[nvar]; 
    vout = new complex<double>[nvar];
    dv = new complex<double>[nvar];

    for (i = 0; i < nvar; i++)
        v[i] = prop[i];    /* initialize vector */

    x = x1;
    h = (x2-x1)/nstep;

    for (k = 1; k <= nstep; k++)
    {
        prop_eqns(x, v, dv);
        rk4(v, dv, nvar, x, h, vout);
        if ((double)(x+h) == x)
        {
            fprintf(stderr, "x: %8.5f\n", x);
            fprintf(stderr, "h: %13.10f\n", h);
            fflush(stderr);

            throw std::runtime_error("Step size too small in rkdriver\n");
        }
        x += h;
        for (i = 0; i < nvar; i++)
            v[i] = vout[i];
    }

    for (i = 0; i < nvar; i++)
        ptemp[i] = vout[i];

    delete [] v;
    delete [] vout;
    delete [] dv;
}

/* ------------------------------------------------------------------------ */
