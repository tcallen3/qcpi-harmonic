/* Implementation file for the SimInfo class, which
 * handles parameter initialization from simulation
 * input file.
 */

#include "sim_info.h"

using namespace qcpiConstNS;

/* ------------------------------------------------------------------------- */

SimInfo::SimInfo()
{
    // initialize some variables

    qmstepsflg = false;
    qmdtflg = false;
    bathtempflg = false;
    icnumflg = false;
    kmaxflg = false;
    inflg = false;
    outflg = false;

    asym = 0.5;             // system asymmetry (in kcal/mol)
    bathModes = 60;         // number of bath oscillators
    mcSteps = 50000;        // default Monte Carlo burn
    chunks = 5;             // must evenly divide step_pts
    rhoSteps = 100;         // points used to integrate U(t)
    seed = 179524;          // RNG seed for consistency
}

/* ------------------------------------------------------------------------- */

void SimInfo::startup(std::string config, Tokenizer & tok)
{
    // set category flags

    bool timeflg;
    bool simflg;
    bool fileflg;

    bool reqflg;

    // read in settings from config file

    parse_file(config, tok);

    // check required flags

    timeflg = qmstepsflg && qmdtflg;

    simflg = icnumflg && kmaxflg && bathtempflg;

    fileflg = inflg && outflg;

    reqflg = timeflg && simflg && fileflg;

    if (!reqflg)
    {
        std::string errMsg;

        if (!timeflg)
        {
            errMsg = 
                "Must specify number of quantum steps and timestep values\n"; 
            throw std::runtime_error(errMsg);
        }

        if (!simflg)
        {
            errMsg =
                "Must specify number of ICs, kmax, and bath temperature\n";
            throw std::runtime_error(errMsg);
        }

        if (!fileflg)
        {
            errMsg =
                "Must specify spectral density input, and data output file\n";
            throw std::runtime_error(errMsg);
        }
    }

    // set additional parameters

    rhoDelta = dt/chunks;
    asym *= kcal_to_hartree;
    beta = 1.0/(kb*bathTemp);

    // make sure MC run is long enough

    const long mc_buff = 10000;

    if (mcSteps < mc_buff * bathModes)
        mcSteps = mc_buff * bathModes;

    // ensure inputs are meaningful

    sanity_check();
} 

/* ------------------------------------------------------------------------- */

void SimInfo::parse_file(std::string config, Tokenizer & tok)
{
    const char comment_char = '#';

    ifstream conf_file;
    std::string buffer;
    std::string arg1;
    std::string arg2;

    conf_file.open(config.c_str(), ios_base::in);

    if (!conf_file.is_open())
        throw std::runtime_error("Could not open configuration file\n");

    // read in file line-by-line

    Tokenizer::iterator iter;

    while (getline(conf_file, buffer))
    {   
        tok.assign(buffer);

        iter = tok.begin();

        // skip empty lines and comments

        if (tok.begin() == tok.end())
            continue;

        arg1 = *iter;

        if (arg1[0] == comment_char)
            continue;

        // assign arguments

        ++iter;

        if (iter == tok.end())
            throw std::runtime_error("Malformed command: " + arg1 + "\n");

        arg2 = *iter;

        // execute logic on tokens (normalize case?)

        if (arg1 == "rho_steps")
            rhoSteps = boost::lexical_cast<int>(arg2);

        else if (arg1 == "qm_steps")
        {
            qmSteps = boost::lexical_cast<int>(arg2);
            qmstepsflg = true;
        }

        else if (arg1 == "timestep")
        {
            dt = boost::lexical_cast<double>(arg2);
            qmdtflg = true;
        }

        else if (arg1 == "ic_num")
        {
            icTotal = boost::lexical_cast<int>(arg2);
            icnumflg = true;
        }

        else if (arg1 == "kmax")
        {
            kmax = boost::lexical_cast<int>(arg2);
            kmaxflg = true;
        }

        else if (arg1 == "temperature")
        {
            bathTemp = boost::lexical_cast<double>(arg2);
            bathtempflg = true;
        }

        else if (arg1 == "input")
        {
            inputName = arg2;
            inflg = true;
        }

        else if (arg1 == "output")
        {
            outputName = arg2;
            outflg = true;
        }

        else if (arg1 == "asymmetry")
        {
            // set system asymmetry

            asym = boost::lexical_cast<double>(arg2);
        }

        else if (arg1 == "bath_modes")
        {
            // set # of bath oscillators

            bathModes = boost::lexical_cast<int>(arg2);
        }

        else if (arg1 == "mc_steps")
        {
            // set MC burn size

            mcSteps = boost::lexical_cast<long>(arg2);
        }

        else if (arg1 == "chunks")
        {
            // set # of action grouping chunks

            chunks = boost::lexical_cast<int>(arg2);
        }

        else if (arg1 == "rng_seed")
        {
            // set RNG seed (ensure seed isn't 0)

            seed = boost::lexical_cast<unsigned long>(arg2);

            if (seed == 0)
                seed = 179524;
        }

        else    // skip unrecognized commands and weird lines
            continue;
    }

    conf_file.close();
}

/* ------------------------------------------------------------------------- */

void SimInfo::sanity_check()
{
    if (bathModes <= 0)
        throw std::runtime_error("Bath oscillator number must be positive\n");

    if (icTotal <= 0)
        throw std::runtime_error("Number of ICs must be positive\n");

    if (dt <= 0)
        throw std::runtime_error("Timestep must be positive\n");

    if (qmSteps <= 0)
        throw std::runtime_error("Total simulation steps must be positive\n");

    if (kmax <= 0)
        throw std::runtime_error("Kmax segments must be positive\n");

    if (chunks <= 0)
        throw std::runtime_error("Action chunk number must be positive\n");

    if (rhoSteps <= 0)
        throw std::runtime_error("Number of ODE steps for rho(t) must be positive\n");

    if (bathTemp <= 0)
        throw std::runtime_error("Bath temperature must be positive\n");

    // check non-negative quantities

    if (mcSteps < 0)
        throw std::runtime_error("Monte Carlo burn length can't be negative\n");

    if (kmax > qmSteps)
        throw std::runtime_error("Memory length cannot exceed total simulation time\n");
 
    if (kmax <= 0)
        throw std::runtime_error("kmax must be positive\n");

    if (dt <= 0)
        throw std::runtime_error("Quantum timestep must be positive\n");

    if (qmSteps < kmax)
        throw std::runtime_error("Quantum step number smaller than kmax\n");

}

/* ------------------------------------------------------------------------- */

void SimInfo::print(FILE * outfile, int repeat)
{
    fprintf(outfile, "Quantum steps: %d\n", qmSteps);
    fprintf(outfile, "Memory length (kmax): %d\n", kmax);
    fprintf(outfile, "Step length (a.u): %.5f\n", dt);
    fprintf(outfile, "IC num: %d\n", icTotal);
    fprintf(outfile, "RNG seed: %lu\n", seed);
    fprintf(outfile, "MC skip: %ld\n", mcSteps);

    fprintf(outfile, "Analytic trajectory integration: on\n");

    fprintf(outfile, "Simulation used EACP reference hopping\n");

    fprintf(outfile, "Input spectral density: %s\n", inputName.c_str());

    fprintf(outfile, "Total simulated time (a.u.): %.4f\n\n", qmSteps*dt);

    for (int i = 0; i < repeat; i++)
        fprintf(outfile, "-");

    fprintf(outfile, "\nBath Summary\n");

    for (int i = 0; i < repeat; i++)
        fprintf(outfile, "-");

    fprintf(outfile, "\n\n");

    fprintf(outfile, "Bath modes: %d\n", bathModes);
    fprintf(outfile, "Bath temperature: %.2f\n", bathTemp);
    fprintf(outfile, "Inverse temperature: %.4f\n", beta);
    fprintf(outfile, "Bath mode mass parameter: %.3f\n", mass);
    fprintf(outfile, "Using shifted W(x,p) (minimum at x=lambda)\n\n");

    for (int i = 0; i < repeat; i++)
        fprintf(outfile, "-");

    fprintf(outfile, "\nSystem Summary\n");

    for (int i = 0; i < repeat; i++)
        fprintf(outfile, "-");

    fprintf(outfile, "\n\n");

    fprintf(outfile, "Off-diagonal TLS element: %f\n", tls_freq);
    fprintf(outfile, "Asymmetry: %.7e hartree\n", asym);
    fprintf(outfile, "Left DVR state: %.3f\n", dvr_left);
    fprintf(outfile, "Right DVR state: %.3f\n\n", dvr_right);
}

/* ------------------------------------------------------------------------- */


