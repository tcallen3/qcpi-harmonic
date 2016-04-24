/* Declaration of the SimInfo class, which handles simulation
 * startup and variable initialization, as well as some output
 * to improve reproducibility. 
 */

#ifndef SIM_INFO_H
#define SIM_INFO_H

#include "harmonic.h"

class SimInfo
{
    private:
        
        // flags to track required parameters

        bool qmstepsflg;
        bool qmdtflg;
        bool bathtempflg;
        bool icnumflg;
        bool kmaxflg;
        bool inflg;
        bool outflg;

        // utility functions

        void parse_file(std::string config, Tokenizer & tok);
        void sanity_check();

    public:

        // externally visible simulation state variables
        // intended for use in other functions/classes

        double asym;
        int bathModes;
        double bathTemp;
        double beta; 

        int icTotal;
        int qmSteps;
        long mcSteps;
        int rhoSteps;

        int kmax;
        double dt;
        double rhoDelta;
        int chunks;

        std::string inputName;
        std::string outputName;

        unsigned long seed;

        // primary initialization and reporting functions

        SimInfo();
        void startup(std::string config, Tokenizer & tok); 
        void print_vars(FILE * outfile, int repeat);
};

#endif
