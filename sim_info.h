/* Declaration of the SimInfo class, which handles aspects of
 * startup and variable initialization for the simulation.
 */

#ifndef SIM_INFO_H
#define SIM_INFO_H

#include "harmonic.h"

class SimInfo
{
    private:
        bool qmstepsflg;
        bool qmdtflg;
        bool bathtempflg;
        bool icnumflg;
        bool kmaxflg;
        bool inflg;
        bool outflg;

        void parse_file(std::string config, Tokenizer & tok);
        void sanity_check();

    public:
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

        SimInfo();
        void startup(std::string config, Tokenizer & tok); 
};

#endif
