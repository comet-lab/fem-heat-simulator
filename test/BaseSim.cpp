#include <gtest/gtest.h>
#include "../src/FEM_Simulator.h"

class BaseSim : public testing::Test {
protected:
    BaseSim() {
        // q0_ remains empty
        std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
                                                       { {0,0,0}, {0,0,0}, {0,0,0} },
                                                       { {0,0,0}, {0,0,0}, {0,0,0} } };
        float tissueSize[3] = { 1,1,1 };
        int Nn1d = 2;
        float mua = 1.0f;
        float tc = 1.0f;
        float vhc = 1.0f;
        float htc = 1.0f;
        
        femSimLin = new FEM_Simulator(Temp, tissueSize, tc, vhc, mua, htc, 2);
        femSimQuad = new FEM_Simulator(Temp, tissueSize, tc, vhc, mua, htc, 3);
    }

    // ~QueueTest() override = default;

    FEM_Simulator* femSimLin;
    FEM_Simulator* femSimQuad;
};