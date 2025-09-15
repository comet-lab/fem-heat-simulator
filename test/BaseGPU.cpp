#include <gtest/gtest.h>
#include "FEM_Simulator.h"

class BaseGPU : public testing::Test {
protected:
    BaseGPU() {
        Eigen::VectorXf inputVec(10);
        inputVec << 0,1,2,3,4,5,6,7,8,9;
        gpu.uploadVector(inputVec,gpu.dVec_d);

        std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
                                                       { {0,0,0}, {0,0,0}, {0,0,0} },
                                                       { {0,0,0}, {0,0,0}, {0,0,0} } };
        float tissueSize[3] = { 1,1,1 };
        int Nn1d = 2;
        float mua = 1.0f;
        float tc = 1.0f;
        float vhc = 1.0f;
        float htc = 1.0f;
        
        femSim = new FEM_Simulator(Temp, tissueSize, tc, vhc, mua, htc, 2);
        femSim->silentMode = true;
        float laserPose[6] = {0, 0, -25, 0, 0 ,0};
        float laserPower = 1;
        float beamWaist = 0.0168;
        femSim->deltaT = 0.05f;
        int BC[6] = { 2,0,0,0,0,0 };
        femSim->setBoundaryConditions(BC);
        femSim->setFlux(0.0f);
        femSim->setAmbientTemp(24);

        femSim->setFluenceRate(laserPose, laserPower, beamWaist);
        femSim->buildMatrices();
        femSim->initializeTimeIntegration();

        gpu.uploadAllMatrices(femSim->Kint,femSim->Kconv,femSim->M,femSim->FirrMat,
		femSim->FluenceRate,femSim->Fq,femSim->Fconv,femSim->Fk,femSim->FirrElem);
    }

    // ~QueueTest() override = default;
    GPUSolver gpu = GPUSolver();
    FEM_Simulator* femSim;
};