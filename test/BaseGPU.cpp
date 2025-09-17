#include <gtest/gtest.h>
#include "FEM_Simulator.h"

class BaseGPU : public testing::Test {
protected:
    BaseGPU() {
        Eigen::VectorXf inputVec(10);
        inputVec << 0,1,2,3,4,5,6,7,8,9;
        gpu.uploadVector(inputVec,gpu.dVec_d);

        int nodesPerAxis[3] = {20,20,20};
        std::vector<std::vector<std::vector<float>>> Temp(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
        for(int i = 0; i < nodesPerAxis[0]; i++){
            for(int j = 0; j < nodesPerAxis[1]; j++){
                for(int k = 0; k < nodesPerAxis[2]; k++){
                    Temp[i][j][k] = 20;
                }
            }
        }
        float tissueSize[3] = { 1,1,1 };
        int Nn1d = 2;
        float mua = 1.0f;
        float tc = 1.0f;
        float vhc = 1.0f;
        float htc = 1.0f;
        
        femSim = new FEM_Simulator(Temp, tissueSize, tc, vhc, mua, htc, 2);
        femSim->alpha = 1/2.0;
        femSim->silentMode = true;
        float laserPose[6] = {0, 0, -25, 0, 0 ,0};
        float laserPower = 1;
        float beamWaist = 0.0168;
        femSim->deltaT = 0.05f;
        int BC[6] = { 2,2,2,2,2,0 };
        femSim->setBoundaryConditions(BC);
        femSim->setFlux(0.0f);
        femSim->setAmbientTemp(24);

        femSim->setFluenceRate(laserPose, laserPower, beamWaist);
        femSim->buildMatrices();
        // femSim->applyParametersCPU();
        // femSim->initializeTimeIntegrationCPU();

        gpu.uploadAllMatrices(femSim->Kint,femSim->Kconv,femSim->M,femSim->FirrMat,
		femSim->FluenceRate,femSim->Fq,femSim->Fconv,femSim->Fk,femSim->FirrElem);
    }

    // ~QueueTest() override = default;
    GPUSolver gpu = GPUSolver();
    FEM_Simulator* femSim;
};