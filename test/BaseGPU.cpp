#include <gtest/gtest.h>
#include "FEM_Simulator.h"

class BaseGPU : public testing::Test {
protected:
    GPUTimeIntegrator *gpu = nullptr;
    FEM_Simulator* femSim = nullptr;     

    void SetUp() override {
        
        //Example FEM_Simulator initialization
        int nodesPerAxis[3] = { 20, 20, 20 };
        std::vector<std::vector<std::vector<float>>> Temp(
            nodesPerAxis[0], std::vector<std::vector<float>>(
                nodesPerAxis[1], std::vector<float>(nodesPerAxis[2], 20.0f)));

        float tissueSize[3] = { 1, 1, 1 };
        femSim = new FEM_Simulator(Temp, tissueSize, 1.0f, 1.0f, 1.0f, 1.0f, 2);
        femSim->alpha = 0.5f;
        femSim->silentMode = true;

        float laserPose[6] = { 0, 0, -25, 0, 0, 0 };
        femSim->deltaT = 0.05f;
        femSim->setFluenceRate(laserPose, 1.0f, 0.0168f);

        femSim->buildMatrices();
        // such a hack because I don't want to instantiate two gpu objects since technically 2 shouldn't exist. 
        gpu = new GPUTimeIntegrator();

        gpu->uploadAllMatrices(femSim->Kint, femSim->Kconv, femSim->M, femSim->FirrMat,
            femSim->FluenceRate, femSim->Fq, femSim->Fconv, femSim->Fk, femSim->FirrElem);
    }


    void TearDown() override {
        // std::cout << "Entered Tear Down" << std::endl;
        delete gpu;
        gpu = nullptr;
        // femSim->gpuHandle = nullptr; // such a hack because I don't want to instantiate two gpu objects
        // std::cout << "Cleared gpu" << std::endl;       
        delete femSim;
        femSim = nullptr;
        // std::cout << "Cleared femSim" << std::endl; 
    }

    static void SetUpTestSuite() {
        
        // AMGX_initialize_plugins();
    }

    static void TearDownTestSuite() {
        // AMGX_finalize();
    }
};