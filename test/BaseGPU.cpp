#include <gtest/gtest.h>
#include "GPUTimeIntegrator.cuh"

class BaseGPU : public testing::Test {
protected:
    GPUTimeIntegrator *gpu = nullptr;
    FEM_Simulator* femSim = nullptr;     

    void SetUp() override {
        
        //Example FEM_Simulator initialization
        int nodesPerAxis[3] = { 20, 20, 20 };
        std::vector<std::vector<std::vector<float>>> TempAsVec(
            nodesPerAxis[0], std::vector<std::vector<float>>(
                nodesPerAxis[1], std::vector<float>(nodesPerAxis[2], 20.0f)));

        float tissueSize[3] = { 1, 1, 1 };
        femSim = new FEM_Simulator(TempAsVec, tissueSize, 1.0f, 1.0f, 1.0f, 1.0f, 2);
        femSim->alpha_ = 0.5f;
        femSim->silentMode = true;

        float laserPose[6] = { 0, 0, -25, 0, 0, 0 };
        femSim->dt_ = 0.05f;
        femSim->setFluenceRate(laserPose, 1.0f, 0.0168f);

        femSim->buildMatrices();
        
        gpu = new GPUTimeIntegrator(femSim->alpha_, femSim->dt_);

        gpu->setModel(femSim);
    }


    void TearDown() override {
        // std::cout << "Entered Tear Down" << std::endl;
        delete gpu;
        gpu = nullptr;
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