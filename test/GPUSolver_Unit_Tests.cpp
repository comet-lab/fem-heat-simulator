#include <gtest/gtest.h>
// #include "GPUTimeIntegrator.cuh"
#include "FEM_Simulator.h"
#include "BaseGPU.cpp"
#include <iostream>
#include <string>
#include "TestHelpers.hpp"



/*
These test cases overall aren't great because BaseGPU creates an instance of FEM_Simulator to actually test 
the GPUTimeIntegrator functionality. This is because FEM_Simulator already creates Symmetric Positive (Semi) Definite 
matrices to be used in the solver. (Solver assumes things are SPD) On the flipside though it feels we aren't testing 
the GPUTimeIntegrator directly. 
*/

TEST_F(BaseGPU, Test_UploadVector)
{
    Eigen::VectorXf inputVec(10);
    inputVec << 0,1,2,3,4,5,6,7,8,9;
    std::cout << "Trying to uplaod vector " << std::endl;
    gpu->uploadVector(inputVec,gpu->dVec_d_);

    std::cout << "Trying to download vector " << std::endl;
    Eigen::VectorXf returnVec(10);
    gpu->downloadVector(returnVec,gpu->dVec_d_.data);

    std::cout << "Testing Equivalence " << std::endl;
    for (int i = 0; i < 10; i++){
        std::cout << i << std::endl;
        EXPECT_FLOAT_EQ(inputVec(i), returnVec(i));
    }

}

TEST_F(BaseGPU, Test_UploadMatrix)
{
    int nRows = 30; 
    int nCols = 30;
    int nnzPerRow = 5;
    Eigen::SparseMatrix<float, Eigen::RowMajor> inputMat;
    Eigen::SparseMatrix<float, Eigen::RowMajor> outputMat; 

    inputMat = Eigen::SparseMatrix<float, Eigen::RowMajor>(nRows,nCols);
	inputMat.reserve(Eigen::VectorXi::Constant(nnzPerRow,nRows)); 
        
    // std::cout << "Setting Matrices" << std::endl;
    for (int j = 0; j < nRows; j++){
        for (int i = 0; i < nnzPerRow; i++){
            inputMat.insert(j,nRows-i-1) = j+i;
        }
    }
    inputMat.makeCompressed();

    // std::cout << "Uploading Sparse Matrix" << std::endl;
    gpu->uploadSparseMatrix(inputMat,gpu->Kint_d_);
    // std::cout << "Downloading Sparse Matrix" << std::endl;
    gpu->downloadSparseMatrix(outputMat,gpu->Kint_d_);
    // std::cout << "Checking Result" << std::endl;
    for (int k=0; k<inputMat.outerSize(); ++k){
        Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itA(inputMat, k);
        Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itB(outputMat, k);
        
        for (; itA && itB; ++itA, ++itB) {
            EXPECT_EQ(itA.row(), itB.row()) << "Row mismatch at outer " << k;
            EXPECT_EQ(itA.col(), itB.col()) << "Col mismatch at outer " << k;
            EXPECT_FLOAT_EQ(itA.value(), itB.value()) << "Value mismatch at (" 
                                                  << itA.row() << "," 
                                                  << itA.col() << ")";
        }
    }

}

TEST_F(BaseGPU, Test_UploadAll)
{
    int nRows = femSim->Kconv.rows();
    // This line is called in BaseGPU.cpp when we call gpu->setMode();
    // gpu->uploadAllMatrices(femSim->Kint,femSim->Kconv,femSim->M,femSim->FirrMat,
	// 	femSim->FluenceRate,femSim->Fq,femSim->Fconv,femSim->Fk,femSim->FirrElem);
    
    // Checking Kint
    std::cout << "Checking Kint" << std::endl;
    Eigen::SparseMatrix<float, Eigen::RowMajor> KintOutput; 
    gpu->downloadSparseMatrix(KintOutput,gpu->Kint_d_);
    compareTwoMats(femSim->Kint,KintOutput);

    // Checking Kconv
    std::cout << "Checking Kconv" << std::endl;
    Eigen::SparseMatrix<float, Eigen::RowMajor> KconvOutput; 
    gpu->downloadSparseMatrix(KconvOutput,gpu->Kconv_d_);
    compareTwoMats(femSim->Kconv,KconvOutput);

    // Checking M
    std::cout << "Checking M" << std::endl;
    Eigen::SparseMatrix<float, Eigen::RowMajor> MOutput; 
    gpu->downloadSparseMatrix(MOutput,gpu->M_d_);
    compareTwoMats(femSim->M,MOutput);

    // Checking FirrMat
    std::cout << "Checking FirrMat" << std::endl;
    Eigen::SparseMatrix<float, Eigen::RowMajor> FirrMatOutput; 
    gpu->downloadSparseMatrix(FirrMatOutput,gpu->FirrMat_d_);
    compareTwoMats(femSim->FirrMat,FirrMatOutput);

    // Checking FluenceRate
    std::cout << "Checking FluenceRate" << std::endl;
    Eigen::VectorXf FluenceRateOutput(femSim->FirrMat.cols()); 
    gpu->downloadVector(FluenceRateOutput,gpu->FluenceRate_d_.data);
    compareTwoVectors(femSim->fluenceRate_,FluenceRateOutput);

    // Checking Fq
    std::cout << "Checking Fq" << std::endl;
    Eigen::VectorXf FqOutput(nRows); 
    gpu->downloadVector(FqOutput,gpu->Fq_d_.data);
    compareTwoVectors(femSim->Fq,FqOutput);

    // Checking Fconv
    std::cout << "Checking Fconv" << std::endl;
    Eigen::VectorXf FconvOutput(nRows); 
    gpu->downloadVector(FconvOutput,gpu->Fconv_d_.data);
    compareTwoVectors(femSim->Fconv,FconvOutput);

    // Checking Fk
    std::cout << "Checking Fk" << std::endl;
    Eigen::VectorXf FkOutput(nRows); 
    gpu->downloadVector(FkOutput,gpu->Fk_d_.data);
    compareTwoVectors(femSim->Fk,FkOutput);

    // Checking FirrElem
    std::cout << "Checking FirrElem" << std::endl;
    Eigen::VectorXf FelemOutput(nRows); 
    gpu->downloadVector(FelemOutput,gpu->FirrElem_d_.data);
    compareTwoVectors(femSim->FirrElem,FelemOutput);
}


TEST_F(BaseGPU, Test_ScaleVector)
{
    float alpha = 2;
    Eigen::VectorXf inputVec(10);
    inputVec << 0,1,2,3,4,5,6,7,8,9;
    gpu->uploadVector(inputVec,gpu->Fq_d_);

    gpu->scaleVector(gpu->Fq_d_.data, alpha, 10);

    Eigen::VectorXf returnVec(10);
    gpu->downloadVector(returnVec,gpu->Fq_d_.data);

    for (int j = 0; j < 10; j++){
        ASSERT_FLOAT_EQ(inputVec(j)*alpha,returnVec(j));
    }

}

TEST_F(BaseGPU, Test_addVector1)
{
    float* outputVec_d = nullptr;
    float* vec1_d = nullptr;
    float* vec2_d = nullptr;
    int nRows = 100;

    Eigen::VectorXf outputVec(nRows), truthVec(nRows), vec1(nRows), vec2(nRows);
    vec1.setRandom();
    vec2.setRandom();
    truthVec = vec1 + vec2;

    cudaMalloc(&outputVec_d,nRows*sizeof(float));
    cudaMalloc((void**)&vec1_d,nRows*sizeof(float));
    cudaMalloc((void**)&vec2_d,nRows*sizeof(float));

    cudaMemcpy(vec1_d, vec1.data(),nRows*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_d, vec2.data(),nRows*sizeof(float),cudaMemcpyHostToDevice);

    gpu->addVectors(vec1_d, vec2_d, outputVec_d, nRows, 1);
    gpu->downloadVector(outputVec, outputVec_d);

    for (int i = 0; i < nRows; i++){
        ASSERT_FLOAT_EQ(truthVec(i),outputVec(i));
    }

    cudaFree(outputVec_d);
    outputVec_d = nullptr;
    cudaFree(vec1_d);
    vec1_d = nullptr;
    cudaFree(vec2_d);
    vec2_d = nullptr;

}

TEST_F(BaseGPU, Test_addVector2)
{
    float* outputVec_d = nullptr;
    float* vec1_d = nullptr;
    float* vec2_d = nullptr;
    int nRows = 100;

    Eigen::VectorXf outputVec(nRows), truthVec(nRows), vec1(nRows), vec2(nRows);
    vec1.setRandom();
    vec2.setRandom();
    float scale = -2;
    truthVec = vec1 + scale*vec2;

    cudaMalloc(&outputVec_d,nRows*sizeof(float));
    cudaMalloc((void**)&vec1_d,nRows*sizeof(float));
    cudaMalloc((void**)&vec2_d,nRows*sizeof(float));

    cudaMemcpy(vec1_d, vec1.data(),nRows*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(vec2_d, vec2.data(),nRows*sizeof(float),cudaMemcpyHostToDevice);

    gpu->addVectors(vec1_d, vec2_d, outputVec_d, nRows, scale);
    gpu->downloadVector(outputVec, outputVec_d);

    for (int i = 0; i < nRows; i++){
        ASSERT_FLOAT_EQ(truthVec(i),outputVec(i));
    }

    cudaFree(outputVec_d);
    outputVec_d = nullptr;
    cudaFree(vec1_d);
    vec1_d = nullptr;
    cudaFree(vec2_d);
    vec2_d = nullptr;
}

TEST_F(BaseGPU, Test_AddSparse1)
{
    float TC_ = 2;
    float HTC = 3;
    gpu->addSparse(gpu->Kint_d_, TC_, gpu->Kconv_d_, HTC, gpu->globK_d_);
    Eigen::SparseMatrix<float, Eigen::RowMajor> truthMat = femSim->Kint*TC_ + femSim->Kconv*HTC;
    Eigen::SparseMatrix<float, Eigen::RowMajor> outputMat = truthMat;

    //make sure output vec is set to 0 though.
    for (int k=0; k<outputMat.outerSize(); ++k)
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(outputMat,k); it; ++it)
        {
            it.valueRef() = 0;
            it.row();   // row index
            it.col();   // col index (here it is equal to k)
            it.index(); // inner index, here it is equal to it.row()
        }

    outputMat.makeCompressed();
    gpu->downloadSparseMatrix(outputMat, gpu->globK_d_);

    compareTwoMats(truthMat,outputMat);

}

TEST_F(BaseGPU, Test_AddSparse2)
{
    float TC_ = 0;
    float HTC = -1;
    gpu->addSparse(gpu->Kint_d_, TC_, gpu->Kconv_d_, HTC, gpu->globK_d_);
    Eigen::SparseMatrix<float, Eigen::RowMajor> truthMat = femSim->Kint*TC_ + femSim->Kconv*HTC;
    Eigen::SparseMatrix<float, Eigen::RowMajor> outputMat = truthMat; // copy for size

    //make sure output vec is set to 0 though.
    //make sure output vec is set to 0 though.
    for (int k=0; k<outputMat.outerSize(); ++k)
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(outputMat,k); it; ++it)
        {
            it.valueRef() = 0;
            it.row();   // row index
            it.col();   // col index (here it is equal to k)
            it.index(); // inner index, here it is equal to it.row()
        }


    outputMat.makeCompressed();
    gpu->downloadSparseMatrix(outputMat, gpu->globK_d_);

    compareTwoMats(truthMat,outputMat);

}

TEST_F(BaseGPU, Test_multiplySparseVector)
{
    float* outputVec_d = nullptr;
    int nRows = femSim->Kconv.rows();
    cudaMalloc(&outputVec_d,nRows*sizeof(float));
    gpu->multiplySparseVector(gpu->FirrMat_d_, gpu->FluenceRate_d_, outputVec_d);
    Eigen::VectorXf truthVec = femSim->FirrMat*femSim->fluenceRate_;
    Eigen::VectorXf outputVec(nRows);

    gpu->downloadVector(outputVec, outputVec_d);

    float epsilon = 0.00001;
    for (int i = 0; i < nRows; i++){
        ASSERT_NEAR(truthVec(i),outputVec(i), epsilon); 
    }

    cudaFree(outputVec_d);
    outputVec_d = nullptr;
}

TEST_F(BaseGPU, Test_applyParameters)
{
    femSim->setTC(0.006);
    femSim->setVHC(4.2);
    femSim->setMUA(100);
    femSim->setHTC(0.01);
    int nRows = femSim->Kconv.rows();
    int nCols = femSim->Kconv.cols();

    // std::cout << "FEM Apply Parameters: " << std::endl;
    femSim->applyParametersCPU();
    // std::cout << "GPU Apply Parameters: " << std::endl;
    gpu->applyParameters(femSim->TC, femSim->HTC_, femSim->VHC_, femSim->MUA_, femSim->elemNFR_);
    

    Eigen::SparseMatrix<float, Eigen::RowMajor> outputM, outputK;
    Eigen::VectorXf outputF(nRows);
    // std::cout << "Downloading Vars" << std::endl;
    gpu->downloadVector(outputF, gpu->globF_d_);
    gpu->downloadSparseMatrix(outputM, gpu->globM_d_);
    gpu->downloadSparseMatrix(outputK, gpu->globK_d_);


    // CHECK GLOB K
    std::cout << "Checking K" << std::endl;
    compareTwoMats(femSim->globK_,outputK);

    // CHECK GLOB M
    std::cout << "Checking M" << std::endl;
    compareTwoMats(femSim->globM_,outputM);

    // CHECK GLOB F
    std::cout << "Checking F: nrows = " << nRows << std::endl;
    for (int i = 0; i < nRows; i++){
        ASSERT_TRUE(abs(femSim->globF_(i) - outputF(i)) < 0.0001);
    }
}

TEST_F(BaseGPU, Test_calculateRHS)
{
    femSim->setTC(0.006);
    femSim->setVHC(4.2);
    femSim->setMUA(100);
    femSim->setHTC(0.01);
    int nRows = femSim->Kconv.rows();
    int nCols = femSim->Kconv.cols();

    // Perform Ground truth
    femSim->initializeTimeIntegrationCPU(); // performs applyParametersCPU() internally
    // Calculate RHS using stored matrices
    Eigen::VectorXf bTrue = (femSim->globF_ - femSim->globK_*femSim->dVec_);
    // Eigen::VectorXf bPartial = femSim->globK_*femSim->dVec;

    // -- Run on GPU 
    Eigen::VectorXf outputB_d(nRows);
    // Make sure parameter have been applied
    gpu->applyParameters(femSim->TC, femSim->HTC_, femSim->VHC_, femSim->MUA_, femSim->elemNFR_);
    Eigen::SparseMatrix<float, Eigen::RowMajor> outputK;
    Eigen::VectorXf outputF(nRows);
    gpu->downloadVector(outputF, gpu->globF_d_);
    gpu->downloadSparseMatrix(outputK, gpu->globK_d_);
    // CHECK GLOB K
    std::cout << "Checking K...." << std::endl;
    compareTwoMats(femSim->globK_,outputK);
    // CHECK GLOB F
    std::cout << "Checking F...." << std::endl;
    compareTwoVectors(femSim->globF_, outputF);

    std::cout << "Uploading dVec to gpu->..." << std::endl;
    // Make sure dVec has been uploaded
    gpu->uploadVector(femSim->dVec_,gpu->dVec_d_); // make sure 
    Eigen::VectorXf dVecCopy(nRows);
    gpu->downloadVector(dVecCopy,gpu->dVec_d_.data);
    compareTwoVectors(femSim->dVec_,dVecCopy);

    // Calculate
    std::cout << "Calculating RHS" << std::endl;
    
    float* RHS_d = nullptr;
    gpu->calculateRHS(RHS_d, nRows);
    gpu->downloadVector(outputB_d, RHS_d); 
    std::cout << "Comparing Vectors" << std::endl;
    compareTwoVectors(bTrue, outputB_d);

    cudaFree(RHS_d);
    RHS_d = nullptr;
}

TEST_F(BaseGPU, Test_initializeDV)
{
    femSim->setTC(0.006);
    femSim->setVHC(4.2);
    femSim->setMUA(100);
    femSim->setHTC(0.01);
    int nRows = femSim->Kconv.rows();
    int nCols = femSim->Kconv.cols();

    // Perform Ground truth
    femSim->initializeTimeIntegrationCPU();
    Eigen::VectorXf trueD = femSim->dVec_;
    Eigen::VectorXf trueV = femSim->vVec_;
    // Apply GPU 
    Eigen::VectorXf outputD(nRows), outputV(nRows);
    gpu->applyParameters(femSim->TC, femSim->HTC_, femSim->VHC_, femSim->MUA_, femSim->elemNFR_);

    gpu->initialize(trueD, outputV); // automatically assigns v to input
    
    std::cout<< "Downloading d" <<std::endl;
    gpu->downloadVector(outputD, gpu->dVec_d_.data);
    // gpu->downloadVector(outputV, gpu->vVec_d_);
    std::cout<< "Comparing d" <<std::endl;
    compareTwoVectors(trueD, outputD);
    std::cout<< "Comparing v" <<std::endl;
    // v is found using two different iterative solvers, so I am allowing a good amount of wiggle
    compareTwoVectors(trueV, outputV, 0.01); 
}

