#include <gtest/gtest.h>
// #include "GPUSolver.cuh"
#include "FEM_Simulator.h"
#include "BaseGPU.cpp"
#include <iostream>
#include <string>

TEST_F(BaseGPU, Test_UploadVector)
{
    Eigen::VectorXf inputVec(10);
    inputVec << 0,1,2,3,4,5,6,7,8,9;

    gpu.uploadVector(inputVec,gpu.dVec_d);

    std::cout << inputVec.size() << std::endl;


    Eigen::VectorXf returnVec(10);
    gpu.downloadVector(returnVec,gpu.dVec_d.data);

    for (int i = 0; i < 10; i++){
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
    outputMat = Eigen::SparseMatrix<float, Eigen::RowMajor>(nRows,nCols);
	outputMat.reserve(Eigen::VectorXi::Constant(nnzPerRow,nRows)); 
        
    std::cout << "Setting Matrices" << std::endl;
    for (int j = 0; j < nRows; j++){
        for (int i = 0; i < nnzPerRow; i++){
            inputMat.insert(j,nRows-i-1) = j+i;
            outputMat.insert(j,nRows-i-1) = -1; // junk data for comparison
        }
    }
    inputMat.makeCompressed();
    outputMat.makeCompressed();


    std::cout << "Uploading Sparse Matrix" << std::endl;
    gpu.uploadSparseMatrix(inputMat,gpu.Kint_d);
    std::cout << "Downloading Sparse Matrix" << std::endl;
    gpu.downloadSparseMatrix(outputMat,gpu.Kint_d);
    std::cout << "Checking Result" << std::endl;
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

TEST_F(BaseGPU, Test_ScaleVector)
{
    float alpha = 2;
    Eigen::VectorXf inputVec(10);
    inputVec << 0,1,2,3,4,5,6,7,8,9;
    gpu.uploadVector(inputVec,gpu.Fq_d);

    gpu.scaleVector(gpu.Fq_d.data, alpha, 10);

    Eigen::VectorXf returnVec(10);
    gpu.downloadVector(returnVec,gpu.Fq_d.data);

    for (int j = 0; j < 10; j++){
        ASSERT_FLOAT_EQ(inputVec(j)*alpha,returnVec(j));
    }

}

TEST_F(BaseGPU, Test_AddSparse1)
{
    float TC = 2;
    float HTC = 3;
    gpu.addSparse(gpu.Kint_d, TC, gpu.Kconv_d, HTC, gpu.globK_d);
    Eigen::SparseMatrix<float, Eigen::RowMajor> truthMat = femSim->Kint*TC + femSim->Kconv*HTC;
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
    gpu.downloadSparseMatrix(outputMat, gpu.globK_d);


    for (int k=0; k<truthMat.outerSize(); ++k){
        Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itA(truthMat, k);
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

TEST_F(BaseGPU, Test_AddSparse2)
{
    float TC = 0;
    float HTC = -1;
    gpu.addSparse(gpu.Kint_d, TC, gpu.Kconv_d, HTC, gpu.globK_d);
    Eigen::SparseMatrix<float, Eigen::RowMajor> truthMat = femSim->Kint*TC + femSim->Kconv*HTC;
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
    gpu.downloadSparseMatrix(outputMat, gpu.globK_d);


    for (int k=0; k<truthMat.outerSize(); ++k){
        Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itA(truthMat, k);
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

TEST_F(BaseGPU, Test_multiplySparseVector)
{
    float* outputVec_d;
    int nRows = femSim->Kconv.rows();
    cudaMalloc(&outputVec_d,nRows*sizeof(float));
    gpu.multiplySparseVector(gpu.Kint_d, gpu.Fq_d, outputVec_d);
    Eigen::VectorXf truthVec = femSim->Kint*femSim->Fq;
    Eigen::VectorXf outputVec(nRows);

    gpu.downloadVector(outputVec, outputVec_d);

    for (int i = 0; i < nRows; i++){
        ASSERT_FLOAT_EQ(truthVec(i),outputVec(i));
    }

}
