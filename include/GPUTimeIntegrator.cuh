#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <Eigen/Sparse>
#include <iostream>
#include "AmgXSolver.hpp"
#include "FEM_Simulator.h"


struct DeviceCSR {
        int rows;
        int cols;
        int nnz;
        int *rowPtr_d = nullptr;
        int *colIdx_d = nullptr;
        float *data_d = nullptr;
        cusparseSpMatDescr_t spMatDescr = nullptr;
    };

struct DeviceVec {
        float *data = nullptr;
        cusparseDnVecDescr_t vecHandle = nullptr;
    };    

class GPUTimeIntegrator {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GPUTimeIntegrator();
    GPUTimeIntegrator(float alpha, float deltaT);
    ~GPUTimeIntegrator();

    void applyParameters(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& Kint,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& Kconv,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& M,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& FirrMat,
        float TC, float HTC, float VHC, float MUA,
        const Eigen::VectorXf& FluenceRate,
        const Eigen::VectorXf& Fq,
        const Eigen::VectorXf& Fconv,
        const Eigen::VectorXf& Fk,
        const Eigen::VectorXf& FirrElem,
        bool elemNFR
    ); 

    void applyParameters();
    void applyParameters(float TC, float HTC, float VHC, float MUA, bool elemNFR);
    void initialize(const Eigen::VectorXf & dVec, Eigen::VectorXf& vVec);
    void initializeWithModel();
    void setup(float alpha);
    void singleStep();
    void singleStepWithUpdate();
    
    void calculateRHS(float* &b_d, int nRows);
    // void calculateLHS(float* b_d, int nRows);

    void setModel(FEM_Simulator* model);
    void updateModel();
    void releaseModel();
    void getMatricesFromModel();

    void uploadAllMatrices(
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& Kint,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& Kconv,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& M,
        const Eigen::SparseMatrix<float, Eigen::RowMajor>& FirrMat,
        const Eigen::VectorXf& FluenceRate,
        const Eigen::VectorXf& Fq,
        const Eigen::VectorXf& Fconv,
        const Eigen::VectorXf& Fk,
        const Eigen::VectorXf& FirrElem
    ); 
    
    // Clear Structs 
    void freeCSR(DeviceCSR& dA);
    void freeModelMatrices();
    void freeDeviceVec(DeviceVec &vec);

    // Commands to upload vectors/matrices to gpu
    void uploadSparseMatrix(const Eigen::SparseMatrix<float,Eigen::RowMajor>& inMat, DeviceCSR& outMat);
    void uploadSparseMatrix(int numRows, int numCols, int nnz, const int* csrOffsets, const int* columns, const float* values, DeviceCSR& dA);
    void uploadVector(const Eigen::VectorXf& v, DeviceVec& dV);
    void uploadVector(const float* data, int n, DeviceVec& dV);
    void uploaddVec_d();
    void uploadFluenceRate();

    // commands to download vectors/matrices from gpu
    void downloadVector(Eigen::VectorXf& v,const float* dv);
    void downloaddVec_d();
    void downloadvVec_d();
    void downloadSparseMatrix(Eigen::SparseMatrix<float,Eigen::RowMajor>& outMat, const DeviceCSR& source);

     // Sparse-sparse addition
    void addSparse(const DeviceCSR& A, float alpha, const DeviceCSR& B, float beta, DeviceCSR& C);

    // Sparse-dense multiplication (csr*vec)
    void multiplySparseVector(const DeviceCSR& A, DeviceVec& vec, float* out);
    bool solveSparseLinearSystem();

    // Setters
    void setAlpha(float alpha);
    void setDeltaT(float deltaT);

    // Kernels
    void scaleCSR(DeviceCSR& dA, float alpha);
    void addVectors(float* v1, float* v2, float* out, int n, float vScale);
    void scaleVector(float *data, float alpha, int n);
    void printVector(float *data, int n);
    

    //-----------------------------------------------------------------------------------------------
    // Variable Declarations 

    // Store the model
    FEM_Simulator* femModel_ = nullptr;

    //Time stepping variables
    float alpha_ = 1; 
    float deltaT_ = 0.05;

    // Store GPU handles
    cusparseHandle_t handle_;
    // Stored Sparse Matrices
    DeviceCSR Kint_d_, Kconv_d_, M_d_, FirrMat_d_;
    // Store Dense Vectors
    DeviceVec FluenceRate_d_, Fq_d_, Fconv_d_, Fk_d_, FirrElem_d_;  
    
    //---GPU resident Global terms --
    DeviceCSR globK_d_;
    DeviceCSR globM_d_;
    float* globF_d_ = nullptr;

    // State and State Velocity Vectors
    DeviceVec dVec_d_;
    float* vVec_d_ = nullptr;

    AmgXSolver& amgxSolver_;

};

// --------------------
// Error Checking Macros
// --------------------
#define CHECK_CUDA(call)                                               \
{                                                                   \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

#define CHECK_CUSPARSE(call)                                           \
{                                                                   \
    cusparseStatus_t status = (call);                                  \
    if (status != CUSPARSE_STATUS_SUCCESS) {                           \
        fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",               \
                __FILE__, __LINE__, status);                           \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

#define CUDA_KERNEL_CHECK()                                            \
{                                                                   \
    cudaError_t err = cudaGetLastError();                              \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "Kernel launch error at %s:%d: %s\n",          \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
}

#define CHECK_CUSOLVER(func)                                               \
{                                                                    \
    cusolverStatus_t status = (func);                                   \
    if (status != CUSOLVER_STATUS_SUCCESS) {                             \
        fprintf(stderr, "CUSOLVER error at %s:%d, code %d\n",           \
                __FILE__, __LINE__, status);                            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}

