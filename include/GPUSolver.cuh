#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <Eigen/Sparse>
#include <iostream>
#include "AmgXSolver.hpp"


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
        cusparseDnVecDescr_t vecHandle;
    };    

struct GPUParameters {
    DeviceCSR Kint_d, Kconv_d, M_d, FirrMat_d;
    float* FluenceRate_d = nullptr;
    float* Fq_d = nullptr;
    float* Fconv_d = nullptr;
    float* Fk_d = nullptr;
    float* FirrElem_d = nullptr;
    
};


class GPUSolver {

public:
    GPUSolver();
    ~GPUSolver();

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

    void applyParameters(float TC, float HTC, float VHC, float MUA, bool elemNFR);
    void initializeDV(const Eigen::VectorXf & dVec, Eigen::VectorXf& vVec);
    void setup(float alpha, float deltaT);
    void singleStep(float alpha, float deltaT);

    void calculateRHS(float* &b_d, int nRows);
    // void calculateLHS(float* b_d, int nRows);

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
    void freeDeviceVec(DeviceVec &vec);

    void uploadSparseMatrix(const Eigen::SparseMatrix<float,Eigen::RowMajor>& inMat, DeviceCSR& outMat);
    void uploadSparseMatrix(int numRows, int numCols, int nnz, const int* csrOffsets, const int* columns, const float* values, DeviceCSR& dA);
    void uploadVector(const Eigen::VectorXf& v, DeviceVec& dV);
    void uploadVector(const float* data, int n, DeviceVec& dV);
    
    void downloadVector(Eigen::VectorXf& v,const float* dv);
    void downloadSparseMatrix(Eigen::SparseMatrix<float,Eigen::RowMajor>& outMat, const DeviceCSR& source);

     // Sparse-sparse addition
    void addSparse(const DeviceCSR& A, float alpha, const DeviceCSR& B, float beta, DeviceCSR& C);

    // Sparse-dense multiplication (csr*vec)
    void multiplySparseVector(const DeviceCSR& A, DeviceVec& vec, float* out);

    // Kernels
    void scaleCSR(DeviceCSR& dA, float alpha);
    void addVectors(float* v1, float* v2, float* out, int n, float vScale);
    void scaleVector(float *data, float alpha, int n);
    void printVector(float *data, int n);

    //-----------------------------------------------------------------------------------------------
    // Variable Declarations 

    // Store GPU handles
    cusparseHandle_t handle;
    // Stored Sparse Matrices
    DeviceCSR Kint_d, Kconv_d, M_d, FirrMat_d;
    // Store Dense Vectors
    DeviceVec FluenceRate_d, Fq_d, Fconv_d, Fk_d, FirrElem_d;  
    
    //---GPU resident Global terms --
    DeviceCSR globK_d;
    DeviceCSR globM_d;
    float* globF_d = nullptr;

    // State and State Velocity Vectors
    DeviceVec dVec_d;
    float* vVec_d;

    AmgXSolver* amgxSolver = nullptr;

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

