// AmgXSolver.hpp
#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "amgx_c.h"

class AmgXSolver {
public:
    AmgXSolver(const std::string& configFile);
    ~AmgXSolver();

    void uploadMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);
    void uploadMatrix(int rows, int cols, int nnz,const float* vals,const int* rowPtr,const int* colIdx);
    void updateMatrixValues(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);
    void setup();
    void solve(const Eigen::VectorXf& b, Eigen::VectorXf& x);
    void solve(const float* b, float * x);

private:
    AMGX_resources_handle rsrc;
    AMGX_config_handle cfg;
    AMGX_solver_handle solver;
    AMGX_matrix_handle Amat;
    AMGX_vector_handle Ax, Ab;
    int rows, cols, nnz;
};
