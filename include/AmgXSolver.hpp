// AmgXSolver.hpp
#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "amgx_c.h"

class AmgXSolver {
public:
    // Delete copy constructor and assignment operator
    AmgXSolver(const AmgXSolver&) = delete;
    AmgXSolver& operator=(const AmgXSolver&) = delete;

    // Public accessor for singleton
    static AmgXSolver& getInstance() {
        static AmgXSolver instance; // guaranteed to be created once
        return instance;
    }

    // Public interface
    void uploadMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);
    void uploadMatrix(int rows, int cols, int nnz, const float* data, const int* rowPtr, const int* colIdx);
    void updateMatrixValues(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);
    void setup();
    void solve(const Eigen::VectorXf& b, Eigen::VectorXf& x);
    void solve(const float* b, float* x);

private:
    // Private constructor and destructor
    AmgXSolver();
    ~AmgXSolver();

    // AMGX handles
    AMGX_config_handle cfg = nullptr;
    AMGX_resources_handle rsrc = nullptr;
    AMGX_solver_handle solver = nullptr;
    AMGX_matrix_handle Amat = nullptr;
    AMGX_vector_handle Ax = nullptr;
    AMGX_vector_handle Ab = nullptr;

    int rows = 0;
    int cols = 0;
    int nnz = 0;
};