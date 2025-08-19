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
    void setup();
    void solve(const Eigen::VectorXf& b, Eigen::VectorXf& x);

private:
    AMGX_resources_handle rsrc;
    AMGX_config_handle cfg;
    AMGX_solver_handle solver;
    AMGX_matrix_handle Amat;
    AMGX_vector_handle Ax, Ab;
    int n_rows, n_cols, nnz;
};
