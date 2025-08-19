// AmgXSolver.cpp
#include "AmgXSolver.hpp"
#include <iostream>

AmgXSolver::AmgXSolver(const std::string& configFile) {
    AMGX_initialize();
    AMGX_initialize_plugins();

    AMGX_resources_create_simple(&rsrc, nullptr, nullptr, nullptr);
    AMGX_config_create_from_file(&cfg, configFile.c_str());
    AMGX_solver_create(&solver, rsrc, AMGX_mode_fDDI, cfg);
    // fDDI = float precision, double indices, real
}

AmgXSolver::~AmgXSolver() {
    AMGX_solver_destroy(solver);
    AMGX_matrix_destroy(Amat);
    AMGX_vector_destroy(Ax);
    AMGX_vector_destroy(Ab);
    AMGX_resources_destroy(rsrc);
    AMGX_config_destroy(cfg);
    AMGX_finalize_plugins();
    AMGX_finalize();
}

void AmgXSolver::uploadMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A) {

    /* Guarding against memory leaking*/
    if (Amat) {
        AMGX_matrix_destroy(Amat);
        Amat = nullptr;
    }
    if (Ax) { AMGX_vector_destroy(Ax); Ax = nullptr; }
    if (Ab) { AMGX_vector_destroy(Ab); Ab = nullptr; }

    /* Actual upload code*/
    Eigen::SparseMatrix<float, Eigen::RowMajor> Ac = A;
    Ac.makeCompressed();
    n = Ac.rows();

    std::vector<int> row_ptr(Ac.rows() + 1);
    std::vector<int> col_idx(Ac.nonZeros());
    std::vector<float> values(Ac.nonZeros());

    std::copy(Ac.outerIndexPtr(), Ac.outerIndexPtr() + Ac.rows() + 1, row_ptr.begin());
    std::copy(Ac.innerIndexPtr(), Ac.innerIndexPtr() + Ac.nonZeros(), col_idx.begin());
    std::copy(Ac.valuePtr(), Ac.valuePtr() + Ac.nonZeros(), values.begin());

    AMGX_matrix_create(&Amat, rsrc, AMGX_mode_fDDI);
    AMGX_matrix_upload_all(Amat, n, Ac.nonZeros(), 1,
        row_ptr.data(), col_idx.data(), values.data(), nullptr);

    AMGX_vector_create(&Ax, rsrc, AMGX_mode_fDDI);
    AMGX_vector_create(&Ab, rsrc, AMGX_mode_fDDI);
}

void AmgXSolver::setup() {
    AMGX_solver_setup(solver, Amat);
}

void AmgXSolver::solve(const Eigen::VectorXf& b, Eigen::VectorXf& x) {
    if (x.size() != n) x.resize(n);

    AMGX_vector_upload(Ab, n, 1, b.data());
    AMGX_vector_upload(Ax, n, 1, x.data());

    AMGX_solver_solve(solver, Ab, Ax);

    AMGX_vector_download(Ax, x.data());
}
