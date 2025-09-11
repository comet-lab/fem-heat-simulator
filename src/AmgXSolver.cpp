// AmgXSolver.cpp
#include "AmgXSolver.hpp"
#include <iostream>

AmgXSolver::AmgXSolver(const std::string& configFile) {
    AMGX_initialize();
    // AMGX_initialize_plugins();
    AMGX_config_create_from_file(&cfg, configFile.c_str());
    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_solver_create(&solver, rsrc, AMGX_mode_dFFI, cfg);
    // dFFI = device, Float, integer indicies
    AMGX_matrix_create(&Amat, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&Ax, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&Ab, rsrc, AMGX_mode_dFFI);
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

    /* A must come in compressed*/

    /* Guarding against memory leaking*/
    if (Amat) {
        AMGX_matrix_destroy(Amat);
        Amat = nullptr;
    }
    if (Ax) { AMGX_vector_destroy(Ax); Ax = nullptr; }
    if (Ab) { AMGX_vector_destroy(Ab); Ab = nullptr; }

    // std::cout << "Memory leak cleaned up" << std::endl;

    /* Actual upload code*/
    // Eigen::SparseMatrix<float, Eigen::RowMajor> Ac = A;
    // A.makeCompressed();
    n_rows = A.rows();
    n_cols = A.cols();
    nnz = A.nonZeros();

    // std::vector<int> row_ptr(A.rows() + 1);
    // std::vector<int> col_idx(A.nonZeros());
    // std::vector<float> values(A.nonZeros());

    // std::cout << "Vectors created" << std::endl;

    // std::copy(A.outerIndexPtr(), A.outerIndexPtr() + A.rows() + 1, row_ptr.begin());
    // std::copy(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros(), col_idx.begin());
    // std::copy(A.valuePtr(), A.valuePtr() + A.nonZeros(), values.begin());

    // std::cout << "Vectors copied" << std::endl;

    // std::cout << "Creating AMGX Matrix" << std::endl;
    AMGX_matrix_create(&Amat, rsrc, AMGX_mode_dFFI);
    // std::cout << "Uploading AMGX Matrix" << std::endl;
    AMGX_matrix_upload_all(Amat,
        n_rows,
        nnz,                // number of nonzeros
        1,                 // block_dimx
        1,                // block_dimy
        A.outerIndexPtr(),                  // row_ptr
        A.innerIndexPtr(),                  // col_idx
        A.valuePtr(),                       // values
        nullptr);   
    // std::cout << "Creating AMGX Vector" << std::endl;
    AMGX_vector_create(&Ax, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&Ab, rsrc, AMGX_mode_dFFI);
}

// Later, if only coefficients change (same sparsity pattern)
void AmgXSolver::updateMatrixValues(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A) {
   /* Assumes A is already compressed */

    if (!Amat) {
        throw std::runtime_error("AMGX matrix not initialized with upload_all!");
    }

    AMGX_matrix_replace_coefficients(
        Amat,
        A.rows(),          // number of rows
        A.nonZeros(),      // number of nonzeros
        A.valuePtr(),      // pointer to updated coefficients
        nullptr);          // optional diag info (rarely needed)
}

void AmgXSolver::setup() {
    AMGX_solver_setup(solver, Amat);
}

void AmgXSolver::solve(const Eigen::VectorXf& b, Eigen::VectorXf& x) {
    if (x.size() != n_rows) x.resize(n_rows);

    AMGX_vector_upload(Ab, n_rows, 1, b.data());
    AMGX_vector_upload(Ax, n_rows, 1, x.data());

    AMGX_solver_solve(solver, Ab, Ax);

    AMGX_vector_download(Ax, x.data());
}
