// AmgXSolver.cpp
#include "AmgXSolver.hpp"
#include <iostream>

AmgXSolver::AmgXSolver() {
    AMGX_initialize();
    std::string configFile = std::string(AMGX_CONFIG_DIR) + "/amgx_config.txt";
    AMGX_config_create_from_file(&cfg, configFile.c_str());
    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_solver_create(&solver, rsrc, AMGX_mode_dFFI, cfg);
    // dFFI = device, Float, integer indicies
    AMGX_matrix_create(&Amat, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&Ax, rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&Ab, rsrc, AMGX_mode_dFFI);
}

AmgXSolver::~AmgXSolver() {
    // std::cout << "AMGX Solver Destructor" << std::endl;
    AMGX_vector_destroy(Ax);
    // std::cout << "Destroyed Ax" << std::endl;
    AMGX_vector_destroy(Ab);
    // std::cout << "Destroyed Ab" << std::endl;
    AMGX_matrix_destroy(Amat);
    // std::cout << "Destroyed Amat" << std::endl;
    AMGX_solver_destroy(solver);
    // std::cout << "Destroyed solver" << std::endl;
    AMGX_resources_destroy(rsrc);
    // std::cout << "Destroyed rsrc" << std::endl;
    AMGX_config_destroy(cfg);
    // std::cout << "AMGX Vars Freed" << std::endl;
    AMGX_finalize();
}

void AmgXSolver::uploadMatrix(const Eigen::SparseMatrix<float, Eigen::RowMajor>& A) {

    this->uploadMatrix(A.rows(),A.cols(),A.nonZeros(),A.valuePtr(),A.outerIndexPtr(), A.innerIndexPtr());
}

void AmgXSolver::uploadMatrix(int rows, int cols, int nnz,const float* data,const int* rowPtr,const int* colIdx) {
    /* Guarding against memory leaking*/
    this->rows = rows;
    this->cols = cols;
    this->nnz = nnz;
    if (this->Amat) {
        AMGX_matrix_destroy(this->Amat);
        this->Amat = nullptr;
    }
    if (this->Ax) { AMGX_vector_destroy(this->Ax); this->Ax = nullptr; }
    if (this->Ab) { AMGX_vector_destroy(this->Ab); this->Ab = nullptr; }

    // std::cout << "Memory leak cleaned up" << std::endl;

    // std::cout << "Creating AMGX Matrix" << std::endl;
    AMGX_matrix_create(&(this->Amat), rsrc, AMGX_mode_dFFI);
    // std::cout << "Uploading AMGX Matrix" << std::endl;
    AMGX_matrix_upload_all(this->Amat,
        this->rows,
        this->nnz,                // number of nonzeros
        1,                 // block_dimx
        1,                // block_dimy
        rowPtr,                  // row_ptr
        colIdx,                  // col_idx
        data,                       // values
        nullptr);   
    // std::cout << "Creating AMGX Vector" << std::endl;
    AMGX_vector_create(&(this->Ax), rsrc, AMGX_mode_dFFI);
    AMGX_vector_create(&(this->Ab), rsrc, AMGX_mode_dFFI);
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
    if (x.size() != this->rows) x.resize(this->rows);

    this->solve(b.data(), x.data());
}

void AmgXSolver::solve(const float* b, float * x) {
    
    AMGX_vector_upload(this->Ab, rows, 1, b);
    AMGX_vector_upload(this->Ax, rows, 1, x);

    AMGX_solver_solve(solver, this->Ab, this->Ax);

    AMGX_vector_download(this->Ax, x);
}
