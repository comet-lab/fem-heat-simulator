#include "TimeIntegrators/GPUTimeIntegrator.cuh"

// =====================
// Constructor / Destructor
// =====================
GPUTimeIntegrator::GPUTimeIntegrator(const ThermalModel& thermalModel, const GlobalMatrices& globalMatrices, float alpha, float deltat)
    : TimeIntegrator(thermalModel, globalMatrices, alpha, deltat), 
      amgxSolver_(AmgXSolver::getInstance())
{
    CHECK_CUSPARSE(cusparseCreate(&handle_));
    uploadAllMatrices();
    std::cout << "Matrices Uploaded" << std::endl;
}

GPUTimeIntegrator::~GPUTimeIntegrator() {
    // std::cout << "GPU Solver Destructor" << std::endl;
    // Free device memory 
    freeModelMatrices();
    // Destroy cuSPARSE handle
    if (handle_) {
        CHECK_CUSPARSE(cusparseDestroy(handle_));    
    }
}

void GPUTimeIntegrator::freeModelMatrices()
{
    // Frees all variables set by the FEM Model
    CHECK_CUDA( cudaFree(globF_d_) );
    globF_d_ = nullptr;
    CHECK_CUDA( cudaFree(vVec_d_) );
    vVec_d_ = nullptr;

    // Free CSR/Vec descriptors
    freeCSR(M_d_);
    freeCSR(K_d_);
    freeCSR(Q_d_);
    freeCSR(Fint_d_);
    freeCSR(FintElem_d_);
    freeCSR(Fconv_d_);
    freeCSR(Fk_d_);
    // Free global matrices
    freeCSR(globK_d_);
    freeCSR(globM_d_);

    freeDeviceVec(FluenceRate_d_);
    freeDeviceVec(FluenceRateElem_d_);
    freeDeviceVec(Fq_d_);
    freeDeviceVec(Fflux_d_);
    freeDeviceVec(Temp_d_);

    freeDeviceVec(dVec_d_);  
}

// =====================
// Apply Parameters
// =====================
void GPUTimeIntegrator::applyParameters()
{
    // upload the fluence rate vector to GPU from thermalModel before applying parameters
    // because of current setup, applyParameters doesn't know if fluence rate or just coefficients were changed
    // so we update both. 
    uploadFluenceRate();
    // These variables get set in here so we want to make sure they are free
    freeCSR(globK_d_);
    freeCSR(globM_d_);
    if (globF_d_)
    {
        CHECK_CUDA(cudaFree(globF_d_));
        globF_d_ = nullptr; // <--- IMPORTANT
    }
    long nRows = K_d_.rows;
    // ---------------- Step 1: Compute globK ----------------
    // This will perform Kint*TC + Kconv*HTC = globK
    // std::cout << "Apply Parameters Step 1" << std::endl;
    addSparse(K_d_, thermalModel_.TC, Q_d_, thermalModel_.HTC, globK_d_);
    // addSparse naturally will define globK_d_ properly as a DeviceCSR

    // ---------------- Step 2: Set globM ----------------
    // this will perform M*VHC + M*0 = globM
    // std::cout << "Apply Parameters Step 2" << std::endl;
    addSparse(M_d_, thermalModel_.VHC, M_d_, 0, globM_d_); // Already scaled
    // addSparse naturally will define globM_d properly as a DeviceCSR

    // ---------------- Step 3: Compute Firr ----------------
    // std::cout << "Apply Parameters Step 3" << std::endl;
    float *Firr_d = nullptr;
    float* Firr_elem_d = nullptr;

    // nodal fluence rate vector
    CHECK_CUDA(cudaMalloc((void **)&Firr_d, nRows * sizeof(float)));
    multiplySparseVector(Fint_d_, FluenceRate_d_, Firr_d);

    //elemental fluence rate vector
    CHECK_CUDA(cudaMalloc((void**)&Firr_elem_d, nRows * sizeof(float)));
    multiplySparseVector(FintElem_d_, FluenceRateElem_d_, Firr_elem_d);
    
    // ---------------- Step 4: Compute globF ----------------
    // std::cout << "Apply Parameters Step 4" << std::endl;
    // Firr = Firr*MUA;
    scaleVector(Firr_d, thermalModel_.MUA, nRows);
    if (!globF_d_)
        CHECK_CUDA(cudaMalloc((void **)&globF_d_, nRows * sizeof(float)));

    // globF = Firr*MUA
    CHECK_CUDA(cudaMemcpy(globF_d_, Firr_d, nRows * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    // globF += FirrElem*MUA
    addVectors(globF_d_, Firr_elem_d, globF_d_, nRows, thermalModel_.MUA);
    // std::cout << "Element Fluence Rate" << std::endl;
    // globF += Fq*HTC*Tinf --> added convection due to ambient temp
    addVectors(globF_d_, Fq_d_.data, globF_d_, nRows, thermalModel_.HTC * thermalModel_.ambientTemp);
    // std::cout << "Ambient temp convection" << std::endl;
    // globF += Fflux*qn --> added flux at the boundary
    addVectors(globF_d_, Fflux_d_.data, globF_d_, nRows, thermalModel_.heatFlux);
    // std::cout << "Heat Flux" << std::endl;

    // globF += Fconv*Temp*HTC --> Forcing vector due to dirichlet on convection
    float* Fconv_d_temp = nullptr; // temporary device storage for FConv*Temp
    CHECK_CUDA(cudaMalloc((void**)&Fconv_d_temp, nRows * sizeof(float))); // allocate rows
    multiplySparseVector(Fconv_d_, Temp_d_, Fconv_d_temp); // perform multiplication

    addVectors(globF_d_, Fconv_d_temp, globF_d_, nRows, thermalModel_.HTC); // adds to F
    // std::cout << "Convection due to current temp at dirichlet nodes" << std::endl;

    // globF += F_k*Temp --> Forcing vector due to dirichlet on conduction
    float* Fk_d_temp = nullptr; // temporary device storage for FConv*Temp
    CHECK_CUDA(cudaMalloc((void**)&Fk_d_temp, nRows * sizeof(float))); // allocate rows
    multiplySparseVector(Fk_d_, Temp_d_, Fk_d_temp); // perform multiplication

    addVectors(globF_d_, Fk_d_temp, globF_d_, nRows, thermalModel_.TC); // adds to F
    // std::cout << "Conduction due to current temp at dirichlet nodes" << std::endl;

    // ---------------- Step 5: Free temporary GPU vectors ----------------
    // std::cout << "Apply Parameters Step 5" << std::endl;
    CHECK_CUDA(cudaFree(Firr_d));
    Firr_d = nullptr;
    CHECK_CUDA(cudaFree(Firr_elem_d));
    Firr_elem_d = nullptr;
    CHECK_CUDA(cudaFree(Fconv_d_temp));
    Fconv_d_temp = nullptr;
    CHECK_CUDA(cudaFree(Fk_d_temp));
    Fk_d_temp = nullptr;
}

void GPUTimeIntegrator::initialize(){
    /*
    This function will initialize the d and v vectors used for Euler integration. The d vector is
    simply passed in, and the v vector is assigned through the equation M*v = (F - K*d).
    */
    // -- Step 0: Upload dVec to GPU and save as attribute -- 
    applyParameters();
    int nRows = globalMatrices_.nNonDirichlet;
    vVec_ = Eigen::VectorXf::Zero(nRows);
    dVec_.resize(nRows);
    dVec_ = thermalModel_.Temp(globalMatrices_.validNodes);
    
    // std::cout << "dVec has size: " << nRows << std::endl;
    uploadVector(dVec_,dVec_d_);
    // -- Step1: Construct Right Hand Side of Ax = b
    // In this case b = (F - K*d);
    // std::cout << "Initialize Step 1" << std::endl;
    float* b_d;
    calculateRHS(b_d, nRows);

    // -- Step 2: construct Left Hand Side of Ax = b and put into AMGX
    // In this case it is just A = M so we run setup with alpha = 0 and dt = 0;
    // std::cout << "Initialize Step 2" << std::endl;
    amgxSolver_.uploadMatrix(globM_d_.rows, globM_d_.cols, globM_d_.nnz,
        globM_d_.data_d, globM_d_.rowPtr_d, globM_d_.colIdx_d);
    // call setup
    amgxSolver_.setup();

    // -- Step 3: Solve using AMGX so we can have a linear solver
    // std::cout << "Initialize Step 3" << std::endl;
    // Allocate vVec_d_ once (device storage for v)
    if (vVec_d_ != nullptr) {
        CHECK_CUDA(cudaFree(vVec_d_));
        vVec_d_ = nullptr;
    }
    CHECK_CUDA(cudaMalloc((void**)&vVec_d_, nRows * sizeof(float)));

    // Solve directly into vVec_d_
    amgxSolver_.solve(b_d, vVec_d_);    

    // -- Step 4a: Assign attribute to solution and return to eigen
    // Copy to host
    downloadvVec_d();

    // -- Step 4b: initialize the LHS of the actual Euler equation for the first time step
    updateLHS();

    // -- Cleanup
    CHECK_CUDA( cudaFree(b_d) ); // free memory of RHS which was temporary
    b_d = nullptr;
}

void GPUTimeIntegrator::updateLHS(){
    /*
    This function initializes the A matrix in AMGX in order to solve the sparse linear system
                Ax = b
    In this case, A is built from the finite element matrices M and F, based on the equation
                (M + alpha_*dt*F)v = (F - K*d)
    --- This function needs to be called before solve. Once setup has been called, solve
    --- can be called repeatedly without need to call setup again. However, if applyParameters()
    --- is called, then setup needs to be called before solve again. 
    */
    // -- Step 1: Construct A for Ax = b
    DeviceCSR A;
    // Perform the addition of A = (M + alpha_*deltaT*K)
    addSparse(globM_d_, 1, globK_d_, alpha_*dt_, A);
    // Upload A to the amgxSolver_
    amgxSolver_.uploadMatrix(A.rows, A.cols, A.nnz,
        A.data_d, A.rowPtr_d, A.colIdx_d);
    // call setup
    amgxSolver_.setup();

    // clean up A since its been uploaded into AMGX
    freeCSR(A);
}

void GPUTimeIntegrator::singleStep(){
    /* This function performs a single step of Euler Integration. For each step, we perform 
    an Explicit Step and Implicit Step
    --- Explicit Step:
        This is given by d = d + (1-alpha_)*dt*v.
        We assume that an initial value for v and d have already been created (either by initializeDV 
        or by a previous call to singleStep())
    --- Implicit Step:
        In the implicit step we first solve the linear equation Ax = b where A was initialized in the
        function setup(), and b is set to (F-K*d). The solution vector, x, is our new value for v. 
        Then we simply update d with d = d + alpha_*dt*v
    */

    // --- Function Assumes GPUTimeIntegrator::setup() has already been called ----
	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	int nRows = globK_d_.rows;
    // -- Step 1: Perform Explicit Step
	// Explicit Forward Step (only if alpha_ < 1). Uses stored value of vVec
	if (alpha_ < 1) {
        // performs the the addition for d = d + (1-alpha)*deltaT*v
        addVectors(dVec_d_.data,vVec_d_,dVec_d_.data,nRows,(1-alpha_)*dt_);
	}

    // -- Step 2: Perform Implicit Step
    // -- Step 2a: Calculate vVec using Ax = b --> (M + alpha*dt*K)v = (F - K*d);
    float* b_d; // temporary variable to store (F - K*d);
    calculateRHS(b_d,nRows);

    amgxSolver_.solve(b_d,vVec_d_); // solve for v

    // -- Step 2b: Add implicit step to dVec --> dVec = dVec + alpha*dt*vVec
    addVectors(dVec_d_.data, vVec_d_ ,dVec_d_.data,nRows,(alpha_)*dt_);

    // -- Cleanup
    CHECK_CUDA( cudaFree(b_d) ); // free memory of RHS which was temporary
    b_d = nullptr;
}

Eigen::VectorXf GPUTimeIntegrator::singleStepWithUpdate()
{
    singleStep();
    downloaddVec_d();
    return dVec_;
}

void GPUTimeIntegrator::calculateRHS(float* &b_d, int nRows){
    CHECK_CUDA( cudaMalloc(&b_d, sizeof(float)*nRows) ); // allocate memory for RHS
    // Perform b_d = K*d
    multiplySparseVector(globK_d_, dVec_d_, b_d); 
    // Perform b_d = F - K*d
    addVectors(globF_d_, b_d, b_d, nRows, -1); // scale is negative one to subtract v2 from v1
}

// =====================
// Upload utilities
// =====================

/* --- Specific To Model --- */
void GPUTimeIntegrator::uploadAllMatrices() {

    uploadSparseMatrix(globalMatrices_.M, M_d_);
    uploadSparseMatrix(globalMatrices_.K, K_d_);
    uploadSparseMatrix(globalMatrices_.Q, Q_d_);
    uploadSparseMatrix(globalMatrices_.Fint, Fint_d_);
    uploadSparseMatrix(globalMatrices_.FintElem, FintElem_d_);
    uploadSparseMatrix(globalMatrices_.Fconv, Fconv_d_);
    uploadSparseMatrix(globalMatrices_.Fk, Fk_d_);

    uploadVector(globalMatrices_.Fflux, Fflux_d_);
    uploadVector(globalMatrices_.Fq, Fq_d_);
}

void GPUTimeIntegrator::uploaddVec_d()
{
    uploadVector(dVec_,dVec_d_);
}

void GPUTimeIntegrator::uploadFluenceRate()
{
    uploadVector(thermalModel_.fluenceRate, FluenceRate_d_);
    uploadVector(thermalModel_.fluenceRateElem, FluenceRateElem_d_);
    uploadVector(thermalModel_.Temp, Temp_d_);
    std::cout << "Fluence Rate Parameters Uploaded" << std::endl;
}

void GPUTimeIntegrator::downloaddVec_d()
{
    downloadVector(dVec_, dVec_d_.data);
}

void GPUTimeIntegrator::downloadvVec_d()
{
    downloadVector(vVec_, vVec_d_);
}


/* --- General Upload and Download Functions ---  */


void GPUTimeIntegrator::uploadSparseMatrix(const Eigen::SparseMatrix<float,Eigen::RowMajor>& A, DeviceCSR& dA){
    // Just in case it wasn't passed in compressed.
    // A.makeCompressed(); we are passing in const so we assume it is already compressed
    //Parameter setting
    int numRows = A.rows();
    int numCols = A.cols();
    int nnz = A.nonZeros();

    uploadSparseMatrix(numRows,numCols,nnz,A.outerIndexPtr(),A.innerIndexPtr(),A.valuePtr(),dA);
}

void GPUTimeIntegrator::uploadSparseMatrix(int numRows, int numCols, int nnz, const int* rowPtr, const int* columnIdx, const float* values, DeviceCSR& dA){
    // Just in case it wasn't passed in compressed.
    //uploads a sparse matrix with individual components instead of Eigen matrix
    freeCSR(dA);

    // set DeviceCSR parameters
    dA.rows = numRows;
    dA.cols = numCols;
    dA.nnz = nnz;

    // -- Step 1: Allocating new memory
    CHECK_CUDA( cudaMalloc((void**) &dA.rowPtr_d, (numRows + 1) * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA.colIdx_d, nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA.data_d,  nnz * sizeof(float))      )
    
    // -- Step 2: Copying to memory
    if (rowPtr != nullptr)
        CHECK_CUDA( cudaMemcpy(dA.rowPtr_d, rowPtr,
                           (numRows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    if (columnIdx != nullptr)
        CHECK_CUDA( cudaMemcpy(dA.colIdx_d, columnIdx, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    if (values != nullptr)
        CHECK_CUDA( cudaMemcpy(dA.data_d, values, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )

    // -- Step 3: Upload   
    CHECK_CUSPARSE( cusparseCreateCsr(&dA.spMatDescr, numRows, numCols, nnz,
                                      dA.rowPtr_d, dA.colIdx_d, dA.data_d,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );

}

void GPUTimeIntegrator::uploadVector(const Eigen::VectorXf& v, DeviceVec& dV) {
    uploadVector(v.data(),v.size(),dV);
}

void GPUTimeIntegrator::uploadVector(const float* data, int n, DeviceVec& dV){
    
    freeDeviceVec(dV);
    //Allocate memory
    CHECK_CUDA(cudaMalloc((void**)&dV.data, n*sizeof(float)));
    //Copy Memory
    CHECK_CUDA(cudaMemcpy(dV.data, data, n*sizeof(float), cudaMemcpyHostToDevice));
    // Create Handle
    CHECK_CUSPARSE( cusparseCreateDnVec(&dV.vecHandle, n, dV.data, CUDA_R_32F) )

    // printVector(dV.data, n);
}

void GPUTimeIntegrator::downloadVector(Eigen::VectorXf &v,const float *dv)
{
    CHECK_CUDA(cudaMemcpy(v.data(),dv,v.size() * sizeof(float),cudaMemcpyDeviceToHost) )
}

void GPUTimeIntegrator::downloadSparseMatrix(Eigen::SparseMatrix<float,Eigen::RowMajor>& outMat, const DeviceCSR& source)
{
    // -- Make sure receiving matrix has an appropriate amount of space
    outMat.resize(source.rows, source.cols);
    // allocate nnz slots
    outMat.reserve(source.nnz);
    // finalize CSR structure to allocate value/inner arrays
    outMat.makeCompressed();

    // -- Copy the necessary components
    CHECK_CUDA( cudaMemcpy(outMat.outerIndexPtr(), source.rowPtr_d,
                           (source.rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )

    CHECK_CUDA( cudaMemcpy(outMat.innerIndexPtr(), source.colIdx_d, source.nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )

    CHECK_CUDA( cudaMemcpy(outMat.valuePtr(), source.data_d, source.nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
}


void GPUTimeIntegrator::freeCSR(DeviceCSR& dA) {
    if (dA.rowPtr_d) {
        CHECK_CUDA(cudaFree(dA.rowPtr_d));
        dA.rowPtr_d = nullptr;
    }
    if (dA.colIdx_d) {
        CHECK_CUDA(cudaFree(dA.colIdx_d));
        dA.colIdx_d = nullptr;
    }
    if (dA.data_d) {
        CHECK_CUDA(cudaFree(dA.data_d));
        dA.data_d = nullptr;
    }
    if (dA.spMatDescr) {
        cusparseDestroySpMat(dA.spMatDescr);
        dA.spMatDescr = nullptr;
    }
    dA.rows = 0;
    dA.cols = 0;
    dA.nnz  = 0;
}

void GPUTimeIntegrator::freeDeviceVec(DeviceVec& vec) {
    if (vec.vecHandle){
        cusparseDestroyDnVec(vec.vecHandle);
        vec.vecHandle = nullptr;
    }
    if (vec.data){
        CHECK_CUDA(cudaFree(vec.data) )
        vec.data = nullptr;
    }
}

////////////////////////////
// MATH UTILITIES
////////////////////////////

void GPUTimeIntegrator::addSparse(const DeviceCSR& A, float alpha, const DeviceCSR& B, float beta, DeviceCSR& C) {
     // Matrix A and B should already be uploaded into the system. C is where we will store the solution
    // need to upload the location of our solution which should be initially empty
    
    freeCSR(C); // Just to make sure it is cleaned before we write into it. 

    CHECK_CUDA(cudaMalloc((void**)&C.rowPtr_d, sizeof(int) * (A.rows + 1)));
    // create the legacy MatDescr
    cusparseMatDescr_t descrA, descrB, descrC;
    cusparseCreateMatDescr(&descrA);
    cusparseCreateMatDescr(&descrB);
    cusparseCreateMatDescr(&descrC);
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO))
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO))

    // --- Step 1: Work estimation ---
    size_t bufferSize1 = 0;
    char*  dBuffer1   = nullptr;
    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(cusparseScsrgeam2_bufferSizeExt(
            handle_, A.rows, A.cols,
            &alpha,
            descrA, A.nnz, A.data_d, A.rowPtr_d, A.colIdx_d,
            &beta,
            descrB, B.nnz, B.data_d, B.rowPtr_d, B.colIdx_d,
            descrC, nullptr, C.rowPtr_d, nullptr,
            &bufferSize1)
         )

    CHECK_CUDA(cudaMalloc(&dBuffer1, bufferSize1));
    

    // -- Step 2: Compute number of nonzeros in C --
    // This is currently general, but it may be possible to know nnz and placements beforehand,
    // allowing us to skip this call
    int nnzC = 0;
    int* dNnzTotal = nullptr;
    CHECK_CUDA(cudaMalloc(&dNnzTotal, sizeof(int)));
    // std::cout << "Doing the actual addition" << std::endl;
    CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(
        handle_, A.rows, A.cols,
        descrA, A.nnz, A.rowPtr_d, A.colIdx_d,
        descrB, B.nnz, B.rowPtr_d, B.colIdx_d,
        descrC,
        C.rowPtr_d, dNnzTotal,
        dBuffer1)
    )
    // Copy nnzC back from device
    CHECK_CUDA(cudaMemcpy(&nnzC, dNnzTotal, sizeof(int), cudaMemcpyDeviceToHost));
    C.nnz = nnzC;

    // --- Step 3: Allocate the colIdx and rowPtr variables ---
    CHECK_CUDA( cudaMalloc((void**)&C.colIdx_d, sizeof(int)*C.nnz) )
    CHECK_CUDA( cudaMalloc((void**)&C.data_d, sizeof(float)*C.nnz) )

    // -- Step 4 : Perfrom addition

    CHECK_CUSPARSE( cusparseScsrgeam2(
        handle_, A.rows, A.cols,
        &alpha,
        descrA, A.nnz, A.data_d, A.rowPtr_d, A.colIdx_d,
        &beta,
        descrB, B.nnz, B.data_d, B.rowPtr_d, B.colIdx_d,
        descrC, C.data_d, C.rowPtr_d, C.colIdx_d,
        dBuffer1)
    )

    C.rows = A.rows;
    C.cols = A.cols;
    CHECK_CUSPARSE(cusparseCreateCsr(&C.spMatDescr,
                                     A.rows, A.cols, C.nnz,
                                     C.rowPtr_d, C.colIdx_d, C.data_d,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    // --- Cleanup ---
    cudaFree(dBuffer1);
    cudaFree(dNnzTotal);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroyMatDescr(descrB);
    cusparseDestroyMatDescr(descrC);
}

void GPUTimeIntegrator::multiplySparseVector(const DeviceCSR& A, DeviceVec& x, float* y) {
    float alpha = 1.0f;
    float beta  = 0.0f;

    // --- Step 1: Create dense vector descriptors for output y ---
    cusparseDnVecDescr_t  vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A.rows, y, CUDA_R_32F));

    // --- Step 2: Query buffer size ---
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.spMatDescr, x.vecHandle, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // --- Step 3: Analysis and computation ---
    CHECK_CUSPARSE(cusparseSpMV(
        handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.spMatDescr, x.vecHandle, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer));

    // --- Step 4: Cleanup ---
    cudaFree(dBuffer);
    cusparseDestroyDnVec(vecY);
}

bool GPUTimeIntegrator::solveSparseLinearSystem()
{

    int nRows = globK_d_.rows;

    if (alpha_ < 1) {
        // performs the the addition for d = d + (1-alpha)*dt_*v
        addVectors(dVec_d_.data,vVec_d_,dVec_d_.data,nRows,(1-alpha_)*dt_);
	}

    // -- Step 2: Perform Implicit Step
    DeviceCSR A;
    addSparse(globM_d_, 1, globK_d_, alpha_*dt_, A);

    // -- Step 2a: Calculate vVec using Ax = b --> (M + alpha_*dt*K)v = (F - K*d);
    float* b_d; // temporary variable to store (F - K*d);
    calculateRHS(b_d, nRows);

    float* x_d; // where we will store our temporary solution to v
    cudaMalloc(&x_d, sizeof(float) * nRows);

    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);
    int singularity;
    // Solve Ax = b (use LU for float)
    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusolverSpScsrlsvqr(handle, nRows,
                        A.nnz,
                        descrA, // default descr
                        A.data_d,
                        A.rowPtr_d,
                        A.colIdx_d,
                        b_d,
                        1e-6f, // tolerance
                        0,     // reorder: 0 = no reordering
                        x_d,
                        &singularity);
    
    if (singularity >= 0) {
        std::cerr << "Matrix is singular at row " << singularity << std::endl;
    return false;
    }

    // -- Step 2b: Add implicit step to dVec --> dVec = dVec + alpha_*dt*vVec
    addVectors(dVec_d_.data, x_d ,dVec_d_.data,nRows,(alpha_)*dt_);

    // -- Step 3: update our value for vVec
    CHECK_CUDA(cudaMemcpy(vVec_d_, x_d, nRows*sizeof(float), cudaMemcpyDeviceToDevice));

    // -- Cleanup
    CHECK_CUDA( cudaFree(b_d) ) // free memory of RHS which was temporary
    b_d = nullptr;
    CHECK_CUDA( cudaFree(x_d) ) // free memory of our solution since its been copied
    x_d = nullptr;
    freeCSR(A);
    
    return true;
}


// =====================
// Kernels and helpers
// =====================
__global__ void scaleCSRValues(float* data, float alpha, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) data[i] *= alpha;
}

void GPUTimeIntegrator::scaleCSR(DeviceCSR& dA, float alpha) {
    if (dA.nnz == 0) return;  // nothing to scale

    int threads = 256;
    int blocks = (dA.nnz + threads -1)/threads;
    scaleCSRValues<<<blocks, threads>>>(dA.data_d, alpha, dA.nnz);
    CUDA_KERNEL_CHECK();
}

__global__ void addVectorsKernel(const float* v1, const float* v2, float* out, int n, float vScale) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) out[i] = v1[i] + vScale*v2[i];
}

void GPUTimeIntegrator::addVectors(float* v1, float* v2, float* out, int n, float vScale) {
    int threads = 256;
    int blocks = (n + threads -1)/threads;
    addVectorsKernel<<<blocks, threads>>>(v1, v2, out, n, vScale);
    CUDA_KERNEL_CHECK();
}

__global__ void scaleVectorValues(float* data, float alpha, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) data[i] *= alpha;
}

void GPUTimeIntegrator::scaleVector(float* data, float alpha, int n) {
    int threads = 256;
    int blocks = (n + threads -1)/threads;
    scaleVectorValues<<<blocks, threads>>>(data, alpha, n);
    CUDA_KERNEL_CHECK();
}

__global__ void printKernel(float *d_data, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        printf("Device: d_data[%d] = %f\n", tid, d_data[tid]);
    }
}

void GPUTimeIntegrator::printVector(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads -1)/threads;
    printKernel<<<blocks, threads>>>(data, n);
    CUDA_KERNEL_CHECK();
}