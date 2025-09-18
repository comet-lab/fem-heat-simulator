#include "GPUSolver.cuh"

// =====================
// Constructor / Destructor
// =====================
GPUSolver::GPUSolver() {
    CHECK_CUSPARSE(cusparseCreate(&this->handle));
    this->amgxSolver = new AmgXSolver();
}

GPUSolver::~GPUSolver() {
    // std::cout << "GPU Solver Destructor" << std::endl;
    // Free device memory
    CHECK_CUDA( cudaFree(globF_d) );
    globF_d = nullptr;
    CHECK_CUDA( cudaFree(vVec_d) );
    vVec_d = nullptr;

    // Free CSR/Vec descriptors
    freeCSR(Kint_d);
    freeCSR(Kconv_d);
    freeCSR(M_d);
    freeCSR(FirrMat_d);
    // Free global matrices
    freeCSR(globK_d);
    freeCSR(globM_d);

    freeDeviceVec(FluenceRate_d);
    freeDeviceVec(Fq_d);
    freeDeviceVec(Fconv_d);
    freeDeviceVec(Fk_d);
    freeDeviceVec(FirrElem_d);
    freeDeviceVec(dVec_d);  

    // Destroy cuSPARSE handle
    if (handle) {
        CHECK_CUSPARSE(cusparseDestroy(handle));    
    }
    // Free AMGXSolver if allocated
    delete amgxSolver;
    amgxSolver = nullptr;
    // std::cout << "GPU Variables Freed" << std::endl;
}

// =====================
// Apply Parameters
// =====================
void GPUSolver::applyParameters(
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
) {
    // ---------------- Step 0: Upload data ----------------
    this->uploadAllMatrices(Kint, Kconv, M, FirrMat,
                                       FluenceRate, Fq, Fconv, Fk, FirrElem);
    
    this->applyParameters(TC, HTC, VHC, MUA, elemNFR);
    }

void GPUSolver::applyParameters(float TC, float HTC, float VHC, float MUA, bool elemNFR){

    // These variables get set in here so we want to make sure they are free
    freeCSR(this->globK_d);
    freeCSR(this->globM_d);
    if (this->globF_d) {
        CHECK_CUDA(cudaFree(this->globF_d));
        this->globF_d = nullptr;  // <--- IMPORTANT
    }

    // ---------------- Step 1: Compute globK ----------------
    // This will perform Kint*TC + Kconv*HTC = globK
    // std::cout << "Apply Parameters Step 1" << std::endl;
    addSparse(this->Kint_d, TC, this->Kconv_d, HTC, this->globK_d); 
    // addSparse naturally will define this->globK_d properly as a DeviceCSR

    // ---------------- Step 2: Set globM ----------------
    // this will perform M*VHC + M*0 = globM
    // std::cout << "Apply Parameters Step 2" << std::endl;
    addSparse(this->M_d, VHC, this->M_d, 0, this->globM_d); ; // Already scaled
    // addSparse naturally will define this->globM_d properly as a DeviceCSR

    // ---------------- Step 3: Compute Firr ----------------
    // std::cout << "Apply Parameters Step 3" << std::endl;
    float* Firr_d = nullptr;
    if (elemNFR) {
        Firr_d = this->FirrElem_d.data;
    } else {
        CHECK_CUDA(cudaMalloc((void**)&Firr_d, this->FirrMat_d.rows * sizeof(float)));
        multiplySparseVector(this->FirrMat_d, this->FluenceRate_d, Firr_d);
    }

    // ---------------- Step 4: Compute globF ----------------
    // std::cout << "Apply Parameters Step 4" << std::endl;
    // Firr = Firr*MUA;
    scaleVector(Firr_d, MUA, this->FirrMat_d.rows);

    if (!this->globF_d)
        CHECK_CUDA(cudaMalloc((void**)&this->globF_d, this->FirrMat_d.rows * sizeof(float)));

    // globF = Firr*MUA
    CHECK_CUDA(cudaMemcpy(this->globF_d, Firr_d, this->FirrMat_d.rows * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // globF = Firr*MUA + Fconv*HTC
    addVectors(this->globF_d, this->Fconv_d.data, this->globF_d, this->FirrMat_d.rows, HTC);
    // globF = Firr*MUA + Fconv*HTC + F_q
    addVectors(this->globF_d, this->Fq_d.data, this->globF_d, this->FirrMat_d.rows, 1);
    // globF = Firr*MUA + Fconv*HTC + F_q + Fk*TC
    addVectors(this->globF_d, this->Fk_d.data, this->globF_d, this->FirrMat_d.rows, TC);

    // ---------------- Step 5: Free temporary GPU vectors ----------------
    // std::cout << "Apply Parameters Step 5" << std::endl;
    CHECK_CUDA(cudaFree(Firr_d));
    Firr_d = nullptr;
}

void GPUSolver::initializeDV(const Eigen::VectorXf & dVec, Eigen::VectorXf& vVec){
    /*
    This function will initialize the d and v vectors used for Euler integration. The d vector is
    simply passed in, and the v vector is assigned through the equation M*v = (F - K*d).
    */
    // -- Step 0: Upload dVec to GPU and save as attribute -- 
    int nRows = dVec.size();
    this->uploadVector(dVec,this->dVec_d);
    // -- Step1: Construct Right Hand Side of Ax = b
    // In this case b = (F - K*d);
    // std::cout << "InitializeDV Step 1" << std::endl;
    float* b_d;
    this->calculateRHS(b_d, nRows);

    // -- Step 2: construct Left Hand Side of Ax = b
    // In this case it is just A = M so we run setup with alpha = 0 and dt = 0;
    // std::cout << "InitializeDV Step 2" << std::endl;
    this->setup(0,0); // alpha = 0, dt = 0 because we are doing forward euler essentially to get v

    // -- Step 3: Solve using AMGX so we can have a linear solver
    // std::cout << "InitializeDV Step 3" << std::endl;
    float* x_d; // where we will store our temporary solution to v
    cudaMalloc(&x_d, sizeof(float) * this->globM_d.cols);

    this->amgxSolver->solve(b_d,x_d);

    // -- Step 4: Assign attribute to solution and return to eigen
    // std::cout << "InitializeDV Step 4" << std::endl;
    CHECK_CUDA( cudaMemcpy(vVec.data(), x_d, nRows*sizeof(float),cudaMemcpyDeviceToHost) )
    // We aren't using upload vector here because upload vector assumes host-to-device copy
    // we don't want to free this data after either because it is a class attribute
    CHECK_CUDA(cudaMalloc((void**)&(this->vVec_d), nRows*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(this->vVec_d, x_d, nRows*sizeof(float), cudaMemcpyDeviceToDevice));

    // -- Cleanup
    CHECK_CUDA( cudaFree(b_d) ) // free memory of RHS which was temporary
    b_d = nullptr;
    CHECK_CUDA( cudaFree(x_d) ) // free memory of our solution since its been copied
    x_d = nullptr;
}

void GPUSolver::setup(float alpha, float deltaT){
    /*
    This function initializes the A matrix in AMGX in order to solve the sparse linear system
                Ax = b
    In this case, A is built from the finite element matrices M and F, based on the equation
                (M + alpha*dt*F)v = (F - K*d)
    --- This function needs to be called before solve. Once setup has been called, solve
    --- can be called repeatedly without need to call setup again. However, if applyParameters()
    --- is called, then setup needs to be called before solve again. 
    */
    // -- Step 1: Construct A for Ax = b
    DeviceCSR A;
    // Perform the addition of A = (M + alpha*deltaT*F)
    this->addSparse(this->globM_d, 1, this->globK_d, alpha*deltaT, A);
    // Upload A to the amgxSolver
    this->amgxSolver->uploadMatrix(A.rows, A.cols, A.nnz,
        A.data_d, A.rowPtr_d, A.colIdx_d);
    // call setup
    this->amgxSolver->setup();

    // clean up A since its been uploaded into AMGX
    this->freeCSR(A);
}

void GPUSolver::singleStep(float alpha, float deltaT){
    /* This function performs a single step of Euler Integration. For each step, we perform 
    an Explicit Step and Implicit Step
    --- Explicit Step:
        This is given by d = d + (1-alpha)*dt*v.
        We assume that an initial value for v and d have already been created (either by initializeDV 
        or by a previous call to singleStep())
    --- Implicit Step:
        In the implicit step we first solve the linear equation Ax = b where A was initialized in the
        function setup(), and b is set to (F-K*d). The solution vector, x, is our new value for v. 
        Then we simply update d with d = d + alpha*dt*v
    */

    // --- Function Assumes GPUSolver::setup() has already been called ----
	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	int nRows = this->globK_d.rows;
    // -- Step 1: Perform Explicit Step
	// Explicit Forward Step (only if alpha < 1). Uses stored value of vVec
	if (alpha < 1) {
        // performs the the addition for d = d + (1-alpha)*deltaT*v
        this->addVectors(this->dVec_d.data,this->vVec_d,this->dVec_d.data,nRows,(1-alpha)*deltaT);
	}

    // -- Step 2: Perform Implicit Step

    // -- Step 2a: Calculate vVec using Ax = b --> (M + alpha*dt*K)v = (F - K*d);
    float* b_d; // temporary variable to store (F - K*d);
    CHECK_CUDA( cudaMalloc(&b_d, sizeof(float)*nRows) ) // allocate memory for RHS
    // Multiplication of K*d --> b_d
    this->multiplySparseVector(this->globK_d, this->dVec_d, b_d); 
    // Subtraction of (F - b_d) --> b_d
    this->addVectors(this->globF_d, b_d, b_d, nRows, -1); // scale is negative one to subtract v2 from v1

    float* x_d; // where we will store our temporary solution to v
    cudaMalloc(&x_d, sizeof(float) * nRows);

    this->amgxSolver->solve(b_d,x_d); // solve for v

    // -- Step 2b: Add implicit step to dVec --> dVec = dVec + alpha*dt*vVec
    this->addVectors(this->dVec_d.data, x_d ,this->dVec_d.data,nRows,(alpha)*deltaT);

    // -- Step 3: update our value for vVec
    CHECK_CUDA(cudaMemcpy(this->vVec_d, x_d, nRows*sizeof(float), cudaMemcpyDeviceToDevice));

    // -- Cleanup
    CHECK_CUDA( cudaFree(b_d) ) // free memory of RHS which was temporary
    b_d = nullptr;
    CHECK_CUDA( cudaFree(x_d) ) // free memory of our solution since its been copied
    x_d = nullptr;
}

void GPUSolver::calculateRHS(float* &b_d, int nRows){
    CHECK_CUDA( cudaMalloc(&b_d, sizeof(float)*nRows) ) // allocate memory for RHS
    // Perform b_d = K*d
    this->multiplySparseVector(this->globK_d, this->dVec_d, b_d); 
    // Perform b_d = F - K*d
    this->addVectors(this->globF_d, b_d, b_d, nRows, -1); // scale is negative one to subtract v2 from v1
}

// =====================
// Upload utilities
// =====================
void GPUSolver::uploadAllMatrices(
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& Kint,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& Kconv,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& M,
    const Eigen::SparseMatrix<float, Eigen::RowMajor>& FirrMat,
    const Eigen::VectorXf& FluenceRate,
    const Eigen::VectorXf& Fq,
    const Eigen::VectorXf& Fconv,
    const Eigen::VectorXf& Fk,
    const Eigen::VectorXf& FirrElem
) {

    uploadSparseMatrix(Kint, this->Kint_d);
    uploadSparseMatrix(Kconv, this->Kconv_d);
    uploadSparseMatrix(M, this->M_d);
    uploadSparseMatrix(FirrMat, this->FirrMat_d);

    uploadVector(FluenceRate, this->FluenceRate_d);
    uploadVector(Fq, this->Fq_d);
    uploadVector(Fconv,this->Fconv_d);
    uploadVector(Fk, this->Fk_d);
    uploadVector(FirrElem, this->FirrElem_d);
}

void GPUSolver::uploadSparseMatrix(const Eigen::SparseMatrix<float,Eigen::RowMajor>& A, DeviceCSR& dA){
    // Just in case it wasn't passed in compressed.
    // A.makeCompressed(); we are passing in const so we assume it is already compressed
    //Parameter setting
    int numRows = A.rows();
    int numCols = A.cols();
    int nnz = A.nonZeros();

    this->uploadSparseMatrix(numRows,numCols,nnz,A.outerIndexPtr(),A.innerIndexPtr(),A.valuePtr(),dA);
}

void GPUSolver::uploadSparseMatrix(int numRows, int numCols, int nnz, const int* rowPtr, const int* columnIdx, const float* values, DeviceCSR& dA){
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

void GPUSolver::uploadVector(const Eigen::VectorXf& v, DeviceVec& dV) {
    this->uploadVector(v.data(),v.size(),dV);
}

void GPUSolver::uploadVector(const float* data, int n, DeviceVec& dV){
    
    freeDeviceVec(dV);
    //Allocate memory
    CHECK_CUDA(cudaMalloc((void**)&dV.data, n*sizeof(float)));
    //Copy Memory
    CHECK_CUDA(cudaMemcpy(dV.data, data, n*sizeof(float), cudaMemcpyHostToDevice));
    // Create Handle
    CHECK_CUSPARSE( cusparseCreateDnVec(&dV.vecHandle, n, dV.data, CUDA_R_32F) )

    // this->printVector(dV.data, n);
}

void GPUSolver::downloadVector(Eigen::VectorXf &v,const float *dv)
{
    CHECK_CUDA(cudaMemcpy(v.data(),dv,v.size() * sizeof(float),cudaMemcpyDeviceToHost) )
}

void GPUSolver::downloadSparseMatrix(Eigen::SparseMatrix<float,Eigen::RowMajor>& outMat, const DeviceCSR& source)
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


void GPUSolver::freeCSR(DeviceCSR& dA) {
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

void GPUSolver::freeDeviceVec(DeviceVec& vec) {
    if (vec.data){
        CHECK_CUDA(cudaFree(vec.data) )
        vec.data = nullptr;
    }
    if (vec.vecHandle){
        cusparseDestroyDnVec(vec.vecHandle);
        vec.vecHandle = nullptr;
    }
}

void GPUSolver::addSparse(const DeviceCSR& A, float alpha, const DeviceCSR& B, float beta, DeviceCSR& C) {
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
            this->handle, A.rows, A.cols,
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
        this->handle, A.rows, A.cols,
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
        this->handle, A.rows, A.cols,
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

void GPUSolver::multiplySparseVector(const DeviceCSR& A, DeviceVec& x, float* y) {
    float alpha = 1.0f;
    float beta  = 0.0f;

    // --- Step 1: Create dense vector descriptors for output y ---
    cusparseDnVecDescr_t  vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A.rows, y, CUDA_R_32F));

    // --- Step 2: Query buffer size ---
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        this->handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.spMatDescr, x.vecHandle, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // --- Step 3: Analysis and computation ---
    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A.spMatDescr, x.vecHandle, &beta, vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        dBuffer));

    // --- Step 4: Cleanup ---
    cudaFree(dBuffer);
    cusparseDestroyDnVec(vecY);
}

// =====================
// Kernels and helpers
// =====================
__global__ void scaleCSRValues(float* data, float alpha, int nnz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) data[i] *= alpha;
}

void GPUSolver::scaleCSR(DeviceCSR& dA, float alpha) {
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

void GPUSolver::addVectors(float* v1, float* v2, float* out, int n, float vScale) {
    int threads = 256;
    int blocks = (n + threads -1)/threads;
    addVectorsKernel<<<blocks, threads>>>(v1, v2, out, n, vScale);
    CUDA_KERNEL_CHECK();
}

__global__ void scaleVectorValues(float* data, float alpha, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) data[i] *= alpha;
}

void GPUSolver::scaleVector(float* data, float alpha, int n) {
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

void GPUSolver::printVector(float* data, int n) {
    int threads = 256;
    int blocks = (n + threads -1)/threads;
    printKernel<<<blocks, threads>>>(data, n);
    CUDA_KERNEL_CHECK();
}