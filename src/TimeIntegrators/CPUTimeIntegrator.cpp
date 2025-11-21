#include "TimeIntegrators/CPUTimeIntegrator.hpp"

void CPUTimeIntegrator::applyParameters()
{
	/*
	Applying MUA, TC, HTC, VHC, and the fluence rate to the matrices we built on the CPU.
	Here we assume constant tissue properties throughout the mesh. This allows us to
	multiply by tissue specific properties after the element construction, which means we can change
	tissue properties without having to reconstruct the matrices
	*/
	// auto startTime = std::chrono::steady_clock::now();

	// Apply parameter specific multiplication for each global matrix.
	// Conductivity matrix
	globK_ = globalMatrices_.K * thermalModel_.TC + globalMatrices_.Q * thermalModel_.HTC;
	globK_.makeCompressed();
	// Thermal mass matrix
	//globM_.setZero();
	globM_ = globalMatrices_.M * thermalModel_.VHC; // M Doesn't have any additions so we just multiply it by the constant
	globM_.makeCompressed();
	// Forcing Vector
	//globF_.setZero();
	globF_ = (globalMatrices_.Fq * thermalModel_.ambientTemp + globalMatrices_.Fconv * thermalModel_.Temp) * thermalModel_.HTC // convection from ambient temp and dirichlet nodes
		+ (globalMatrices_.Fflux * thermalModel_.heatFlux) // heat flux 
		+ (globalMatrices_.Fk * thermalModel_.Temp * thermalModel_.TC) // conduction on dirichlet nodes
		+ (globalMatrices_.Fint * thermalModel_.fluenceRate + globalMatrices_.FintElem * thermalModel_.fluenceRateElem) * thermalModel_.MUA; // forcing function
	//globF_.noalias() += (globalMatrices_.Fq * thermalModel_.ambientTemp + globalMatrices_.Fconv * thermalModel_.Temp) * thermalModel_.HTC; // convection from ambient temp and dirichlet nodes
	//globF_.noalias() += (globalMatrices_.Fflux * thermalModel_.heatFlux); // heat flux 
	//globF_.noalias() += (globalMatrices_.Fk * thermalModel_.Temp * thermalModel_.TC); // conduction on dirichlet nodes
	//globF_.noalias() += (globalMatrices_.Fint * thermalModel_.fluenceRate + globalMatrices_.FintElem * thermalModel_.fluenceRateElem) * thermalModel_.MUA; // forcing function
}

void CPUTimeIntegrator::setMatrixSparsity()
{
	globK_ = globalMatrices_.K + globalMatrices_.Q;
	globK_.makeCompressed();
	globM_ = globalMatrices_.M;
	globM_.makeCompressed();
	globF_ = Eigen::VectorXf::Zero(globalMatrices_.nNonDirichlet);
}

void CPUTimeIntegrator::initialize()
{
	/*
	Creates the d and v vectors used for time integration. V vector is created by solving the
	linear system M*v = (F-K*d). Then the Eigen conjugate gradient solver is initialized with the
	left-hand-side of the system (M + alpha*dt*K)*v = (F - K*d).
	*/
	auto startTime = std::chrono::steady_clock::now();
	setMatrixSparsity();
	applyParameters();

	/* PERFORMING TIME INTEGRATION USING EULER FAMILY */
	// Initialize d, v, and dTilde vectors
	dVec_.resize(globalMatrices_.nNonDirichlet);
	vVec_ = Eigen::VectorXf::Zero(globalMatrices_.nNonDirichlet);
	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	dVec_ = thermalModel_.Temp(globalMatrices_.validNodes);
	if (alpha_ < 1) {
		// Perform the conjugate gradiant to compute the initial vVec value
		// This is a bit odd because if the user hasn't specified the initial fluence rate it will be 0 initially. 
		// And can mess up the first few timesteps
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> initSolver;
		initSolver.compute(globM_);
		if (initSolver.info() != Eigen::Success) {
			std::cout << "Factorization failed!" << std::endl;
			return;
		}
		Eigen::VectorXf RHSinit = (globF_) - globK_ * dVec_;
		//Eigen::VectorXf RHSinit = Eigen::VectorXf::Constant(globM_.rows(), 1);
		//std::cout << "Right before v solve" << std::endl;
		vVec_ = initSolver.solve(RHSinit);
	} // if we are using backwards Euler we can skip this initial computation of vVec. It is only
	// needed for explicit steps. 

	// Prepare solver for future iterations
	LHS_ = globM_ + alpha_ * dt_ * globK_;
	LHS_.makeCompressed();
	//std::cout << "LHS setup" << std::endl;
	// These two steps form the cgSolver_.compute() function. By calling them separately, we 
	// only ever need to call factorize when the tissue properties change.
	cgSolver_.analyzePattern(LHS_);
	//std::cout << "LHS analyzed" << std::endl;
	cgSolver_.factorize(LHS_);
	//std::cout << "LHS factorized" << std::endl;
	if (cgSolver_.info() != Eigen::Success) {
		std::cout << "Decomposition Failed" << std::endl;
	}
}

void CPUTimeIntegrator::updateLHS()
{
	LHS_ = globM_ + alpha_ * dt_ * globK_; // Create new left hand side 
	LHS_.makeCompressed();
	cgSolver_.factorize(LHS_); // Perform factoriziation based on analysis which should have been called with initializeModel();
	if (cgSolver_.info() != Eigen::Success) {
		std::cout << "Decomposition Failed" << std::endl;
	}
}

void CPUTimeIntegrator::singleStep()
{
	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	dVec_ = thermalModel_.Temp(globalMatrices_.validNodes);

	//this->cgSolver_.factorize(this->LHS_); // Perform factoriziation based on analysis which should have been called with initializeModel();
	// Explicit Forward Step (only if alpha < 1)
	if (alpha_ < 1) {
		dVec_ = dVec_ + (1 - alpha_) * dt_ * vVec_; // normally the output of this equation is assigned to dTilde for clarity...
	}
	// Create Right-hand side of v(M + alpha*deltaT*K) = (F - K*dTilde);
	Eigen::VectorXf RHS = globF_ - globK_ * dVec_; // ... and dTilde would be used here
	// Solve Ax = b using conjugate gradient
	// Time derivative should not change too much between time steps so we can use previous v to initialize conjugate gradient. 
	vVec_ = cgSolver_.solveWithGuess(RHS, vVec_);
	//this->vVec = this->cgSolver_.solve(RHS);
	if (cgSolver_.info() != Eigen::Success) {
		std::cout << "Issue With Solver" << std::endl;
	}
	/*if (!this->silentMode) {
		std::cout << "Iterations: " << this->cgSolver_.iterations() << std::endl;
	}*/
	// Implicit Backward Step (only if alpha > 0) 
	dVec_ = dVec_ + alpha_ * dt_ * vVec_; // ... dTilde would also be on the righ-hand side here. 
}