#include "TimeIntegrator.hpp"
#include "CPUTimeIntegrator.hpp"

void CPUTimeIntegrator::initialize()
{
	dVec_.resize(mb_->nNonDirichlet());
	vVec_ = Eigen::VectorXf::Zero(mb_->nNonDirichlet());

	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	dVec_ = Temp_(mb_->validNodes());

	if (alpha_ < 1) {
		// Perform the conjugate gradiant to compute the initial vVec value
		// This is a bit odd because if the user hasn't specified the initial fluence rate it will be 0 initially. 
		// And can mess up the first few timesteps
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> initSolver;
		initSolver.compute(globM_);
		Eigen::VectorXf RHSinit = (globF_)-globK_ * dVec_;
		vVec_ = initSolver.solve(RHSinit);
	} // if we are using backwards Euler we can skip this initial computation of vVec. It is only
	// needed for explicit steps. 

	startTime = printDuration("Time Stepping Initialized in ", startTime);

	// Prepare solver for future iterations
	LHS_ = globM_ + alpha_ * deltaT_ * globK_;
	LHS_.makeCompressed();
	startTime = printDuration("LHS created: ", startTime);

	// These two steps form the cgSolver_.compute() function. By calling them separately, we 
	// only ever need to call factorize when the tissue properties change.
	cgSolver_.analyzePattern(LHS_);
	cgSolver_.factorize(LHS_);
	if (cgSolver_.info() != Eigen::Success) {
		std::cout << "Decomposition Failed" << std::endl;
	}
	startTime = printDuration("Initial Matrix Factorization Completed: ", startTime);
}
