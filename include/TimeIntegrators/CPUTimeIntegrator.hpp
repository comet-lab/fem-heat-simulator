#pragma once
#include "TimeIntegrators/TimeIntegrator.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>

class CPUTimeIntegrator : public TimeIntegrator {

public:
	//EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	CPUTimeIntegrator(const ThermalModel& thermalModel, const GlobalMatrices& globalMatrices, float alpha, float deltat) 
		: TimeIntegrator(thermalModel, globalMatrices, alpha, deltat) {}
	void applyParameters() override;
	void initialize() override;
	void singleStep() override;
	void updateLHS() override;

	void setMatrixSparsity(); // Sets the sparsity pattern so that applyParameters can run faster

private:
	Eigen::SparseMatrix<float> globK_; // Global Conductivity Matrix
	Eigen::SparseMatrix<float> globM_; // Global Thermal Mass Matrix
	Eigen::VectorXf globF_; // Global Heat Source Matrix 

	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cgSolver_;
	Eigen::SparseMatrix<float> LHS_; // this stores the left hand side of our matrix inversion, so the solver doesn't lose the reference.
};