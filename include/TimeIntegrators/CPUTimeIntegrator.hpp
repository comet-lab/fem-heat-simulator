#pragma once
#include "TimeIntegrators/TimeIntegrator.hpp"

class CPUTimeIntegrator : public TimeIntegrator {

public:
	virtual void applyParameters() override;
	virtual void initialize() override;
	virtual void initializeWithModel() override;
	virtual Eigen::VectorXf singleStep(const Eigen::VectorXf& Temp, float alpha, float deltat) override;
	virtual void singleStepWithUpdate() override;

private:
	FEM_Simulator* model_;
	Eigen::VectorXf dVec;
	Eigen::VectorXf vVec;

	Eigen::SparseMatrix<float, Eigen::RowMajor> globK_; // Global Conductivity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> globM_; // Global Thermal Mass Matrix
	Eigen::VectorXf globF_; // Global Heat Source Matrix 
};