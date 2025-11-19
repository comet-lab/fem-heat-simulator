#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <stdexcept>
#include "FEM_Simulator.h"


class TimeIntegrator {
public:
	virtual void applyParameters() = 0;
	virtual void initialize() = 0;
	virtual void initializeWithModel() = 0;
	virtual Eigen::VectorXf singleStep(const Eigen::VectorXf& Temp, float alpha, float deltat) = 0;
	virtual void singleStepWithUpdate() = 0;

private: 
	FEM_Simulator* model_;
	Eigen::VectorXf dVec;
	Eigen::VectorXf vVec;


};