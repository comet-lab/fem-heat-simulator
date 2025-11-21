#pragma once
#include <Eigen/Dense>
#include "ThermalModel.hpp"
#include "GlobalMatrices.hpp"
#include <iostream>

class TimeIntegrator {
public:
	//EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	TimeIntegrator(const ThermalModel& thermalModel, const GlobalMatrices& globalMatrices, float alpha, float dt)
		: thermalModel_(thermalModel), globalMatrices_(globalMatrices), alpha_(alpha), dt_(dt) 
	{
		setAlpha(alpha);
		setDt(dt);
		/*dVec_.setZero(globalMatrices_.nNonDirichlet);
		vVec_.setZero(globalMatrices_.nNonDirichlet);*/
	}
	virtual ~TimeIntegrator() = default;
	virtual void applyParameters() = 0;
	virtual void initialize() = 0;
	virtual void singleStep() = 0;
	virtual void updateLHS() = 0;
	

	virtual void singleStep(float alpha, float dt)
	{
		setAlpha(alpha);
		setDt(dt);
		singleStep();
	}
	Eigen::VectorXf singleStepWithUpdate() 
	{
		singleStep();
		return dVec_;
	}
	Eigen::VectorXf singleStepWithUpdate(float alpha, float dt)
	{
		singleStep(alpha, dt);
		return dVec_;
	}
	float alpha() const { return alpha_; }
	float dt() const { return dt_; }
	Eigen::VectorXf dVec() const { return dVec_; }
	Eigen::VectorXf vVec() const { return vVec_; }

	void setAlpha(float alpha) {
		if ((alpha > 1) || (alpha < 0))
			throw std::runtime_error("Alpha needs to be between 0 and 1 (inclusive)");
		alpha_ = alpha;
	}

	void setDt(float dt) {
		if (dt <= 0)
			throw std::runtime_error("Time step (dt) must be greater than 0");
		dt_ = dt;
	}


protected: 
	const ThermalModel& thermalModel_;
	const GlobalMatrices& globalMatrices_;
	Eigen::VectorXf dVec_;
	Eigen::VectorXf vVec_;
	float alpha_;
	float dt_;
};