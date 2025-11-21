#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <stdexcept>
#include "Mesh.hpp"
#include "MatrixBuilder.hpp"
#include "TimeIntegrators/TimeIntegrator.hpp"
#include "TimeIntegrators/CPUTimeIntegrator.hpp"
#include "ThermalModel.hpp"

/* These are needed here so that compiler is aware and MATLAB mex doesn't crash */
class TimeIntegrator; // specifically for compiling and circular references
//class CPUTimeIntegrator; // seems to prevent mex from crashing

class FEM_Simulator
{
public:
	//EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	bool silentMode = false; // controls print statements

	// -- Main Function calls for use -- 
	FEM_Simulator();
	FEM_Simulator(float MUA, float VHC, float TC, float HTC);
	FEM_Simulator(const FEM_Simulator& inputSim);
	~FEM_Simulator();
	void multiStep(float duration); // simulates multiple steps of time integration
	void singleStep();// simulates a single step of time integration
	void buildMatrices(); // creates global matrices and performs spatial discretization
	void initializeTimeIntegration(float alpha, float dt); // initializes solver for time integration
	void initializeTimeIntegration(); // initializes solver for time integration
	void initializeModel();
	void updateTemperatureSensors(); // update sensor temp vec and sensors variable
	void initializeContainers();

	void setEigenThreads(int nThreads) { Eigen::setNbThreads(nThreads); }

	// -- Setters and Getters -- 
	void setTemp(std::vector<std::vector<std::vector<float>>> Temp);
	void setTemp(Eigen::VectorXf& Temp);
	Eigen::VectorXf Temp() const { return thermalModel_->Temp; }

	void setFluenceRate(std::vector<std::vector<std::vector<float>>> fluenceRate);
	void setFluenceRate(Eigen::VectorXf& fluenceRate);
	void setFluenceRate(std::array<float,6> laserPose, float laserPower, float beamWaist);
	void setFluenceRate(Eigen::Vector<float,6> laserPose, float laserPower, float beamWaist);
	Eigen::VectorXf fluenceRate() const  { return thermalModel_->fluenceRate; }
	Eigen::VectorXf fluenceRateElem() const { return thermalModel_->fluenceRateElem; }

	void setDt(float deltaT);
	float dt() const { return dt_; }

	void setHeatFlux(float heatFlux);
	//float heatFlux() const { return heatFlux_; }
	
	void setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations);
	std::vector<std::array<float, 3>> sensorLocations() const;

	std::vector<float> sensorTemps() const { return sensorTemps_; }

	void setMesh(const Mesh& mesh);
	//std::shared_ptr<const Mesh> mesh() const { return mesh_; }
	
	void setTC(float TC);
	//float TC() const { return TC_; }
	void setVHC(float VHC);
	//float VHC() const { return VHC_; }
	void setMUA(float MUA);
	//float MUA() const { return MUA_; }
	void setHTC(float HTC);
	//float HTC() const { return HTC_; }
	void setAmbientTemp(float ambientTemp);
	//float ambientTemp() const { return ambientTemp_; }

	void setAlpha(float alpha) {
		if ((alpha > 1) || (alpha < 0))
			throw std::runtime_error("Alpha needs to be between 0 and 1 (inclusive)");
		alpha_ = alpha;
	}
	float alpha() const { return alpha_; }


	/**********************************************************************************************************************/
	/***************	 These were all private but I made them public so I could unit test them **************************/
private:
	struct Sensor
	{
		std::array<float, 3> pos;
		long elemIdx;
		std::array<float, 3> xi;
		float temp = 0;
	};

	template <typename ShapeFunc>
	void calculateSensorTemp(Sensor& sensor);

	const Mesh* mesh_ = nullptr;
	std::shared_ptr<GlobalMatrices> globalMatrices_ = nullptr;
	std::unique_ptr<ThermalModel> thermalModel_ = nullptr;
	TimeIntegrator* solver_ = nullptr;
	float alpha_ = 0.5; // time step weight
	float dt_ = 0.01; // time step [s]
	
	bool parameterUpdate_ = true;
	bool fluenceUpdate_ = true;
	std::vector<Sensor> sensors_; // locations of temperature sensors
	std::vector<float> sensorTemps_; // stored temperature information for each sensor

	std::chrono::steady_clock::time_point printDuration(const std::string& message, std::chrono::steady_clock::time_point startTime);
};

template<typename ShapeFunc>
inline void FEM_Simulator::calculateSensorTemp(Sensor& sensor)
{
	float sTemp = 0;
	Element elem = mesh_->elements()[sensor.elemIdx];
	for (int A = 0; A < ShapeFunc::nNodes; A++)
	{
		long nodeIdx = elem.nodes[A];
		float nodeTemp = thermalModel_->Temp(nodeIdx);
		sTemp += ShapeFunc::N(sensor.xi, A) * nodeTemp;
	}
	sensor.temp = sTemp;
}
