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


struct ThermalModel
{
	float TC = 0; // Thermal Conductivity [W/cm C]
	float VHC = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA = 0; // Absorption Coefficient [cm^-1]
	float HTC = 0; // convective heat transfer coefficient [W/cm^2]
	float ambientTemp = 0;  // Temperature surrounding the tissue for Convection [C]
	Eigen::VectorXf Temp; // Our values for temperature at the nodes of the elements
	Eigen::VectorXf fluenceRate; // Our values for Heat addition
	Eigen::VectorXf fluenceRateElem; // Our values for Heat addition
	float heatFlux = 0; // heat escaping the Neumann Boundary
};

class FEM_Simulator
{
public:
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
	void initializeSensorTemps(int numSteps); // initialize sensor temps vec with 0s
	void updateTemperatureSensors(int timeIdx); // update sensor temp vec
	void initializeContainers();

	// -- Setters and Getters -- 
	void setTemp(std::vector<std::vector<std::vector<float>>> Temp);
	void setTemp(Eigen::VectorXf& Temp);
	Eigen::VectorXf Temp() const { return thermalModel_.Temp; }

	void setFluenceRate(std::vector<std::vector<std::vector<float>>> fluenceRate);
	void setFluenceRate(Eigen::VectorXf& fluenceRate);
	void setFluenceRate(std::array<float,6> laserPose, float laserPower, float beamWaist);
	void setFluenceRate(Eigen::Vector<float,6> laserPose, float laserPower, float beamWaist);
	Eigen::VectorXf fluenceRate() const  { return thermalModel_.fluenceRate; }
	Eigen::VectorXf fluenceRateElem() const { return thermalModel_.fluenceRateElem; }

	void setDeltaT(float deltaT);
	float deltaT() const { return dt_; }

	void setHeatFlux(float heatFlux);
	//float heatFlux() const { return heatFlux_; }
	
	void setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations);
	std::vector<std::array<float, 3>> sensorLocations() const { return sensorLocations_; }

	std::vector<std::vector<float>> sensorTemps() const { return sensorTemps_; }
	Eigen::VectorXf getLatestSensorTemp() const;

	void setMesh(std::shared_ptr<const Mesh> mesh);
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
	Mesh* mesh_;
	GlobalMatrices globalMatrices_;
	ThermalModel thermalModel_;
	TimeIntegrator* solver;
	float alpha_ = 0.5; // time step weight
	float dt_ = 0.01; // time step [s]
	
	bool parameterUpdate_ = true;
	bool fluenceUpdate_ = true;
	std::vector< std::array<float, 3 >> sensorLocations_; // locations of temperature sensors
	std::vector<std::vector<float>> sensorTemps_; // stored temperature information for each sensor over time. 

	std::chrono::steady_clock::time_point printDuration(const std::string& message, std::chrono::steady_clock::time_point startTime);
};


