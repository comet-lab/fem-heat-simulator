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

class FEM_Simulator
{
public:
	bool silentMode = false; // controls print statements

	// One node can belong to multiple faces so we use binary to set the face value
	// 0 - internal node, 1 - top face, 2 - bottom face, 4 - front face, 8 - right fce, 16 - back face, 32 - left face
	enum tissueFace { INTERNAL = 0, TOP = 1, BOTTOM = 2, FRONT = 4, RIGHT = 8, BACK = 16, LEFT = 32 };
	// This maps the face to the axis number and direction of the face: top, bottom, front, right, back, left
	// The values listed here are heavily tied to our reference frame decisions and the function @determineNodeFace
	const int dimMap[6] = { -3, 3, 1, 2, -1, -2 };
	// -3: We are on the top face    - surface normal is in -z direction
	//  3: We are on the bottom face - surface normal is in +z direction
	//  1: We are on the front face  - surface normal is +x direction
	//  2: We are on the right face  - surface normal is +y direction
	// -1: We are on the back face   - surface normal is -x direction
	// -2: We are on the left face   - surface normal is -y direction

	// -- Main Function calls for use -- 
	FEM_Simulator();
	FEM_Simulator(float MUA, float VHC, float TC, float HTC);
	FEM_Simulator(const FEM_Simulator& inputSim);
	~FEM_Simulator();
	void multiStep(float duration); // simulates multiple steps of time integration
	void singleStep();
	void singleStepCPU(); // simulates a single step of time integration
	void buildMatrices(); // creates global matrices and performs spatial discretization
	void applyParametersCPU();
	void initializeTimeIntegrationCPU();
	void initializeModel();
	void initializeSensorTemps(int numSteps); // initialize sensor temps vec with 0s
	void updateTemperatureSensors(int timeIdx); // update sensor temp vec
	void initializeContainers();
	void setGlobalSparsityPattern();

	// -- Setters and Getters -- 
	void setTemp(std::vector<std::vector<std::vector<float>>> Temp);
	void setTemp(Eigen::VectorXf& Temp);
	std::vector<std::vector<std::vector<float>>> TempAsVec() const;
	Eigen::VectorXf Temp() const { return Temp_; }

	void setFluenceRate(std::vector<std::vector<std::vector<float>>> fluenceRate);
	void setFluenceRate(Eigen::VectorXf& fluenceRate);
	void setFluenceRate(std::array<float,6> laserPose, float laserPower, float beamWaist);
	void setFluenceRate(Eigen::Vector<float,6> laserPose, float laserPower, float beamWaist);
	Eigen::VectorXf fluenceRate() const  { return fluenceRate_; }
	Eigen::VectorXf fluenceRateElem() const { return fluenceRateElem_; }

	void setDeltaT(float deltaT);
	float deltaT() const { return deltaT_; }

	void setHeatFlux(float heatFlux);
	float heatFlux() const { return heatFlux_; }
	
	void setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations);
	std::vector<std::array<float, 3>> sensorLocations() const { return sensorLocations_; }

	std::vector<std::vector<float>> sensorTemps() const { return sensorTemps_; }
	Eigen::VectorXf getLatestSensorTemp() const;

	void setMesh(Mesh mesh);
	
	void setTC(float TC);
	float TC() const { return TC_; }
	void setVHC(float VHC);
	float VHC() const { return VHC_; }
	void setMUA(float MUA);
	float MUA() const { return MUA_; }
	void setHTC(float HTC);
	float HTC() const { return HTC_; }
	void setAmbientTemp(float ambientTemp);
	float ambientTemp() const { return ambientTemp_; }

	void setAlpha(float alpha) {
		if ((alpha > 1) || (alpha < 0))
			throw std::runtime_error("Alpha needs to be between 0 and 1 (inclusive)");
		alpha_ = alpha;
	}
	float alpha() const { return alpha_; }


	/**********************************************************************************************************************/
	/***************	 These were all private but I made them public so I could unit test them **************************/
private:
	Mesh mesh_;
	MatrixBuilder* mb_ = nullptr; 
	float TC_ = 0; // Thermal Conductivity [W/cm C]
	float VHC_ = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA_ = 0; // Absorption Coefficient [cm^-1]
	float HTC_ = 0; // convective heat transfer coefficient [W/cm^2]
	float ambientTemp_ = 0;  // Temperature surrounding the tissue for Convection [C]
	Eigen::VectorXf Temp_; // Our values for temperature at the nodes of the elements
	Eigen::VectorXf dVec_; // This is our discrete temperature at non-dirichlet nodes from Time-stepping
	Eigen::VectorXf vVec_; // This is our discrete temperature velocity from Time-stepping
	Eigen::VectorXf fluenceRate_; // Our values for Heat addition
	Eigen::VectorXf fluenceRateElem_; // Our values for Heat addition
	float alpha_ = 0.5; // time step weight
	float deltaT_ = 0.01; // time step [s]
	float heatFlux_ = 0; // heat escaping the Neumann Boundary
	
	bool parameterUpdate_ = true;
	bool fluenceUpdate_ = true;
	bool elemNFR_ = false; // whether the FluenceRate pertains to an element or a node
	std::vector< std::array<float, 3 >> sensorLocations_; // locations of temperature sensors
	std::vector<std::vector<float>> sensorTemps_; // stored temperature information for each sensor over time. 

	/* This section of variables stores all the matrices used for building the first order ODE
	There are both global and elemental matrices specified. The global matrices are also distinguished
	by how they are created (e.g. conduction, laser, etc). These are saved as class attributes because
	the current build assumes them to be relatively constant throughout the mesh so its easier to save once*/
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cgSolver_;

	Eigen::SparseMatrix<float> LHS_; // this stores the left hand side of our matrix inversion, so the solver doesn't lose the reference.

	Eigen::SparseMatrix<float, Eigen::RowMajor> globK_; // Global Conductivity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> globM_; // Global Thermal Mass Matrix
	Eigen::VectorXf globF_; // Global Heat Source Matrix 

	std::chrono::steady_clock::time_point printDuration(const std::string& message, std::chrono::steady_clock::time_point startTime);
};


