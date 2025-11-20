#include "FEM_Simulator.h"
#include "FEM_Simulator.h"
#include "FEM_Simulator.h"
#include <iostream>

FEM_Simulator::FEM_Simulator() {
}

FEM_Simulator::FEM_Simulator(float MUA, float VHC, float TC, float HTC)
{
	setMUA(MUA);
	setVHC(VHC);
	setTC(TC);
	setHTC(HTC);
}

FEM_Simulator::FEM_Simulator(const FEM_Simulator& other)
{
	// copy simple types
	silentMode = other.silentMode;
	thermalModel_ = other.thermalModel_;
	globalMatrices_ = other.globalMatrices_;
	parameterUpdate_ = other.parameterUpdate_;
	fluenceUpdate_ = other.fluenceUpdate_;

	sensorLocations_ = other.sensorLocations_;
	sensorTemps_ = other.sensorTemps_;
	alpha_ = other.alpha_;
	dt_ = other.dt_;

	// copy mesh (assuming Mesh has a safe copy constructor)
	mesh_ = other.mesh_;
}

FEM_Simulator::~FEM_Simulator(){
	
	if (!solver)
	{
		delete solver;
		solver = nullptr;
	}
}

void FEM_Simulator::multiStep(float duration) {
	/* This function simulates multiple steps of the heat equation. A single step duration is given by deltaT. If the total
	duration is not easily divisible by deltaT, we will round (up or down) and potentially perform an extra step or one step 
	fewer. This asumes that initializeModel() has already been run to create the the global matrices. 
	It repeatedly calls to singleStepCPU(). This function will also update the temperature sensors vector.  */ 
	auto startTime = std::chrono::steady_clock::now();
	int numSteps = round(duration / this->dt_);
	this->initializeSensorTemps(numSteps);
	this->updateTemperatureSensors(0);
	if (!this->silentMode) {
		std::cout << "Number of Steps: " << numSteps << std::endl;
		std::cout << "MultiStep(): Number of threads " << Eigen::nbThreads() << std::endl;
	}

	for (int t = 1; t <= numSteps; t++) {
		this->singleStep();
		this->updateTemperatureSensors(t);
	}

	startTime = this->printDuration("Time Stepping Completed in ", startTime);
}

void FEM_Simulator::singleStep() {
	/* 
	Simulates a single step of the heat equation using CPU. A single step is given by the duration deltaT. To run single step,
	it is assumed that initializeModel() has already been run to create the global matrices and perform initial factorization
	of the matrix inversion. This function can handle changes in fluence rate or changes in tissue properties. 
	*/
	if (fluenceUpdate_ || parameterUpdate_) {
		// only happens if fluenceUpdate is true
		//happens regardless of fluenceUpdate or parameterUpdate but has to happen after Firr update
		solver->applyParameters();
		fluenceUpdate_ = false;

		if (parameterUpdate_) { // Happens only if parameters were updated
			solver->updateLHS();
			parameterUpdate_ = false;
		}
	}
	Eigen::VectorXf dVec = solver->singleStepWithUpdate();

	// Adjust our Temp with new d vector
	thermalModel_.Temp(globalMatrices_.validNodes) = dVec;
}

void FEM_Simulator::buildMatrices()
{
	MatrixBuilder mb = MatrixBuilder();
	globalMatrices_ = mb.buildMatrices(*mesh_);
}

void FEM_Simulator::initializeTimeIntegration(float alpha, float dt)
{
	setDeltaT(dt);
	setAlpha(alpha);
	initializeTimeIntegration();
}

void FEM_Simulator::initializeTimeIntegration()
{
	auto startTime = std::chrono::steady_clock::now();
	parameterUpdate_ = false;
	TimeIntegrator* solver = new CPUTimeIntegrator(thermalModel_, globalMatrices_, alpha_, dt_);
	startTime = printDuration("Initial Matrix Factorization Completed: ", startTime);
}


void FEM_Simulator::initializeModel()
{ 
	/* initializeModel gets the system ready to perform time stepping. This function needs to be called whenever
	the geometry of the tissue changes. This includes things like changing the number of nodes, changing the boundary
	conditions, changing the layers in the mesh, etc. This function also needs to be called if alpha or the time step 
	changes. 
	
	This function does not need to be called if we are only changing the fluenceRate, or the value of tissue properties
	*/
	buildMatrices();
	fluenceUpdate_ = false;
	initializeTimeIntegration();
	// We are now ready to call single step
}

void FEM_Simulator::initializeSensorTemps(int numSteps) {
	// Create sensorTemperature Vectors and reserve sizes
	int nSensors = sensorLocations_.size();
	if (nSensors == 0) { // if there weren't any sensors added, put one at 0,0,0
		nSensors = 1;
		sensorLocations_.push_back({ 0,0,0 });
	}
	sensorTemps_.resize(nSensors);
	for (int s = 0; s < nSensors; s++) {
		sensorTemps_[s].resize(numSteps + 1);
	}
}

void FEM_Simulator::updateTemperatureSensors(int timeIdx) {
	/*TODO THIS DOES NOT WORK IF WE AREN'T USING LINEAR BASIS FUNCTIONS */
	//int nSensors = this->sensorLocations_.size();
	//int Nne = pow(this->Nn1d, 3);
	//// Input time = 0 information into temperature sensors
	////spacingLayer contains the distance between nodes in eacy layer.
	//for (int s = 0; s < nSensors; s++) {
	//	std::array<float,3> sensorLocation = this->sensorLocations_[s];
	//	// Determine the element (which for 1D linear elements is equivalent to the starting global node)
	//	// as well as the location in the bi-unit domain of that element, xi
	//	float xi[3];
	//	std::array<int, 3> elementLocation = positionToElement(sensorLocation,xi);
	//	std::array<int, 3> globalNodeStart;
	//	// For linear elements, there will be no change between globalNodeStart and elementLocation
	//	for (int i = 0; i < 3; i++) {
	//		globalNodeStart[i] = elementLocation[i] * (this->Nn1d - 1);
	//	}
	//	float tempValue = 0;	
	//	for (int Ai = 0; Ai < Nne; Ai++) { // iterate through each node in the element
	//		// adjust the global starting node based on the current node we should be visiting
	//		int globalNodeSub[3] = { globalNodeStart[0] + (Ai&1),globalNodeStart[1]+((Ai & 2) >> 1), globalNodeStart[2]+ ((Ai & 4) >> 2) };
	//		// convert subscript to index
	//		int globalNode = globalNodeSub[0] + globalNodeSub[1] * this->nodesPerAxis[0] + globalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];
	//		// add temperature contribution of node
	//		tempValue += this->calculateNA(xi, Ai) * this->Temp_(globalNode);			
	//	}
	//	this->sensorTemps_[s][timeIdx] = tempValue;
	//}

}

void FEM_Simulator::initializeContainers()
{
	thermalModel_.Temp = Eigen::VectorXf::Zero(mesh_->nodes().size()); // Our values for temperature at the nodes of the elements
	thermalModel_.fluenceRate = Eigen::VectorXf::Zero(mesh_->nodes().size()); // Our values for Heat addition
	thermalModel_.fluenceRateElem = Eigen::VectorXf::Zero(mesh_->elements().size()); // Our values for Heat addition
}

void FEM_Simulator::setTemp(std::vector<std::vector<std::vector<float>>> Temp) {

	// Convert nested vectors into a single column Eigen Vector. 
	Eigen::VectorXf TempVec = Eigen::VectorXf::Zero(Temp.size() * Temp[0].size() * Temp[0][0].size());
	for (int i = 0; i < Temp.size(); i++) // associated with x and is columns of matlab matrix
	{
		for (int j = 0; j < Temp[0].size(); j++) // associated with y and is rows of matlab matrix
		{
			for (int k = 0; k < Temp[0][0].size(); k++) // associated with z and is depth of matlab matrix
			{
				TempVec(i + j*Temp.size() + k*Temp.size()*Temp[0].size()) = Temp[i][j][k];
			}
		}
	}
	setTemp(TempVec);
}

void FEM_Simulator::setTemp(Eigen::VectorXf &Temp)
{	
	//TODO make sure Temp is the correct size
	if (mesh_->nodes().size() != Temp.size()) {
		throw std::runtime_error("Total number of elements in Temp does not match number of nodes in mesh.");
	}
	thermalModel_.Temp = Temp;
}

void FEM_Simulator::setFluenceRate(Eigen::VectorXf& fluenceRate)
{	/* 
	Main function to set the fluence rate of the laser. All other setFluenceRate() functions will call this one in the end. 
	*/
	// -- Checking size of vector to make sure it is appropriate
	if (fluenceRate.size() == mesh_->nodes().size()){
		thermalModel_.fluenceRate = fluenceRate;
		thermalModel_.fluenceRateElem.setZero();
	} else if (fluenceRate.size() == mesh_->elements().size()){
		thermalModel_.fluenceRateElem = fluenceRate;
		thermalModel_.fluenceRate.setZero();
	} else {
		std::cout << "NFR must have the same number of entries as the node space or element space" << std::endl;
		throw std::invalid_argument("NFR must have the same number of entries as the node space or element space");
	}
	// -- setting flag that fluence has been updated
	fluenceUpdate_ = true;
}

void FEM_Simulator::setFluenceRate(std::vector<std::vector<std::vector<float>>> inputFluence)
{
	int xNumNodes = inputFluence.size();
	int yNumNodes = inputFluence[0].size();
	int zNumNodes = inputFluence[0][0].size();
	Eigen::VectorXf FR = Eigen::VectorXf::Zero(xNumNodes * yNumNodes * zNumNodes);
	// Convert nested vectors into a single column Eigen Vector. 
	for (int i = 0; i < xNumNodes; i++) // associated with x and is columns of matlab matrix
	{
		for (int j = 0; j < yNumNodes; j++) // associated with y and is rows of matlab matrix
		{
			for (int k = 0; k < zNumNodes; k++) // associated with z and is depth of matlab matrix
			{
				FR(i + j * xNumNodes + k * xNumNodes * yNumNodes) = inputFluence[i][j][k];
			}
		}
	}
	this->setFluenceRate(FR);
}

void FEM_Simulator::setFluenceRate(std::array<float,6> laserPose, float laserPower, float beamWaist)
{
	Eigen::VectorXf FR = Eigen::VectorXf::Zero(mesh_->nodes().size());
	
	// Precompute constants
	const float pi = 3.14159265358979323846f;
	const float lambda = 10.6e-4f;  // wavelength in cm
	// ASSUMING THERE IS NO ORIENTATION SHIFT ON THE LASER
	//TODO: account for orientation shift on the laser
	// I(x,y,z) = 2*P/(pi*w^2) * exp(-2*(x^2 + y^2)/w^2 - mua*z)

	const float mua = thermalModel_.MUA;
	
	for (int i = 0; i < mesh_->nodes().size(); i++)
	{
		Node node = mesh_->nodes()[i];
		float xf = node.x - laserPose[0];
		float yf = node.y - laserPose[1]; // distance between node and laser focal point in y
		float zf = node.z - laserPose[2]; // distance between node and laser focal point in x
		float waist2 = beamWaist * beamWaist;
		float width = beamWaist * std::sqrt(1.0f + ((lambda * zf) / (pi * waist2)) * ((lambda * zf) / (pi * waist2)));
		float width2 = width * width;

		// ****ASSUMES A FLAT TISSUE SURFACE THAT STARTS AT Z=0**********
		// We are assuming that the laser penetration distance (mua * zpen) is equivalent to z position of the node
		// which means that we are assuming the tissue surface is flat. 
		float exponent = -2.0f * (xf * xf + yf * yf) / width2 - mua * node.z;
		float irr = 2.0f * laserPower / (pi * width2) * std::exp(exponent);
		FR(i) = irr;
	}
	setFluenceRate(FR);
}

void FEM_Simulator::setFluenceRate(Eigen::Vector<float, 6> laserPose, float laserPower, float beamWaist)
{
	std::array<float,6> laserPoseVec;
	for (int i = 0; i < 6; i++) {
		laserPoseVec[i] = laserPose(i);
	}
	setFluenceRate(laserPoseVec, laserPower, beamWaist);
}

void FEM_Simulator::setDeltaT(float dt)
{
	dt_ = dt;
	parameterUpdate_ = true;
}


void FEM_Simulator::setTC(float TC) {
	thermalModel_.TC = TC;
	parameterUpdate_ = true;
}

void FEM_Simulator::setVHC(float VHC) {
	thermalModel_.VHC = VHC;
	parameterUpdate_ = true;
}

void FEM_Simulator::setMUA(float MUA) {
	thermalModel_.MUA = MUA;
	parameterUpdate_ = true;
}

void FEM_Simulator::setHTC(float HTC) {
	thermalModel_.HTC = HTC;
	parameterUpdate_ = true;
}

void FEM_Simulator::setHeatFlux(float heatFlux)
{
	thermalModel_.heatFlux = heatFlux;
}

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	thermalModel_.ambientTemp = ambientTemp;
}


void FEM_Simulator::setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations)
{
	sensorLocations_ = tempSensorLocations;
	bool errorFlag = false;
	/* TODO - need to find a way to check if sensors are in tissue with arbitrary mesh setup */
	//for (int s = 0; s < sensorLocations_.size(); s++) {
	//	if ((sensorLocations_[s][0] < -this->tissueSize[0] / 2.0f) || (sensorLocations_[s][0] > this->tissueSize[0] / 2.0f) ||
	//		(sensorLocations_[s][1] < -this->tissueSize[1] / 2.0f) || (sensorLocations_[s][1] > this->tissueSize[1] / 2.0f) ||
	//		(sensorLocations_[s][2] < 0) || (sensorLocations_[s][2] > this->tissueSize[2])) {
	//		errorFlag = true;
	//		break;
	//	}	
	//}
	if (errorFlag) {
		throw std::invalid_argument("All sensor locations must be within the tissue block");
	}
}

Eigen::VectorXf FEM_Simulator::getLatestSensorTemp() const
{
	Eigen::VectorXf sensorVector(sensorTemps_.size());
	for (int s = 0; s < sensorTemps_.size(); s++) {
		sensorVector(s) = this->sensorTemps_[s][sensorTemps_[s].size()-1];
	}
	return sensorVector;
}

void FEM_Simulator::setMesh(std::shared_ptr<const Mesh> mesh)
{
	mesh_ = mesh;
	initializeContainers();
}

std::chrono::steady_clock::time_point FEM_Simulator::printDuration(const std::string& message, std::chrono::steady_clock::time_point startTime) {
	if (!this->silentMode) {
		auto stopTime = std::chrono::steady_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << message << duration.count() / 1000000.0 << " s" << std::endl;
		startTime = stopTime;
		
	}
	return startTime;
}
