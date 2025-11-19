#include "FEM_Simulator.h"
#include "FEM_Simulator.h"
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
	TC_ = other.TC_;
	VHC_ = other.VHC_;
	MUA_ = other.MUA_;
	HTC_ = other.HTC_;
	ambientTemp_ = other.ambientTemp_;
	alpha_ = other.alpha_;
	deltaT_ = other.deltaT_;
	heatFlux_ = other.heatFlux_;
	parameterUpdate_ = other.parameterUpdate_;
	fluenceUpdate_ = other.fluenceUpdate_;

	// copy STL containers
	sensorLocations_ = other.sensorLocations_;
	sensorTemps_ = other.sensorTemps_;

	// copy Eigen vectors/matrices (Eigen copies deeply by default)
	Temp_ = other.Temp_;
	dVec_ = other.dVec_;
	vVec_ = other.vVec_;
	fluenceRate_ = other.fluenceRate_;
	fluenceRateElem_ = other.fluenceRateElem_;

	// copy sparse matrices and solver members
	globK_ = other.globK_;
	globM_ = other.globM_;
	globF_ = other.globF_;
	LHS_ = other.LHS_;
//	cgSolver_ = other.cgSolver_; // this is safe; solver will copy settings but not references

	// copy mesh (assuming Mesh has a safe copy constructor)
	mesh_ = other.mesh_;

	// IMPORTANT: shallow copy the MatrixBuilder pointer
	mb_ = other.mb_;

}

FEM_Simulator::~FEM_Simulator(){
	// std::cout << "FEM_Simulator Destructor" << std::endl;
}

void FEM_Simulator::multiStep(float duration) {
	/* This function simulates multiple steps of the heat equation. A single step duration is given by deltaT. If the total
	duration is not easily divisible by deltaT, we will round (up or down) and potentially perform an extra step or one step 
	fewer. This asumes that initializeModel() has already been run to create the the global matrices. 
	It repeatedly calls to singleStepCPU(). This function will also update the temperature sensors vector.  */ 
	auto startTime = std::chrono::steady_clock::now();
	int numSteps = round(duration / this->deltaT_);
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
	Performs either the GPU single step or the CPU single step
	*/
	this->singleStepCPU();
}

void FEM_Simulator::singleStepCPU() {
	/* Simulates a single step of the heat equation using CPU. A single step is given by the duration deltaT. To run single step,
	it is assumed that initializeModel() has already been run to create the global matrices and perform initial factorization
	of the matrix inversion. This function can handle changes in fluence rate or changes in tissue properties. */

	if (fluenceUpdate_ || parameterUpdate_) {
		// only happens if fluenceUpdate is true
		//happens regardless of fluenceUpdate or parameterUpdate but has to happen after Firr update
		applyParametersCPU();
		fluenceUpdate_ = false;

		if (parameterUpdate_) { // Happens only if parameters were updated
			LHS_ = globM_ + alpha_ * deltaT_ * globK_; // Create new left hand side 
			LHS_.makeCompressed(); // compress it for potential speed improvements
			cgSolver_.factorize(LHS_); // Perform factoriziation based on analysis which should have been called with initializeModel();
			if (cgSolver_.info() != Eigen::Success) {
				std::cout << "Decomposition Failed" << std::endl;
			}
			parameterUpdate_ = false;
		}
	}

	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	dVec_ = Temp_(mb_->validNodes());

	//this->cgSolver_.factorize(this->LHS_); // Perform factoriziation based on analysis which should have been called with initializeModel();
	// Explicit Forward Step (only if alpha < 1)
	if (alpha_ < 1) {
		dVec_ = dVec_ + (1 - alpha_) * deltaT_ * vVec_; // normally the output of this equation is assigned to dTilde for clarity...
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
	dVec_ = dVec_ + alpha_ * deltaT_ * vVec_; // ... dTilde would also be on the righ-hand side here. 

	// Adjust our Temp with new d vector
	Temp_(mb_->validNodes()) = dVec_;
}

void FEM_Simulator::buildMatrices()
{
	mb_ = std::make_shared<MatrixBuilder>(*mesh_);
	mb_->buildMatrices();

	setGlobalSparsityPattern();
}

void FEM_Simulator::applyParametersCPU()
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
	globK_.setZero();
	globK_ += mb_->K() * TC_;
	globK_ += mb_->Q() * HTC_;
	// Thermal mass matrix
	globM_.setZero();
	globM_ += mb_->M() * VHC_; // M Doesn't have any additions so we just multiply it by the constant
	// Forcing Vector
	globF_.setZero();
	globF_ += (mb_->Fq() * ambientTemp_ + mb_->Fconv() * Temp_ ) * HTC_; // convection from ambient temp and dirichlet nodes
	globF_ += (mb_->Fflux() * heatFlux_); // heat flux 
	globF_ += (mb_->Fk() * Temp_ * TC_); // conduction on dirichlet nodes
	globF_ += MUA_ * (mb_->Fint() * fluenceRate_ + mb_->FintElem() * fluenceRateElem_); // forcing function
	// startTime = this->printDuration("Parameter Multiplication Performed: ", startTime);
}

void FEM_Simulator::initializeTimeIntegrationCPU()
{
	/* 
	Creates the d and v vectors used for time integration. V vector is created by solving the 
	linear system M*v = (F-K*d). Then the Eigen conjugate gradient solver is initialized with the 
	left-hand-side of the system (M + alpha*dt*K)*v = (F - K*d). 
	*/
	auto startTime = std::chrono::steady_clock::now();
	applyParametersCPU();
	parameterUpdate_ = false;

	int nNodes = mesh_->nodes().size();
	/* PERFORMING TIME INTEGRATION USING EULER FAMILY */
	// Initialize d, v, and dTilde vectors
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
		Eigen::VectorXf RHSinit = (globF_) - globK_ * dVec_;
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

void FEM_Simulator::initializeModel()
{ 
	/* initializeModel gets the system ready to perform time stepping. This function needs to be called whenever
	the geometry of the tissue changes. This includes things like changing the number of nodes, changing the boundary
	conditions, changing the layers in the mesh, etc. This function also needs to be called if alpha or the time step 
	changes. 
	
	This function does not need to be called if we are only changing the fluenceRate, or the value of tissue properties
	*/
	this->buildMatrices();
	this->fluenceUpdate_ = false;

	this->initializeTimeIntegrationCPU();
	
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
	Temp_ = Eigen::VectorXf::Zero(mesh_->nodes().size()); // Our values for temperature at the nodes of the elements
	fluenceRate_ = Eigen::VectorXf::Zero(mesh_->nodes().size()); // Our values for Heat addition
	fluenceRateElem_ = Eigen::VectorXf::Zero(mesh_->elements().size()); // Our values for Heat addition
}

void FEM_Simulator::setGlobalSparsityPattern()
{
	/* Set the sparsity pattern for the global matrices, so that future updates are faster in Eigen*/
	globK_ = mb_->K() + mb_->Q();
	globK_.makeCompressed();
	globM_ = mb_->M();
	globM_.makeCompressed();
	globF_ = Eigen::VectorXf::Zero(mb_->nNonDirichlet());
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
	Temp_ = Temp;
}

std::vector<std::vector<std::vector<float>>> FEM_Simulator::TempAsVec() const
{
	/*std::vector<std::vector<std::vector<float>>>
		TempOut(this->nodesPerAxis[0], std::vector<std::vector<float>>(this->nodesPerAxis[1], std::vector<float>(this->nodesPerAxis[2])));

	for (int i = 0; i < this->nodesPerAxis[0]; i++) {
		for (int j = 0; j < this->nodesPerAxis[1]; j++) {
			for (int k = 0; k < this->nodesPerAxis[2]; k++) {
				TempOut[i][j][k] = Temp_(i + j * (this->nodesPerAxis[0]) + k * (this->nodesPerAxis[0] * this->nodesPerAxis[1]));
			}
		}
	}*/

	//return TempOut;
	std::vector<std::vector<std::vector<float>>> tempOut;
	return tempOut;
}

void FEM_Simulator::setFluenceRate(Eigen::VectorXf& fluenceRate)
{	/* 
	Main function to set the fluence rate of the laser. All other setFluenceRate() functions will call this one in the end. 
	*/
	// -- Checking size of vector to make sure it is appropriate
	if (fluenceRate.size() == mesh_->nodes().size()){
		fluenceRate_ = fluenceRate;
		fluenceRateElem_.setZero();
	} else if (this->fluenceRate_.size() == mesh_->elements().size()){
		fluenceRateElem_ = fluenceRate;
		fluenceRate_.setZero();
	} else {
		std::cout << "NFR must have the same number of entries as the node space or element space" << std::endl;
		throw std::invalid_argument("NFR must have the same number of entries as the node space or element space");
	}
	// -- setting flag that fluence has been updated
	this->fluenceUpdate_ = true;
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

	const float mua = this->MUA_;
	
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
	this->setFluenceRate(FR);
}

void FEM_Simulator::setFluenceRate(Eigen::Vector<float, 6> laserPose, float laserPower, float beamWaist)
{
	std::array<float,6> laserPoseVec;
	for (int i = 0; i < 6; i++) {
		laserPoseVec[i] = laserPose(i);
	}
	this->setFluenceRate(laserPoseVec, laserPower, beamWaist);
}

void FEM_Simulator::setDeltaT(float deltaT)
{
	this->deltaT_ = deltaT;
	this->parameterUpdate_ = true;
}


void FEM_Simulator::setTC(float TC) {
	this->TC_ = TC;
	this->parameterUpdate_ = true;
}

void FEM_Simulator::setVHC(float VHC) {
	this->VHC_ = VHC;
	this->parameterUpdate_ = true;
}

void FEM_Simulator::setMUA(float MUA) {
	this->MUA_ = MUA;
	this->parameterUpdate_ = true;
}

void FEM_Simulator::setHTC(float HTC) {
	this->HTC_ = HTC;
	this->parameterUpdate_ = true;
}

void FEM_Simulator::setHeatFlux(float heatFlux)
{
	heatFlux_ = heatFlux;
}

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	this->ambientTemp_ = ambientTemp;
}


void FEM_Simulator::setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations)
{
	this->sensorLocations_ = tempSensorLocations;
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
