#include "FEM_Simulator.h"


FEM_Simulator::FEM_Simulator() {
	thermalModel_ = std::make_unique<ThermalModel>();
}

FEM_Simulator::FEM_Simulator(float MUA, float VHC, float TC, float HTC)
{
	thermalModel_ = std::make_unique<ThermalModel>();
	setMUA(MUA);
	setVHC(VHC);
	setTC(TC);
	setHTC(HTC);

}

FEM_Simulator::FEM_Simulator(const FEM_Simulator& other)
{
	// copy simple types
	silentMode = other.silentMode;
	globalMatrices_ = other.globalMatrices_;
	parameterUpdate_ = other.parameterUpdate_;
	fluenceUpdate_ = other.fluenceUpdate_;

	sensors_ = other.sensors_;
	sensorTemps_ = other.sensorTemps_;
	alpha_ = other.alpha_;
	dt_ = other.dt_;

	if(other.thermalModel_)
		thermalModel_ = std::make_unique<ThermalModel>(*other.thermalModel_);
	else
		thermalModel_ = nullptr;

	// copy mesh (assuming Mesh has a safe copy constructor)
	mesh_ = other.mesh_;
}

FEM_Simulator::~FEM_Simulator(){
	
	if (!solver_)
	{
		delete solver_;
		solver_ = nullptr;
	}
}

void FEM_Simulator::multiStep(float duration) {
	/* This function simulates multiple steps of the heat equation. A single step duration is given by deltaT. If the total
	duration is not easily divisible by deltaT, we will round (up or down) and potentially perform an extra step or one step 
	fewer. This asumes that initializeModel() has already been run to create the the global matrices. 
	It repeatedly calls to singleStep(). This function will also update the temperature sensors vector.  */ 
	auto startTime = std::chrono::steady_clock::now();
	int numSteps = round(duration / dt_);
	if (!silentMode) {
		std::cout << "Number of Steps: " << numSteps << std::endl;
		std::cout << "MultiStep(): Number of threads " << Eigen::nbThreads() << std::endl;
	}

	updateSolver(); // Handle any changes in fluence rate or parameters before solving
	for (int t = 1; t <= numSteps; t++) {
		solver_->singleStep();
	}
	Eigen::VectorXf dVec = solver_->dVec(); // retrieve dVec
	thermalModel_->Temp(globalMatrices_->validNodes) = dVec;
	updateTemperatureSensors();
	startTime = printDuration("Time Stepping Completed in ", startTime);
}

void FEM_Simulator::singleStep() {
	/* 
	Simulates a single step of the heat equation using CPU. A single step is given by the duration deltaT. To run single step,
	it is assumed that initializeModel() has already been run to create the global matrices and perform initial factorization
	of the matrix inversion. This function can handle changes in fluence rate or changes in tissue properties. 
	*/
	updateSolver();
	Eigen::VectorXf dVec = solver_->singleStepWithUpdate();
	// Adjust our Temp with new d vector
	thermalModel_->Temp(globalMatrices_->validNodes) = dVec;
	updateTemperatureSensors();
}

void FEM_Simulator::updateSolver()
{
	if (!solver_)
		throw std::runtime_error("Solver not initialized. Call initializeModel() or initializeTimeIntegration before singleStep()");

	if (fluenceUpdate_ || parameterUpdate_) {
		// only happens if fluenceUpdate is true
		//happens regardless of fluenceUpdate or parameterUpdate but has to happen after Firr update
		solver_->calculateGlobF(); 
		fluenceUpdate_ = false;

		if (parameterUpdate_) { // Happens only if parameters were updated
			solver_->calculateGlobK();
			solver_->calculateGlobM();
			solver_->updateLHS();
			parameterUpdate_ = false;
		}
	}
}

void FEM_Simulator::buildMatrices()
{
	MatrixBuilder mb = MatrixBuilder();
	if (!mesh_)
		throw std::runtime_error("Mesh has not been set yet. Cannot build matrices");
	globalMatrices_ = std::make_shared<GlobalMatrices>(mb.buildMatrices(*mesh_));
	fluenceUpdate_ = false;
}

void FEM_Simulator::initializeTimeIntegration(float alpha, float dt)
{
	setDt(dt);
	setAlpha(alpha);
	initializeTimeIntegration();
}

void FEM_Simulator::initializeTimeIntegration()
{
	auto startTime = std::chrono::steady_clock::now();
	parameterUpdate_ = false;
	if (solver_) // if it already exists lets make sure to clear the memory
	{
		delete solver_;
		solver_ = nullptr;
	}
#ifdef USE_CUDA
	if (gpuEnabled_)
		solver_ = new GPUTimeIntegrator(*thermalModel_, *globalMatrices_, alpha_, dt_);
	else
#endif
	{
		solver_ = new CPUTimeIntegrator(*thermalModel_, *globalMatrices_, alpha_, dt_);
	}
	solver_->initialize();
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
	initializeTimeIntegration();
	// We are now ready to call single step
}

void FEM_Simulator::updateTemperatureSensors() {
	
	int nSensors = sensors_.size();
	// Input time = 0 information into temperature sensors
	for (int s = 0; s < nSensors; s++) {
		if ((mesh_->order() == LINEAR) && (mesh_->elementShape() == HEXAHEDRAL))
			calculateSensorTemp<ShapeFunctions::HexLinear>(sensors_[s]);
		else if ((mesh_->order() == LINEAR) && (mesh_->elementShape() == TETRAHEDRAL))
			calculateSensorTemp<ShapeFunctions::TetLinear>(sensors_[s]);
		else
			throw std::runtime_error("Can't calculate sensor temperature for element type");
		sensorTemps_[s] = (sensors_[s].temp);
	}

}

void FEM_Simulator::initializeContainers()
{
	thermalModel_->Temp = Eigen::VectorXf::Zero(mesh_->nodes().size()); // Our values for temperature at the nodes of the elements
	thermalModel_->fluenceRate = Eigen::VectorXf::Zero(mesh_->nodes().size()); // Our values for Heat addition
	thermalModel_->fluenceRateElem = Eigen::VectorXf::Zero(mesh_->elements().size()); // Our values for Heat addition
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
	thermalModel_->Temp = Temp;
}

void FEM_Simulator::setFluenceRate(Eigen::VectorXf& fluenceRate)
{	/* 
	Main function to set the fluence rate of the laser. All other setFluenceRate() functions will call this one in the end. 
	*/
	// -- Checking size of vector to make sure it is appropriate
	if (fluenceRate.size() == mesh_->nodes().size()){
		thermalModel_->fluenceRate = fluenceRate;
		thermalModel_->fluenceRateElem.setZero();
	} else if (fluenceRate.size() == mesh_->elements().size()){
		thermalModel_->fluenceRateElem = fluenceRate;
		thermalModel_->fluenceRate.setZero();
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

void FEM_Simulator::setFluenceRate(std::array<float,6> laserPose,const float laserPower,const float beamWaist, const float lambda)
{
	Eigen::VectorXf FR = Eigen::VectorXf::Zero(mesh_->nodes().size());
	
	// Precompute constants
	const float pi = 3.14159265358979323846f;
	// ASSUMING THERE IS NO ORIENTATION SHIFT ON THE LASER
	//TODO: account for orientation shift on the laser
	// I(x,y,z) = 2*P/(pi*w^2) * exp(-2*(x^2 + y^2)/w^2 - mua*z)

	const float mua = thermalModel_->MUA;
	
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

void FEM_Simulator::setFluenceRate(Eigen::Vector<float, 6> laserPose, const float laserPower, const float beamWaist, const float lambda)
{
	std::array<float,6> laserPoseVec;
	for (int i = 0; i < 6; i++) {
		laserPoseVec[i] = laserPose(i);
	}
	setFluenceRate(laserPoseVec, laserPower, beamWaist, lambda);
}

void FEM_Simulator::setDt(float dt)
{
	if (dt < 0)
		throw std::runtime_error("Time step must be greater than 0");
	dt_ = dt;
	parameterUpdate_ = true;
}


void FEM_Simulator::setTC(float TC) {
	thermalModel_->TC = TC;
	parameterUpdate_ = true;
}

void FEM_Simulator::setVHC(float VHC) {
	thermalModel_->VHC = VHC;
	parameterUpdate_ = true;
}

void FEM_Simulator::setMUA(float MUA) {
	thermalModel_->MUA = MUA;
	parameterUpdate_ = true;
}

void FEM_Simulator::setHTC(float HTC) {
	thermalModel_->HTC = HTC;
	parameterUpdate_ = true;
}

void FEM_Simulator::setHeatFlux(float heatFlux)
{
	thermalModel_->heatFlux = heatFlux;
	parameterUpdate_ = true;
}

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	thermalModel_->ambientTemp = ambientTemp;
	parameterUpdate_ = true;
}


void FEM_Simulator::setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations)
{
	int nSensors = tempSensorLocations.size();

	sensorTemps_.clear();
	sensorTemps_.resize(nSensors);
	sensors_.clear();
	sensors_.resize(nSensors);
	for (int s = 0; s < nSensors; s++)
	{	
		Sensor currSens; 
		std::array<float, 3> xi;
		long e = mesh_->findPosInMesh(tempSensorLocations[s],xi);
		if (e < 0)
			throw std::invalid_argument("All sensor locations must be within the mesh");
		currSens.elemIdx = e;
		currSens.pos = tempSensorLocations[s];
		currSens.xi = xi;
		currSens.temp = 0;
		sensors_[s] = currSens;
		sensorTemps_[s] = 0;
	}
}

std::vector<std::array<float, 3>> FEM_Simulator::sensorLocations() const
{
	std::vector<std::array<float, 3>> sensorLocations(sensors_.size());
	for (int s = 0; s < sensors_.size(); s++)
	{
		sensorLocations[s] = sensors_[s].pos;
	}

	return sensorLocations;
}

void FEM_Simulator::setMesh(const Mesh& mesh)
{
	mesh_ = &mesh;
	initializeContainers();
}

int FEM_Simulator::enableGPU()
{
	return setGPU(true);
}

void FEM_Simulator::disableGPU()
{
	setGPU(false);
}

int FEM_Simulator::setGPU(bool useGPU)
{
	// returns 1 if the gpu was enabled and 0 otherwise

	if (solver_ && (useGPU != gpuEnabled_))
	{
		// if we are turning off or turning on the gpu, reset the solver
		delete solver_;
		solver_ = nullptr;	
	}
#ifdef USE_CUDA
	if (useGPU)
	{
		// Check to make sure we have a cuda device 
		int deviceCount = 0;
		cudaError_t error = cudaGetDeviceCount(&deviceCount);

		if (error != cudaSuccess) {
			if (!silentMode)
				std::cout << "CUDA error: " << cudaGetErrorString(error) << "\n";
		}

		if (deviceCount > 0)
		{
			gpuEnabled_ = true;
			return 1;
		}	
	}
#endif
	// default behavior
	gpuEnabled_ = false;
	return 0;
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
