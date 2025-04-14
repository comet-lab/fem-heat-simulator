#include "FEM_Simulator.h"
#include <iostream>

const int FEM_Simulator::A[8][3] = {{-1, -1, -1},{1,-1,-1},{-1,1,-1},{1,1,-1}, {-1,-1,1},{1,-1,1},{-1,1,1}, { 1,1,1 } };

FEM_Simulator::FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA, float HTC, int Nn1d)
{
	this->Nn1d = Nn1d;
	this->setTemp(Temp);
	this->setTissueSize(tissueSize);
	this->setLayer(tissueSize[2], (Temp[0][0].size()-1) / (Nn1d-1));
	this->setTC(TC);
	this->setVHC(VHC);
	this->setMUA(MUA);
	this->setHTC(HTC);
	
}

void FEM_Simulator::performTimeStepping()
{
	auto startTime = std::chrono::high_resolution_clock::now();
	auto stopTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);

	// Apply parameter specific multiplication for each global matrix.
	Eigen::SparseMatrix<float, Eigen::RowMajor> globK = this->Kint*this->TC + this->Kconv*this->HTC;
	this->M = this->M * this->VHC; // M Doesn't have any additions so we just multiply it by the constant
	Eigen::VectorXf globF = this->Firr*this->MUA + this->Fconv*this->HTC + this->Fq;

	//this->NFR = NFR;
	int numElems = this->gridSize[0] * this->gridSize[1] * this->gridSize[2];
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];

	// Create sensorTemperature Vectors and reserve sizes
	int nSensors = this->tempSensorLocations.size();
	if (nSensors == 0) { // if there weren't any sensors initialized, put one at 0,0,0
		nSensors = 1;
		this->tempSensorLocations.push_back({ 0,0,0 });
	}
	this->sensorTemps.resize(nSensors);
	for (int s = 0; s < nSensors; s++) {
		this->sensorTemps[s].resize(round(this->tFinal / deltaT) + 1);
	}

	// Solve Euler Family 
	// Initialize d vector
	Eigen::VectorXf dVec(nNodes - dirichletNodes.size());
	Eigen::VectorXf vVec(nNodes - dirichletNodes.size());
	Eigen::VectorXf dTilde(nNodes - dirichletNodes.size());
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> initSolver;
	int counter = 0;
	for (int n : validNodes) {
		dVec(counter) = this->Temp(n);
		counter++;
	}
	Eigen::SparseMatrix<float> LHSinit = this->M;
	initSolver.compute(LHSinit);
	Eigen::VectorXf RHSinit = (globF) - globK*dVec;
	vVec = initSolver.solve(RHSinit);

	this->updateTemperatureSensors(0, dVec);
	if (!this->silentMode) {
		stopTime = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "D initialized: " << duration.count() / 1000000.0 << std::endl;
		startTime = stopTime;
	}

	// Perform TimeStepping
	// Eigen documentation says using Lower|Upper gives the best performance for the solver with a full matrix. 
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> solver;
	Eigen::SparseMatrix<float> LHS = this->M + this->alpha * this->deltaT * globK;
	solver.compute(LHS);
	if (solver.info() != Eigen::Success) {
		std::cout << "Decomposition Failed" << std::endl;
	}
	if (!this->silentMode) {
		stopTime = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "Matrix Factorized: " << duration.count() / 1000000.0 << std::endl;
		startTime = stopTime;
	}

	Eigen::VectorXf RHS;
	for (float t = 1; t <= round(this->tFinal/this->deltaT); t ++) { 
		/*std::stringstream msg;
		msg << "T: " << t << ", TID: " << omp_get_thread_num() << "\n";
		std::cout << msg.str();*/
		dTilde = dVec + (1 - this->alpha) * this->deltaT * vVec;	
		RHS = globF - globK * dTilde;
		vVec = solver.solveWithGuess(RHS,vVec);
		//vVec = solver.solve(RHS);
		/*std::cout << "#iterations:     " << solver.iterations() << std::endl;
		std::cout << "estimated error: " << solver.error() << std::endl;*/
		if (solver.info() != Eigen::Success) {
			std::cout << "Issue With Solver" << std::endl;
		}
		dVec = dTilde + this->alpha * this->deltaT * vVec;

		this->updateTemperatureSensors(t, dVec);
	}
	if (!this->silentMode) {
		stopTime = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "Time Stepping Completed: " << duration.count() / 1000000.0 << std::endl;
		startTime = stopTime;
	}

	// Adjust our Temp with new d vector
	counter = 0;
	for (int n : validNodes) {
			//int nodeSub[3];
			//ind2sub(n, this->nodeSize, nodeSub);
			this->Temp(n) = dVec(counter);
			counter++;
	}

	if (!this->silentMode) {
		stopTime = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "Updated Temp Variable: " << duration.count() / 1000000.0 << std::endl;
		startTime = stopTime;
	}
}

void FEM_Simulator::createKMFelem()
{
	int nodeFace;
	int matrixInd[2];
	int elemNodeSize[3] = {this->Nn1d, this->Nn1d, this->Nn1d};
	int eSub[3];
	int AiSub[3];
	int BiSub[3];
	int globalNodeSub[3];
	int globalNodeIdx;
	int BglobalNodeIdx;
	bool dirichletFlag = false;
	bool fluxFlag = false;
	auto startTime = std::chrono::high_resolution_clock::now();

	if (!this->silentMode) {
		std::cout << "Creating Global Matrices" << std::endl;
	}

	int numElems = this->gridSize[0] * this->gridSize[1] * this->gridSize[2];
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];
	if (nNodes != (((this->Nn1d-1)*gridSize[0] + 1) * ((this->Nn1d - 1)*gridSize[1] + 1) * ((this->Nn1d - 1)*gridSize[2] + 1))) {
		std::cout << "Nodes does not match: \n" << "Elems: " << numElems << "\nNodes: " << nNodes << std::endl;
	}

	this->initializeElementNodeSurfaceMap(); // link nodes of an element to face of the element
	this->initializeBoundaryNodes(); // Create a mapping between global node number and index in global matrices and locating dirichlet nodes
	this->initializeElementMatrices(1); // Initialize the element matrices assuming we are in the first layer

	// Initialize matrices so that we don't have to resize them later
	this->Firr = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	// TODO: make these three vectors sparse because they will only be non zero on the boundary nodes
	this->Fconv = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->Fk = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->Fq = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());

	this->M = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->M.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), pow((this->Nn1d*2 - 1),3))); // at most 27 non-zero entries per column
	this->Kint = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->Kint.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), pow((this->Nn1d * 2 - 1), 3))); // at most 27 non-zero entries per column
	// The Kconv matrix may also be able to be initialized differently since we know that it will only have values on the boundary ndoes.
	this->Kconv = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->Kconv.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), pow((this->Nn1d * 2 - 1), 3))); // at most 27 non-zero entries per column
	int Nne = pow(this->Nn1d, 3); // number of nodes in an element is equal to the number of nodes in a single dimension cubed
	
	/*std::vector<Eigen::Triplet<float>> Ktriplets;
	Ktriplets.reserve(numElems * Nne * Nne);
	std::vector<Eigen::Triplet<float>> Mtriplets;
	Mtriplets.reserve(numElems * Nne * Nne);*/

	bool layerFlag = false;
	for (int e = 0; e < numElems; e++) {
		// When adding variable voxel height, we are preserving the nodal layout. The number of nodes in each x,y, and z axis is unchanged.
		// The only thing we are asssuming to change is the length along the z axis for elements after a certain threshold. 

		this->currElement.elementNumber = e;
		this->ind2sub(e, this->gridSize, eSub);
		if ((eSub[2] >= this->layerSize) && !layerFlag) {
			// we currently assume there will only be one element height change and once the transition has happened, 
			// we won't encounter any elements that have the original height. Therefore, we just reset J and all the elemental 
			// matrices once and we are good to go. 
			this->initializeElementMatrices(2);
			layerFlag = true;
		}

		for (int Ai = 0; Ai < Nne; Ai++) {
			this->ind2sub(Ai, elemNodeSize, AiSub);
			for (int ii = 0; ii < 3; ii++) {
				globalNodeSub[ii] = eSub[ii] * (this->Nn1d - 1) + AiSub[ii];
			}
			globalNodeIdx = globalNodeSub[0] + globalNodeSub[1] * this->nodeSize[0] + globalNodeSub[2] * this->nodeSize[0] * this->nodeSize[1];

			nodeFace = this->determineNodeFace(globalNodeIdx);
			matrixInd[0] = this->nodeMap[globalNodeIdx];
			if (matrixInd[0] >= 0) { // Verify that the node we are working with is not a dirichlet node.
				
				// Determine if the node lies on a boundary and then determine what kind of boundary
				dirichletFlag = false;
				fluxFlag = false;
				if (nodeFace > 0) { // This check saves a lot time since most nodes are not on a surface.
					for (int f = 0; f < 6; f++) { // Iterate through each face of the element
						if ((nodeFace >> f) & 1) { // Node lies on face f
							if ((this->boundaryType[f] == FLUX)) { // flux boundary
								this->Fq(matrixInd[0]) += this->FeQ(Ai, f);
							}
							else if (this->boundaryType[f] == CONVECTION) { // Convection Boundary
								this->Fconv(matrixInd[0]) += this->FeConv(Ai, f);
								for (int Bi : this->elemNodeSurfaceMap[f]) {
									this->ind2sub(Bi, elemNodeSize, BiSub);
									int BglobalNodeSub[3] = { eSub[0] * (this->Nn1d - 1) + BiSub[0], eSub[1] * (this->Nn1d - 1) + BiSub[1], eSub[2] * (this->Nn1d - 1) + BiSub[2] };
									BglobalNodeIdx = BglobalNodeSub[0] + BglobalNodeSub[1] * this->nodeSize[0] + BglobalNodeSub[2] * this->nodeSize[0] * this->nodeSize[1];

									matrixInd[1] = this->nodeMap[BglobalNodeIdx];
									if (matrixInd[1] >= 0) {
										//int AiBi = Bi * Nne + Ai; // had to be creative here to encode Ai and Bi in a single variable. We are using base Nne. 
										//// If we say Nne = 8, if Bi is 1 and Ai is 7, the value is 15. 15 in base 8 is 17. 
										this->Kconv.coeffRef(matrixInd[0], matrixInd[1]) += this->KeConv[f](Ai, Bi);
										//Ktriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->KeConv[f](Ai, Bi)));
									}
									else {
										this->Fconv(matrixInd[0]) += -this->KeConv[f](Ai, Bi) * this->Temp(BglobalNodeIdx);
									}
								}
							} 
						} // if Node is face f
					} // iterate through faces
				} // if node is a face

				for (int Bi = 0; Bi < Nne; Bi++) {
					this->ind2sub(Bi, elemNodeSize, BiSub);
					int BglobalNodeSub[3] = { eSub[0] * (this->Nn1d - 1) + BiSub[0], eSub[1] * (this->Nn1d - 1) + BiSub[1], eSub[2] * (this->Nn1d - 1) + BiSub[2] };
					BglobalNodeIdx = BglobalNodeSub[0] + BglobalNodeSub[1] * this->nodeSize[0] + BglobalNodeSub[2] * this->nodeSize[0] * this->nodeSize[1];

					matrixInd[1] = this->nodeMap[BglobalNodeIdx];
					if (matrixInd[1] >= 0) { // Ai and Bi are both valid positions so we add it to Kint and M and Firr
						this->Kint.coeffRef(matrixInd[0], matrixInd[1]) += this->KeInt(Ai, Bi);
						//Ktriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->KeInt(Ai, Bi)));
						this->M.coeffRef(matrixInd[0], matrixInd[1]) += this->Me(Ai, Bi);
						//Mtriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->Me(Ai, Bi)));
						if (elemNFR) {// element-wise NFR so we assume each node on the element has NFR
							this->Firr(matrixInd[0]) += this->FeIrr(Ai, Bi) * this->NFR(e);
						}
						else {//nodal NFR so use as given
							this->Firr(matrixInd[0]) += this->FeIrr(Ai, Bi) * this->NFR(BglobalNodeIdx);
						}
					}
					else if (matrixInd[1] < 0) { // valid row, but column is dirichlet node so we add to Firr... could be an if - else
						this->Fk(matrixInd[0]) += -this->KeInt(Ai, Bi) * this->Temp(BglobalNodeIdx);
						if (elemNFR) { // element-wise NFR so we assume each node on the element has NFR
							this->Firr(matrixInd[0]) += this->FeIrr(Ai, Bi) * this->NFR(e);
						}
						else {//nodal NFR so use as given
							this->Firr(matrixInd[0]) += this->FeIrr(Ai, Bi) * this->NFR(BglobalNodeIdx);
						}
					} // if both are invalid we ignore, if column is valid but row is invalid we ignore
				} // For loop through Bi
			} // If our node is not a dirichlet node
		} // For loop through Ai
	}
	//this->Kint.setFromTriplets(Ktriplets.begin(),Ktriplets.end());
	//this->M.setFromTriplets(Mtriplets.begin(),Mtriplets.end());
	this->Kconv.makeCompressed();
	this->Kint.makeCompressed();
	this->M.makeCompressed();

	if (!this->silentMode) {
		auto stopTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "Built the Matrices: " << duration.count() / 1000000.0 << std::endl;
	}
}

void FEM_Simulator::updateTemperatureSensors(int timeIdx, Eigen::VectorXf& dVec) {
	/*TODO THIS DOES NOT WORK IF WE AREN'T USING LINEAR BASIS FUNCTIONS */
	/*TODO SENSORS IN LAYER 2 ALSO DON'T WORK. THEY MIGHT WORK IN LAYER 1*/
	int nSensors = this->tempSensorLocations.size();
	int Nne = pow(this->Nn1d, 3);
	// Input time = 0 information into temperature sensors
	//spacingLayer contains the distance between nodes in eacy layer.
	float spacingLayer1[3] = { this->tissueSize[0] / float(this->gridSize[0]), this->tissueSize[1] / float(this->gridSize[1]) , this->layerHeight / float(this->layerSize) };
	float spacingLayer2[3] = { this->tissueSize[0] / float(this->gridSize[0]), this->tissueSize[1] / float(this->gridSize[1]) , (this->tissueSize[2]-this->layerHeight) / float(this->gridSize[2]-this->layerSize) };
	for (int s = 0; s < nSensors; s++) {
		std::array<float,3> sensorLocation = this->tempSensorLocations[s];
		//globalNodeStart holds the global node subscript of the first node in the element that contains this sensor
		int globalNodeStart[3] = { floor((sensorLocation[0] + this->tissueSize[0] / 2.0f) / spacingLayer1[0]),
			floor((sensorLocation[1] + this->tissueSize[1] / 2.0f) / spacingLayer1[1]),
			floor(sensorLocation[2] / spacingLayer1[2]) };
		if (sensorLocation[2] > this->layerHeight) {
			// This should compensate for the change in layer appropriately.
			globalNodeStart[2] = this->layerSize + floor((sensorLocation[2] - this->layerHeight) / spacingLayer2[2]);
		}
		float tempValue = 0;
		float xi[3];
		// Xi should be  avalue between -1 and 1 along each axis relating the placement of the sensor in that element.
		xi[0] = -1 + ((sensorLocation[0] + this->tissueSize[0] / 2.0f) / spacingLayer1[0] - globalNodeStart[0]) * 2;
		xi[1] = -1 + ((sensorLocation[1] + this->tissueSize[1] / 2.0f) / spacingLayer1[1] - globalNodeStart[1]) * 2;
		xi[2] = -1 + ((sensorLocation[2]) / spacingLayer1[2] - globalNodeStart[2]) * 2;
		if (sensorLocation[2] > this->layerHeight) {
			xi[2] = -1 + (this->layerSize + (sensorLocation[2] - this->layerHeight) / spacingLayer2[2] - globalNodeStart[2]) * 2;
		}
		for (int Ai = 0; Ai < Nne; Ai++) { // iterate through each node in the element
			// adjust the global starting node based on the current node we should be visiting
			int globalNodeSub[3] = { globalNodeStart[0] + (Ai&1),globalNodeStart[1]+((Ai & 2) >> 1), globalNodeStart[2]+ ((Ai & 4) >> 2) };
			// convert subscript to index
			int globalNode = globalNodeSub[0] + globalNodeSub[1] * this->nodeSize[0] + globalNodeSub[2] * this->nodeSize[0] * this->nodeSize[1];
			// add temperature contribution of node
			if (this->nodeMap[globalNode] >= 0) {//non-dirichlet node
				tempValue += this->calculateNA(xi, Ai) * dVec(this->nodeMap[globalNode]);
			}
			else { // dirichlet node
				tempValue += this->calculateNA(xi, Ai) * this->Temp(globalNode);
			}
			
		}
		this->sensorTemps[s][timeIdx] = tempValue;
	}

}

float FEM_Simulator::calculateNA(float xi[3], int Ai)
{
	float output = 1.0f;
	int AiVec[3]; 
	int size[3] = { this->Nn1d,this->Nn1d,this->Nn1d };
	//2-node ex: 0-(0,0,0), 1-(1,0,0), 2-(0,1,0), 3-(1,1,0), 4-(0,0,1), 5-(1,0,1), 6-(0,1,1), 7-(1,1,1)
	ind2sub(Ai, size, AiVec); // convert element node A to a subscript (xi,eta,zeta)
	for (int i = 0; i < 3; i++) {
		// multiply each polynomial shape function in 1D across all 3 dimensions
		output *= this->calculateNABase(xi[i], AiVec[i]);
	}
	return output;
}

float FEM_Simulator::calculateNABase(float xi, int Ai) {
	/* This function calculates the building block for the basis functions given the number of nodes in 1 Dimension
	* It uses the equation \prod^Nn1d_{B = 1; B != A} (xi - xi^B)/(xi^A - xi^B)
	*/
	float output = 1.0f;
	// This will produce a value of -1,1 for 2-node elements, and -1,0,1 for 3-node elements
	float xiA = -1 + Ai * (2 / float(this->Nn1d - 1));
	// Get the product of the linear functions to build polynomial shape function in 1D
	for (int i = 0; i < this->Nn1d; i++) { // for 2-node elements its a single product.
		if (i != Ai) {
			float xiB = -1 + i * (2 / float(this->Nn1d - 1)); //same as above
			output *= (xi - xiB) / (xiA - xiB);
		}
	}
	return output;
}

float FEM_Simulator::calculateNADotBase(float xi, int Ai) {
	float output = 0;
	if (this->Nn1d == 2) {
		if (Ai == 0) {
			output = -1 / 2.0f;
		}
		else if (Ai == 1) {
			output = 1 / 2.0f;
		}
	}
	else if (this->Nn1d == 3) {
		if (Ai == 0) {
			output = (2 * xi - 1) / 2.0f;
		}
		else if (Ai == 1) {
			output = -2 * xi;
		}
		else if (Ai == 2) {
			output = (2 * xi + 1) / 2.0f;
		}
	}
	return output;
}

Eigen::Vector3<float> FEM_Simulator::calculateNA_dot(float xi[3], int Ai)
{
	Eigen::Vector3<float> NA_dot;
	int AiVec[3];
	int size[3] = { this->Nn1d,this->Nn1d,this->Nn1d };
	this->ind2sub(Ai, size, AiVec);
	NA_dot(0) = this->calculateNADotBase(xi[0], AiVec[0]) * this->calculateNABase(xi[1], AiVec[1]) * this->calculateNABase(xi[2], AiVec[2]);
	NA_dot(1) = this->calculateNADotBase(xi[1], AiVec[1]) * this->calculateNABase(xi[0], AiVec[0]) * this->calculateNABase(xi[2], AiVec[2]);
	NA_dot(2) = this->calculateNADotBase(xi[2], AiVec[2]) * this->calculateNABase(xi[0], AiVec[0]) * this->calculateNABase(xi[1], AiVec[1]);
	return NA_dot;
}

Eigen::Matrix3<float> FEM_Simulator::calculateJ(int layer)
{
	/* While below is the proper way to calculat the Jacobian for an arbitrary element, we can take advantage of the fact that
	we are using a cubiod whose axis (x,y,z) are aligned with our axis in the bi-unit domain (xi, eta, zeta). Therefore, the Jacobian
	will only contain values along the diagonal and their values will be equal to (deltaX/2, deltaY/2, and deltaZ/2). 

	Eigen::Vector3<float> NA_dot;
	for (int i = 0; i < 3; i++) {
		J(i, 0) = 0; // Make sure we don't have dummy values stored
		J(i, 1) = 0; // Make sure we don't have dummy values stored
		J(i, 2) = 0; // Make sure we don't have dummy values stored
		for (int Ai = 0; Ai < 8; Ai++) {
			FEM_Simulator::calculateNA_dot(xi, Ai, NA_dot);
			for (int j = 0; j < 3; j++) {
				J(i, j) += NA_dot(j) * pos[Ai][i];
			}
		}
	}*/

	Eigen::Matrix3<float> J;
	//**** ASSUMING X-Y VOXEL SIZE IS CONSTANT THROUGHOUT VOLUME **********
	// we assume the z height can change once in the volume. 
	float deltaX = this->tissueSize[0] / float(this->gridSize[0]);
	float deltaY = this->tissueSize[1] / float(this->gridSize[1]);
	float deltaZ = 0;
	if (layer == 1) {
		deltaZ = this->layerHeight / float(this->layerSize);
	}
	else if (layer == 2) {
		deltaZ = (this->tissueSize[2] - this->layerHeight) / float(this->gridSize[2] - this->layerSize);
	}
	else {
		std::cout << "layer must be 1 or 2" << std::endl;
		throw std::invalid_argument("Invalid layer");
	}
	
	J(0, 0) = deltaX / 2.0;
	J(0, 1) = 0;
	J(0, 2) = 0;
	J(1, 0) = 0;
	J(1, 1) = deltaY/2.0;
	J(1, 2) = 0;
	J(2, 0) = 0;
	J(2, 1) = 0;
	J(2, 2) = deltaZ / 2.0;
	return J;

}

Eigen::Matrix2<float> FEM_Simulator::calculateJs(int dim, int layer)
{
	// dim should be +-{1,2,3}. The dimension indiciates the axis of the normal vector of the plane. 
	// +1 is equivalent to (1,0,0) normal vector. -3 is equivalent to (0,0,-1) normal vector. 
	// We assume the values of xi correspond to the values of the remaining two axis in ascending order.
	// If dim = 2, then xi[0] is for the x-axis and xi[1] is for the z axis. 

	//**** ASSUMING X-Y VOXEL SIZE IS CONSTANT THROUGHOUT VOLUME **********
	// we assume the z height can change once in the volume. 
	float deltaX = this->tissueSize[0] / float(this->gridSize[0]);
	float deltaY = this->tissueSize[1] / float(this->gridSize[1]);
	float deltaZ = 0;
	if (layer == 1) {
		deltaZ = this->layerHeight / float(this->layerSize);
	}
	else if (layer == 2) {
		deltaZ = (this->tissueSize[2] - this->layerHeight) / float(this->gridSize[2] - this->layerSize);
	}
	else {
		std::cout << "layer must be 1 or 2" << std::endl;
		throw std::invalid_argument("Invalid layer");
	}
	int direction = dim / abs(dim);
	dim = abs(dim);
	Eigen::Matrix2f Js;
	Js(0, 1) = 0;
	Js(1, 0) = 0;
	if (dim == 1) {
		Js(0, 0) = deltaY / 2.0;
		Js(1, 1) = deltaZ / 2.0;
	}
	else if (dim == 2) {
		Js(0, 0) = deltaX / 2.0;
		Js(1, 1) = deltaZ / 2.0;
	}
	else if (dim == 3) {
		Js(0, 0) = deltaX / 2.0;
		Js(1, 1) = deltaY / 2.0;
	}

	return Js;
	/* While below is the proper way to calculat the Jacobian for an arbitrary element, we can take advantage of the fact that
	we are using a cubiod whose axis (x,y,z) are aligned with our axis in the bi-unit domain (xi, eta, zeta). Therefore, the Jacobian
	will only contain values along the diagonal and their values will be equal to (deltaX/2, deltaY/2, and deltaZ/2).

	int direction = dim / abs(dim);
	dim = abs(dim);
	float xiExt[3] = { 0.0,0.0,0.0 };
	if (dim == 1) { // we are in the y-z plane
		xiExt[0] = direction;
		xiExt[1] = xi[0];
		xiExt[2] = xi[1];
		for (int i = 1; i < 3; i++) {
			Js(i, 0) = 0; // Make sure we don't have dummy values stored
			Js(i, 0) = 0; // Make sure we don't have dummy values stored
			for (int Ai = 0; Ai < 8; Ai++) {
				for (int j = 0; j < 2; j++) {
					if (j == 0) {
						Js(i, j) += FEM_Simulator::calculateNA_eta(xiExt, Ai) * pos[Ai][i];
					}
					else if (j == 1) {
						Js(i, j) += FEM_Simulator::calculateNA_zeta(xiExt, Ai) * pos[Ai][i];
					}
				}
			}
		}
	}
	else if (dim == 2) { // we are in the x-z plane
		xiExt[0] = xi[0];
		xiExt[1] = direction;
		xiExt[2] = xi[1];
		float NA_dot[3] = { 0,0,0 };
		for (int i = 0; i < 3; i = i + 2) {
			Js(i, 0) = 0; // Make sure we don't have dummy values stored
			Js(i, 0) = 0; // Make sure we don't have dummy values stored
			for (int Ai = 0; Ai < 8; Ai++) {
				for (int j = 0; j < 2; j++) {
					if (j == 0) {
						Js(i, j) += FEM_Simulator::calculateNA_xi(xiExt, Ai) * pos[Ai][i];
					}
					else if (j == 1) {
						Js(i, j) += FEM_Simulator::calculateNA_zeta(xiExt, Ai) * pos[Ai][i];
					}
				}
			}
		}
	}
	else if (dim == 3) { // we are in the x-y plane
		xiExt[0] = xi[0];
		xiExt[1] = xi[1];
		xiExt[2] = direction;
		for (int i = 0; i < 2; i++) {
			Js(i, 0) = 0; // Make sure we don't have dummy values stored
			Js(i, 0) = 0; // Make sure we don't have dummy values stored
			for (int Ai = 0; Ai < 8; Ai++) {
				for (int j = 0; j < 2; j++) {
					if (j == 0) {
						Js(i, j) += FEM_Simulator::calculateNA_xi(xiExt, Ai) * pos[Ai][i];
					}
					else if (j == 1) {
						Js(i, j) += FEM_Simulator::calculateNA_eta(xiExt, Ai) * pos[Ai][i];
					}
				}
			}
		}
	} */
}

void FEM_Simulator::ind2sub(int index, int size[3], int sub[3])
{
	sub[0] = 0; sub[1] = 0; sub[2] = 0; // Making sure output array has zeros

	sub[0] = (index % size[0]);
	sub[1] = (index % (size[0] * size[1])) / size[0];
	sub[2] = index / (size[0] * size[1]);

	return;
}

float FEM_Simulator::integrate(float (FEM_Simulator::* func)(float[3], int, int), int points, int dim, int param1, int param2)
{
	std::vector<float> zeros;
	std::vector<float> weights;
	float output = 0;
	// TODO: Replace with automatic determination of zeros and weights
	if (points == 1) {
		zeros.push_back(0.0f);
		weights.push_back(2.0f);
	}
	else if (points == 2) {
		zeros.push_back(-sqrt(1 / 3.0f));
		zeros.push_back(sqrt(1 / 3.0f));
		weights.push_back(1.0f);
		weights.push_back(1.0f);
	}
	else if (points == 3) {
		zeros.push_back(-sqrt(3 / 5.0f));
		zeros.push_back(0.0f);
		zeros.push_back(sqrt(3 / 5.0f));
		weights.push_back(5 / 9.0f);
		weights.push_back(8 / 9.0f);
		weights.push_back(5 / 9.0f);
	}

	for (int i = 0; i < points; i++) {
		for (int j = 0; j < points; j++) {
			if (abs(dim) == 0) { // integrate across all 3 axis
				for (int k = 0; k < points; k++) {
					float xi[3] = { zeros[i], zeros[j], zeros[k] };
					output += (this->*func)(xi, param1, param2) * weights[i] * weights[j] * weights[k];
				}
			} if (abs(dim) == 1) { // we are in the y-z plane
				float xi[3] = { dim / abs(dim), zeros[i], zeros[j] };
				output += (this->*func)(xi, param1, param2) * weights[i] * weights[j];
			}
			else if (abs(dim) == 2) { // we are in the x-z plane
				float xi[3] = { zeros[i], dim / abs(dim), zeros[j] };
				output += (this->*func)(xi, param1, param2) * weights[i] * weights[j];
			}
			else if (abs(dim) == 3) { // we are in the x-y plane
				float xi[3] = { zeros[i], zeros[j], dim / abs(dim) };
				output += (this->*func)(xi, param1, param2) * weights[i] * weights[j];
			}
		}
	}
	return output;
}

void FEM_Simulator::getGlobalPosition(int globalNode, float position[3])
{
	// This function is only needed if we are using a non-uniform cuboid. Since we are assuming a uniform cuboid
	// this function has not been tested. 
	position[0] = 0; position[1] = 0; position[2] = 0;

	float deltaX = this->tissueSize[0] / float(this->gridSize[0]);
	float deltaY = this->tissueSize[1] / float(this->gridSize[1]);
	float deltaZ = this->tissueSize[2] / float(this->gridSize[2]);
	int sub[3];
	this->ind2sub(globalNode, this->nodeSize, sub);
	position[0] = sub[0] * deltaX;
	position[1] = sub[1] * deltaY;
	position[2] = sub[2] * deltaZ;
}

float FEM_Simulator::calcKintAB(float xi[3], int Ai, int Bi)
{
	float KABfunc = 0;
	Eigen::Vector3<float> NAdotA;
	Eigen::Vector3<float> NAdotB;
	Eigen::Matrix3<float> J = this->J;
	Eigen::Matrix3f Jinv = J.inverse();


	NAdotA = this->calculateNA_dot(xi, Ai);
	NAdotB = this->calculateNA_dot(xi, Bi);
	
	KABfunc = (NAdotA.transpose()* Jinv * Jinv.transpose()*NAdotB); // matrix math
	// The 1 should be replaced with this->TC if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	KABfunc = float(J.determinant() * 1 * KABfunc); // Type issues if this multiplication is done with the matrix math so i am doing it on its own line
	return KABfunc;
}

float FEM_Simulator::calcMAB(float xi[3], int Ai, int Bi)
{
	float MABfunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3<float> J = this->J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	// The 1 should be replaced with this->VHC if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	MABfunc = (NAa * NAb) * J.determinant() * 1; // matrix math
	return MABfunc;
}

float FEM_Simulator::calcFintAB(float xi[3], int Ai, int Bi)
{
	float FintFunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3<float> J = this->J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	// The 1 should be replaced with this->MUA if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	FintFunc = 1 * (NAa * NAb) * J.determinant();
	// Output of this still needs to get multiplied by the NFR at node Bi
	return FintFunc;
}

float FEM_Simulator::calcFqA(float xi[3], int Ai, int dim)
{
	float FjFunc = 0;
	float NAa;
	Eigen::Matrix2f Js; 
	dim = abs(dim);
	if (dim == 1) {
		Js = this->Js1;
	}
	else if (dim == 2) {
		Js = this->Js2;
	}
	else if (dim == 3) {
		Js = this->Js3;
	}

	NAa = this->calculateNA(xi, Ai);
	FjFunc = (NAa * this->Qn) * Js.determinant();
	return FjFunc;
}

float FEM_Simulator::calcFconvA(float xi[3], int Ai, int dim)
{
	float FvFunc = 0;
	float NAa;
	Eigen::Matrix2f Js;
	dim = abs(dim);
	if (dim == 1) {
		Js = this->Js1;
	}
	else if (dim == 2) {
		Js = this->Js2;
	}
	else if (dim == 3) {
		Js = this->Js3;
	}

	NAa = this->calculateNA(xi, Ai);
	// The 1 should be replaced with this->HTC if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	FvFunc = (NAa * 1) * this->ambientTemp * Js.determinant();
	return FvFunc;
}

float FEM_Simulator::calcKconvAB(float xi[3], int AiBi, int dim)
{
	int Nne = pow(this->Nn1d, 3);
	float FvuFunc = 0;
	int Ai = AiBi % Nne; // AiBi is passed in in base 8. the ones digit is Ai, the 8s digit is Bi.
	int Bi = AiBi / Nne;
	float NAa;
	float NAb;
	int dir = dim / abs(dim);
	dim = abs(dim);
	Eigen::Matrix2f Js;
	if (dim == 1) { // y-z plane
		Js = this->Js1;
	}
	else if (dim == 2) { // x-z plane
		Js = this->Js2;
	}
	else if (dim == 3) { // x-y plane
		Js = this->Js3;
	}
	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	// The 1 should be replaced with this->HTC if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	FvuFunc += (NAa * NAb * 1) * Js.determinant();
	
	return FvuFunc;
}

void FEM_Simulator::initializeBoundaryNodes()
{
	// we only need to scan nodes on the surface. Since we are assuming a cuboid this is easy to predetermine
	// Create mapping
	this->validNodes.clear();
	this->dirichletNodes.clear();

	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];
	int positionCounter = 0;
	bool validNode = true;
	this->nodeMap = std::vector<int>(nNodes, -1); // initialize the mapping to -1. -1 indicates the node passed in is a dirichlet node.
	for (int i = 0; i < nNodes; i++) {
		int nodeFace = this->determineNodeFace(i);
		int nodeSub[3];
		this->ind2sub(i, this->nodeSize, nodeSub);
		validNode = true;
		// Determine if the node lies on a boundary and then determine what kind of boundary
		if (nodeFace != 0) { // This check saves a lot time since most nodes are not on a surface.
			for (int f = 0; f < 6; f++) {
				if ((nodeFace >> f) & 1) { // Node lies on face f
					if (this->boundaryType[f] == HEATSINK) { // flux boundary
						validNode = false;
						this->dirichletNodes.push_back(i);
						break;
					} // if heatsink
				} // if node is on boundary
			} // loop through each face
		} // if node is not on a face
		if (validNode) { // if none of faces of the nodes are dirichlet boundaries, the node is valid
			this->validNodes.push_back(i);
			this->nodeMap[i] = positionCounter;
			positionCounter++;
		} // The else condition is handled above in the for loop. 
	}
}

void FEM_Simulator::initializeElementNodeSurfaceMap()
{
	// This function is tightly linked with @determineNodeFace to determine the ordering of the nodes
	// Again we define the origin on the top tissue surface, centered -- This is a left handed coordinate frame
	// The z-axis points into the tissue
	// The x-axis points towards the front face of the tissue
	// The y-axis points towards the right face of the tissue 
	int Nne = pow(this->Nn1d, 3);
	int AiSub[3];
	int elemNodeSize[3] = { this->Nn1d,this->Nn1d, this->Nn1d };
	
	for (int Ai = 0; Ai < Nne; Ai++) {
		if (Ai == 0) {
			for (int f = 0; f < 6; f++) {
				this->elemNodeSurfaceMap[f].clear();
			}
		}
		this->ind2sub(Ai, elemNodeSize, AiSub);
		if (AiSub[2] == 0) { //top
			this->elemNodeSurfaceMap[0].push_back(Ai);
		}
		if (AiSub[2] == (this->Nn1d - 1)) { //bottom
			this->elemNodeSurfaceMap[1].push_back(Ai);
		}
		if (AiSub[0] == (this->Nn1d - 1)) { //Front
			this->elemNodeSurfaceMap[2].push_back(Ai);
		}
		if (AiSub[1] == (this->Nn1d - 1)) { //Right
			this->elemNodeSurfaceMap[3].push_back(Ai);
		}
		if (AiSub[0] == 0) { //Back
			this->elemNodeSurfaceMap[4].push_back(Ai);
		}
		if (AiSub[1] == 0) { // Left
			this->elemNodeSurfaceMap[5].push_back(Ai);
		}
	}
}

void FEM_Simulator::initializeElementMatrices(int layer)
{	
	//This function will initialize the elemental matrices of a node
	// It starts with the jacobian because the Jacobian is used in every integration. 
	this->setJ(layer);
	this->setKeInt();
	this->setFeIrr();
	this->setMe();
	this->setFeQ();
	this->setFeConv();
	this->setKeConv();
}

int FEM_Simulator::determineNodeFace(int globalNode)
{	// This function implicitly determines the position of our reference frame on the surface of the tissue
	// and how that reference frame relates to the index of the matrix. This function is
	// linked with @initializeElementNodeSurfMap(). Changing one requires a change in the other.

	// We define the origin on the top tissue surface, centered -- This is a left handed coordinate frame
	// The z-axis points into the tissue
	// The x-axis points towards the front face of the tissue
	// The y-axis points towards the right face of the tissue 
	int output = INTERNAL;
	int nodeSub[3];
	this->ind2sub(globalNode, this->nodeSize, nodeSub);
	if (nodeSub[2] == 0) { 
		// Nodes on the top of the tissue have z-element = 0
		output += TOP;
	}
	if (nodeSub[2] == (this->nodeSize[2] - 1)) {
		// Nodes on the bottom of the tissue have z-element = nodeSize[2]-1
		output += BOTTOM;
	}
	if (nodeSub[0] == (this->nodeSize[0] - 1)) {
		// Nodes on the front of the tissue have x-element = nodeSize[0] - 1
		output += FRONT;
	}
	if (nodeSub[1] == (this->nodeSize[0] - 1)) {
		// Nodes on the right of the tissue have y-element = nodeSize[1] - 1
		output += RIGHT;
	}
	if (nodeSub[0] == 0) {
		// Nodes on the back of the the tissue have x-element = 0
		output += BACK;
	}
	if (nodeSub[1] == 0) {
		// Nodes on the left of the tissue have y-element = 0
		output += LEFT;
	}
	return output;
}

void FEM_Simulator::setTemp(std::vector<std::vector<std::vector<float>>> Temp) {
	
	this->Temp = Eigen::VectorXf::Zero(Temp.size() * Temp[0].size() * Temp[0][0].size());
	// Convert nested vectors into a single column Eigen Vector. 
	for (int i = 0; i < Temp.size(); i++) // associated with x and is columns of matlab matrix
	{
		for (int j = 0; j < Temp[0].size(); j++) // associated with y and is rows of matlab matrix
		{
			for (int k = 0; k < Temp[0][0].size(); k++) // associated with z and is depth of matlab matrix
			{
				this->Temp(i + j*Temp.size() + k*Temp.size()*Temp[0].size()) = Temp[i][j][k];
			}
		}
	}
	int gridSize[3]; 
	if (((Temp.size() - 1) % (this->Nn1d - 1) != 0)|| ((Temp[0].size() - 1) % (this->Nn1d - 1) != 0) || ((Temp[0][0].size() - 1) % (this->Nn1d - 1) != 0)) {
		std::cout << "Invalid Node dimensions given the number of nodes in a single elemental axis" << std::endl;
	}
	gridSize[0] = (Temp.size() - 1) / (this->Nn1d - 1); // Temp contains the temperature at the nodes, so we need to subtract 1 to get the elements
	gridSize[1] = (Temp[0].size() - 1) / (this->Nn1d - 1);
	gridSize[2] = (Temp[0][0].size() - 1) / (this->Nn1d - 1);
	this->setGridSize(gridSize);
}

void FEM_Simulator::setTemp(Eigen::VectorXf &Temp)
{	
	//TODO make sure Temp is the correct size
	if (this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2] != Temp.size()) {
		throw std::runtime_error("Total number of elements in Temp does not match current Node size.");
	}
	this->Temp = Temp;
}

std::vector<std::vector<std::vector<float>>> FEM_Simulator::getTemp()
{
	std::vector<std::vector<std::vector<float>>> 
		TempOut(this->nodeSize[0], std::vector<std::vector<float>>(this->nodeSize[1], std::vector<float>(this->nodeSize[2])));

	for (int i = 0; i < this->nodeSize[0]; i++) {
		for (int j = 0; j < this->nodeSize[1]; j++) {
			for (int k = 0; k < this->nodeSize[2]; k++) {
				TempOut[i][j][k] = Temp(i + j * (this->nodeSize[0]) + k * (this->nodeSize[0] * this->nodeSize[1]));
			}
		}
	}

	return TempOut;
}

void FEM_Simulator::setNFR(std::vector<std::vector<std::vector<float>>> NFR)
{
	this->NFR = Eigen::VectorXf::Zero(NFR.size() * NFR[0].size() * NFR[0][0].size());
	// Convert nested vectors into a single column Eigen Vector. 
	for (int i = 0; i < NFR.size(); i++) // associated with x and is columns of matlab matrix
	{
		for (int j = 0; j < NFR[0].size(); j++) // associated with y and is rows of matlab matrix
		{
			for (int k = 0; k < NFR[0][0].size(); k++) // associated with z and is depth of matlab matrix
			{
				this->NFR(i + j * NFR.size() + k * NFR.size() * NFR[0].size()) = NFR[i][j][k];
			}
		}
	}

	if ((NFR.size() == this->gridSize[0]) && (NFR[0].size() == this->gridSize[1]) && (NFR[0][0].size() == this->gridSize[2])) {
		this->elemNFR = true;
	}
	else if ((NFR.size() == this->nodeSize[0]) && (NFR[0].size() == this->nodeSize[1]) && (NFR[0][0].size() == this->nodeSize[2])) {
		this->elemNFR = false;
	}
	else {
		std::cout << "NFR must have the same number of entries as the node space or element space" << std::endl;
		throw std::invalid_argument("NFR must have the same number of entries as the node space or element space");
	}
}

void FEM_Simulator::setNFR(Eigen::VectorXf& NFR)
{
	this->NFR = NFR;
	//TODO Check for element or nodal NFR;
	this->elemNFR = false;
}

void FEM_Simulator::setNFR(float laserPose[6], float laserPower, float beamWaist)
{
	this->NFR = Eigen::VectorXf::Zero(this->nodeSize[0]* this->nodeSize[1]* this->nodeSize[2]);
	float lambda = 10.6 * pow(10, -4); // wavelength of laser in cm
	// ASSUMING THERE IS NO ORIENTATION SHIFT ON THE LASER
	//TODO: account for orientation shift on the laser
	// I(x,y,z) = 2*P/(pi*w^2) * exp(-2*(x^2 + y^2)/w^2 - mua*z)

	float irr = 0;
	float width = 0; 
	float xPos = -this->tissueSize[0] / 2;
	float xStep = this->tissueSize[0] / this->gridSize[0];
	float yPos = -this->tissueSize[0] / 2;
	float yStep = this->tissueSize[0] / this->gridSize[0];
	float zPos = 0;
	float zStep = this->layerHeight / this->layerSize;

	for (int i = 0; i < this->nodeSize[0]; i++) {
		yPos = -this->tissueSize[0] / 2;
		for (int j = 0; j < this->nodeSize[1]; j++) {
			zPos = 0;
			zStep = this->layerHeight / this->layerSize;
			for (int k = 0; k < this->nodeSize[2]; k++) {
				if (k >= this->layerSize) {
					// if we have passed the layer size
					zStep = (tissueSize[2] - this->layerHeight) / (this->gridSize[2] - this->layerSize);
				}
				// calculate beam width at depth
				width = beamWaist * std::sqrt(1 + pow((lambda * (zPos + laserPose[2]) / (std::acos(-1) * pow(beamWaist, 2))), 2));
				// calculate laser irradiance
				irr = 2 * laserPower / (std::acos(-1) * pow(width, 2)) * std::expf(-2 * (pow((xPos - laserPose[0]), 2) + pow((yPos - laserPose[1]), 2)) / pow(width,2) - this->MUA * zPos);
				// set laser irradiane
				this->NFR(i + j * this->nodeSize[0] + k * this->nodeSize[0] * this->nodeSize[1]) = irr;
				// increase z pos
				zPos = zPos + zStep;
			}
			// increase y pos
			yPos = yPos + yStep;
		}
		// increase x pos 
		xPos = xPos + xStep;
	}
}

void FEM_Simulator::setTissueSize(float tissueSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->tissueSize[i] = tissueSize[i];
	}
	this->setJ();
}

void FEM_Simulator::setLayer(float layerHeight, int layerSize) {
	if ((layerSize > this->gridSize[2])||(layerSize < 0)) {
		std::cout << "Invalid layer size. The layer must be equal"
			<< " to or less than the total number of elements in the z direction and greater than 0" << std::endl;
		throw std::runtime_error("Layer Size must be equal to or less than the toal number of elements in the z direction and greater than 0");
	}
	if ((layerHeight < 0) || (layerHeight > this->tissueSize[2])) {
		std::cout << "Invalid layer height. The layer dimension must be less than or equal to the tissue size " 
			<< "and greater than zero" << std::endl;
		throw std::runtime_error("Invalid layer height. The layer dimension must be less than or equal to the tissue size and greater than zero");
	}
	if ((layerHeight == 0) != (layerSize == 0)) {
		std::cout << "Layer Height must be 0 if layer size is 0 and vice versa" << std::endl;
		throw std::runtime_error("Layer Height must be 0 if layer size is 0 and vice versa");
	}
	if ((layerHeight == this->tissueSize[2]) != (layerSize == this->gridSize[2])) {
		std::cout << "Layer Height must be the tissue height if layer size is the grid size and vice versa" << std::endl;
		throw std::runtime_error("Layer Height must be the tissue height if layer size is the grid size and vice versa");
	}
	this->layerHeight = layerHeight;
	this->layerSize = layerSize;
	this->setJ(1);
}

void FEM_Simulator::setTC(float TC) {
	this->TC = TC;
}

void FEM_Simulator::setVHC(float VHC) {
	this->VHC = VHC;
}

void FEM_Simulator::setMUA(float MUA) {
	this->MUA = MUA;
}

void FEM_Simulator::setHTC(float HTC) {
	this->HTC = HTC;
}

void FEM_Simulator::setFlux(float Qn)
{
	this->Qn = Qn;
}

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	this->ambientTemp = ambientTemp;
}

void FEM_Simulator::setGridSize(int gridSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->gridSize[i] = gridSize[i];
		this->nodeSize[i] = gridSize[i] * (this->Nn1d - 1) + 1;
	}
}

void FEM_Simulator::setNodeSize(int nodeSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->gridSize[i] = nodeSize[i] - 1;
		this->nodeSize[i] = nodeSize[i];
	}
}

void FEM_Simulator::setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations)
{
	this->tempSensorLocations = tempSensorLocations;
	bool errorFlag = false;
	for (int s = 0; s < tempSensorLocations.size(); s++) {
		if ((tempSensorLocations[s][0] < -this->tissueSize[0] / 2.0f) || (tempSensorLocations[s][0] > this->tissueSize[0] / 2.0f) ||
			(tempSensorLocations[s][1] < -this->tissueSize[1] / 2.0f) || (tempSensorLocations[s][1] > this->tissueSize[1] / 2.0f) ||
			(tempSensorLocations[s][2] < 0) || (tempSensorLocations[s][2] > this->tissueSize[2])) {
			errorFlag = true;
			break;
		}	
	}
	if (errorFlag) {
		throw std::invalid_argument("All sensor locations must be within the tissue block");
	}
}

void FEM_Simulator::setJ(int layer) {
	this->J = this->calculateJ(layer);
	this->Js1 = this->calculateJs(1, layer);
	this->Js2 = this->calculateJs(2, layer);
	this->Js3 = this->calculateJs(3, layer);
	// The jacobian influences all of the other elemental matrices (KeInt, Me, Fe, etc)
	// Should setting the Jacobian automatically recompute the other matrices?
}

void FEM_Simulator::setBoundaryConditions(int BC[6])
{
	for (int i = 0; i < 6; i++) {
		this->boundaryType[i] = static_cast<boundaryCond>(BC[i]);
	}
	this->initializeBoundaryNodes();
}

Eigen::VectorXf FEM_Simulator::getSensorTemps()
{
	this->sensorTemps;
	Eigen::VectorXf sensorVector(sensorTemps.size());
	for (int s = 0; s < sensorTemps.size(); s++) {
		sensorVector(s) = this->sensorTemps[s][sensorTemps[s].size()-1];
	}
	return sensorVector;
}

void FEM_Simulator::setKeInt() {
	// Taking advantage of the fact that J is costant across element and TC is constant across elements
	int Nne = pow(this->Nn1d, 3);
	this->KeInt = Eigen::MatrixXf::Zero(Nne, Nne);
	for (int Ai = 0; Ai < Nne; Ai++) {
		for (int Bi = 0; Bi < Nne; Bi++) {
			this->KeInt(Ai, Bi) = this->integrate(&FEM_Simulator::calcKintAB, 3, 0, Ai, Bi);
		}
	}

}

void FEM_Simulator::setMe() {
	// Taking advantage of the fact that J is costant across element and VHC is constant across elements
	int Nne = pow(this->Nn1d, 3);
	this->Me = Eigen::MatrixXf::Zero(Nne, Nne);
	for (int Ai = 0; Ai < Nne; Ai++) {
		for (int Bi = 0; Bi < Nne; Bi++) {
			this->Me(Ai, Bi) = this->integrate(&FEM_Simulator::calcMAB, 3, 0, Ai, Bi);
		}
	}
}

void FEM_Simulator::setFeIrr()
{
	int Nne = pow(this->Nn1d, 3);
	this->FeIrr = Eigen::MatrixXf::Zero(Nne, Nne);
	for (int Ai = 0; Ai < Nne; Ai++) {
		for (int Bi = 0; Bi < Nne; Bi++) {
			this->FeIrr(Ai, Bi) = this->integrate(&FEM_Simulator::calcFintAB, 3, 0, Ai, Bi);
		}
	}
}

void FEM_Simulator::setFeQ() {
	int Nne = pow(this->Nn1d, 3);
	this->FeQ = Eigen::MatrixXf::Zero(Nne, 6);
	for (int f = 0; f < 6; f++) { // iterate through each face
		for (int Ai : this->elemNodeSurfaceMap[f]) { // Go through nodes on face surface 
			this->FeQ(Ai,f) = this->integrate(&FEM_Simulator::calcFqA, 3, this->dimMap[f], Ai, this->dimMap[f]); // calculate FjA
		}
	} // iterate through faces
}

void FEM_Simulator::setFeConv() {
	int Nne = pow(this->Nn1d, 3);
	this->FeConv = Eigen::MatrixXf::Zero(Nne, 6);
	for (int f = 0; f < 6; f++) { // iterate through each face
		for (int Ai : this->elemNodeSurfaceMap[f]) { // Go through nodes on face surface 
			this->FeConv(Ai,f) = this->integrate(&FEM_Simulator::calcFconvA, 3, this->dimMap[f], Ai, this->dimMap[f]); // calculate FjA
		}
	} // iterate through faces
}

void FEM_Simulator::setKeConv() {
	int Nne = pow(this->Nn1d, 3);
	for (int f = 0; f < 6; f++) {
		this->KeConv[f] = Eigen::MatrixXf::Zero(Nne, Nne);
		for (int Ai : this->elemNodeSurfaceMap[f]) {
			for (int Bi : this->elemNodeSurfaceMap[f]) {
				int AiBi = Bi * Nne + Ai; // had to be creative here to encode Ai and Bi in a single variable. We are using base 8. 
				// So if Bi is 1 and Ai is 7, the value is 15. 15 in base 8 is 17. 
				this->KeConv[f](Ai,Bi) = this->integrate(&FEM_Simulator::calcKconvAB, 3, this->dimMap[f], AiBi, this->dimMap[f]);
			}
		}
	}
}