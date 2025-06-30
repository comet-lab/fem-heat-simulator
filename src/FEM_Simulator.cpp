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

FEM_Simulator::FEM_Simulator(FEM_Simulator& inputSim)
{
	for (int i = 0; i < 3; i++) {
		this->elementsPerAxis[i] = inputSim.elementsPerAxis[i]; // Number of elements in x, y, and z [voxels]
		this->nodesPerAxis[i] = inputSim.nodesPerAxis[i]; // Number of nodes in x, y, and z. Should be elementsPerAxis + 1;
		this->tissueSize[i] = inputSim.tissueSize[i];  // Length of the tissue in x, y, and z [cm]
	}
	this->layerHeight = inputSim.layerHeight; // the z-location where we change element height
	this->elemsInLayer = inputSim.elemsInLayer; // The number of elements corresponding to the first layer height
	this->TC = inputSim.TC; // Thermal Conductivity [W/cm C]
	this->VHC = inputSim.VHC; // Volumetric Heat Capacity [W/cm^3]
	this->MUA = inputSim.MUA; // Absorption Coefficient [cm^-1]
	this->ambientTemp = inputSim.ambientTemp;  // Temperature surrounding the tissue for Convection [C]
	this->Temp = inputSim.Temp; // Our values for temperature at the nodes of the elements
	this->FluenceRate = inputSim.FluenceRate; // Our values for Heat addition
	this->alpha = inputSim.alpha; // time step weight
	this->deltaT = inputSim.deltaT; // time step [s]
	this->heatFlux = inputSim.heatFlux; // heat escaping the Neumann Boundary
	this->HTC = inputSim.HTC; // convective heat transfer coefficient [W/cm^2]
	this->Nn1d = inputSim.Nn1d;
	this->elemNFR = inputSim.elemNFR; // whether the FluenceRate pertains to an element or a node
	this->boundaryType = inputSim.boundaryType; // Individual boundary type for each face: 0: heat sink. 1: Flux Boundary. 2: Convective Boundary
	this->tempSensorLocations = inputSim.tempSensorLocations;
	this->sensorTemps = inputSim.sensorTemps;
	this->Kint = inputSim.Kint; // Conductivity matrix for non-dirichlet nodes
	this->Kconv = inputSim.Kconv; //Conductivity matrix due to convection
	this->M = inputSim.M; // Row Major because we fill it in one row at a time for nodal build -- elemental it doesn't matter
	// FirrElem = FirrElem*muA + Fconv*h + Fk*kappa + Fq
	this->FirrElem = inputSim.FirrElem; // forcing function due to irradiance
	this->Fconv = inputSim.Fconv; // forcing functino due to convection
	this->Fk = inputSim.Fk; // forcing function due conductivity matrix on dirichlet nodes
	this->Fq = inputSim.Fq; // forcing function due to constant flux boundary

	// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	this->KeInt = inputSim.KeInt; // Elemental Construction of Kint
	this->Me = inputSim.Me; // Elemental construction of M
	this->FeIrr = inputSim.FeIrr; // Elemental Construction of FirrElem
	// FeQ is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	this->FeQ = inputSim.FeQ; // Element Construction of Fq
	// FeConv is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	this->FeConv = inputSim.FeConv; // Elemental Construction of FConv
	// KeConv is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having local node coordinates A
	this->KeConv = inputSim.KeConv; // Elemental construction of KConv
	this->J = inputSim.J;
	this->Js1 = inputSim.Js1;
	this->Js2 = inputSim.Js2;
	this->Js3 = inputSim.Js3;
	this->validNodes = inputSim.validNodes; // global indicies on non-dirichlet boundary nodes
	this->dirichletNodes = inputSim.dirichletNodes;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations.
	// A value of -1 at index i, indicates that global node i is a dirichlet node.
	this->nodeMap = inputSim.nodeMap;
}

void FEM_Simulator::multiStep(float duration) {
	/* This function simulates multiple steps of the heat equation. A single step duration is given by deltaT. If the total
	duration is not easily divisible by deltaT, we will round (up or down) and potentially perform an extra step or one step 
	fewer. This asumes that initializeModel() has already been run to create the the global matrices. 
	It repeatedly calls to singleStep(). This function will also update the temperature sensors vector.  */ 
	auto startTime = std::chrono::high_resolution_clock::now();
	this->initializeSensorTemps(duration);
	this->updateTemperatureSensors(0);
	int numSteps = round(duration / this->deltaT);
	for (int t = 1; t <= numSteps; t++) {
		this->singleStep();
		this->updateTemperatureSensors(t);
	}

	startTime = this->printDuration("Time Stepping Completed: ", startTime);
}

void FEM_Simulator::singleStep() {
	/* Simulates a single step of the heat equation. A single step is given by the duration deltaT. To run single step,
	it is assumed that initializeModel() has already been run to create the global matrices and perform initial factorization
	of the matrix inversion. This function can handle changes in fluence rate or changes in tissue properties. */

	if (this->fluenceUpdate){ // check if fluence rate has changed
		if (this->elemNFR)
		{ 
			createFirr();
		}
		this->applyParameters();
		this->fluenceUpdate = false;
	}
	if (this->parameterUpdate) { // Check if parameters have changed
		this->applyParameters(); // Apply parameters
		this->LHS = globM + this->alpha * this->deltaT * globK; // Create new left hand side 
		this->LHS.makeCompressed(); // compress it for potential speed improvements
		this->cgSolver.factorize(this->LHS); // Perform factoriziation based on analysis which should have been called with initializeModel();
		if (this->cgSolver.info() != Eigen::Success) {
			std::cout << "Decomposition Failed" << std::endl;
		}
		this->parameterUpdate = false;
	}
	//this->cgSolver.factorize(this->LHS); // Perform factoriziation based on analysis which should have been called with initializeModel();
	// Explicit Forward Step (only if alpha < 1)
	this->dVec = this->dVec + (1 - this->alpha) * this->deltaT * this->vVec; // normally the output of this equation is assigned to dTilde for clarity...
	// Create Right-hand side of v(M + alpha*deltaT*K) = (F - K*dTilde);
	Eigen::VectorXf RHS = this->globF - this->globK * this->dVec; // ... and dTilde would be used here
	// Solve Ax = b using conjugate gradient
	// Time derivative should not change too much between time steps so we can use previous v to initialize conjugate gradient. 
	this->vVec = this->cgSolver.solveWithGuess(RHS,this->vVec);
	if (this->cgSolver.info() != Eigen::Success) {
		std::cout << "Issue With Solver" << std::endl;
	}
	// Implicit Backward Step (only if alpha > 0) 
	this->dVec = this->dVec + this->alpha * this->deltaT * vVec; // ... dTilde would also be on the righ-hand side here. 

	// Adjust our Temp with new d vector
	this->Temp(validNodes) = this->dVec;
}

void FEM_Simulator::createKMF()
{
	/* This function constructs the global matrices by iterating over each element and summing the local contributions. */
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

	int numElems = this->elementsPerAxis[0] * this->elementsPerAxis[1] * this->elementsPerAxis[2];
	int nNodes = this->nodesPerAxis[0] * this->nodesPerAxis[1] * this->nodesPerAxis[2];
	if (nNodes != (((this->Nn1d-1)*elementsPerAxis[0] + 1) * ((this->Nn1d - 1)*elementsPerAxis[1] + 1) * ((this->Nn1d - 1)*elementsPerAxis[2] + 1))) {
		std::cout << "Nodes does not match: \n" << "Elems: " << numElems << "\nNodes: " << nNodes << std::endl;
	}

	this->initializeElementNodeSurfaceMap(); // link nodes of an element to face of the element
	this->initializeBoundaryNodes(); // Create a mapping between global node number and index in global matrices and locating dirichlet nodes
	this->initializeElementMatrices(1); // Initialize the element matrices assuming we are in the first layer

	// Initialize matrices so that we don't have to resize them later
	this->FirrElem = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->FirrMat = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes);
	this->FirrMat.reserve(Eigen::VectorXi::Constant(nNodes, pow((this->Nn1d * 2 - 1), 3))); // at most 27 non-zero entries per column
	// TODO: make these three vectors sparse because they will only be non zero on the boundary nodes
	this->Fconv = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->Fk = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->Fq = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());

	// M and K will be sparse matrices because nodes are shared by relatively few elements
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

		this->ind2sub(e, this->elementsPerAxis, eSub);
		if ((eSub[2] >= this->elemsInLayer) && !layerFlag) {
			// we currently assume there will only be one element height change and once the transition has happened, 
			// we won't encounter any elements that have the original height. Therefore, we just reset J and all the elemental 
			// matrices once and we are good to go. 
			this->initializeElementMatrices(2);
			layerFlag = true;
		}

		// iterate over the nodes in the element. 
		for (int Ai = 0; Ai < Nne; Ai++) {
			// convert node index to a subscript
			this->ind2sub(Ai, elemNodeSize, AiSub);
			for (int ii = 0; ii < 3; ii++) {
				// get the globalNodeSubscript from local node subscript
				globalNodeSub[ii] = eSub[ii] * (this->Nn1d - 1) + AiSub[ii];
			}
			// get global node index
			globalNodeIdx = globalNodeSub[0] + globalNodeSub[1] * this->nodesPerAxis[0] + globalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];

			// determine if the node lies on a face of the Mesh
			nodeFace = this->determineNodeFace(globalNodeIdx);
			// this is the row of the global matrix associated with this local node
			matrixInd[0] = this->nodeMap[globalNodeIdx];
			if (matrixInd[0] >= 0) { // Verify that the node we are working with is not a dirichlet node.
				
				// Determine if the node lies on a boundary and then determine what kind of boundary
				if (nodeFace > 0) { // This check saves a lot time since most nodes are not on a surface.
					for (int f = 0; f < 6; f++) { // Iterate through each face of the element
						if ((nodeFace >> f) & 1) { // Node lies on face f
							if ((this->boundaryType[f] == FLUX)) { // heatFlux boundary
								this->Fq(matrixInd[0]) += this->FeQ(Ai, f);
							}
							else if (this->boundaryType[f] == CONVECTION) { // Convection Boundary
								// add componenet due to ambient temperature 
								this->Fconv(matrixInd[0]) += this->FeConv(Ai, f);

								// iterate again over nodes in the element, but only nodes on face 'f'
								for (int Bi : this->elemNodeSurfaceMap[f]) {
									this->ind2sub(Bi, elemNodeSize, BiSub);
									int BglobalNodeSub[3] = { eSub[0] * (this->Nn1d - 1) + BiSub[0], eSub[1] * (this->Nn1d - 1) + BiSub[1], eSub[2] * (this->Nn1d - 1) + BiSub[2] };
									BglobalNodeIdx = BglobalNodeSub[0] + BglobalNodeSub[1] * this->nodesPerAxis[0] + BglobalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];

									// this is the column in the global matrix associated with the local node Bi
									matrixInd[1] = this->nodeMap[BglobalNodeIdx];
									if (matrixInd[1] >= 0) {
										this->Kconv.coeffRef(matrixInd[0], matrixInd[1]) += this->KeConv[f](Ai, Bi);
										//Ktriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->KeConv[f](Ai, Bi)));
									}
									else {
										this->Fconv(matrixInd[0]) += -this->KeConv[f](Ai, Bi) * this->Temp(BglobalNodeIdx);
									}
								} // iterate through nodes
							} // if convection boundary
						} // if Node is face f
					} // iterate through faces
				} // if node is a face

				// for all nodes regardless of whether it is on a surface
				for (int Bi = 0; Bi < Nne; Bi++) { // iterate again over local nodes
					// this loop gets the effect of node Bi on node Ai. 

					this->ind2sub(Bi, elemNodeSize, BiSub);
					int BglobalNodeSub[3] = { eSub[0] * (this->Nn1d - 1) + BiSub[0], eSub[1] * (this->Nn1d - 1) + BiSub[1], eSub[2] * (this->Nn1d - 1) + BiSub[2] };
					BglobalNodeIdx = BglobalNodeSub[0] + BglobalNodeSub[1] * this->nodesPerAxis[0] + BglobalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];

					// get column in global matrix associated with local node Bi
					matrixInd[1] = this->nodeMap[BglobalNodeIdx];
					// Apply Fluence Rate regardless of what type of node B is 
					if (elemNFR) {// element-wise FluenceRate so we assume each node on the element has FluenceRate
						this->FirrElem(matrixInd[0]) += this->FeIrr(Ai, Bi) * this->FluenceRate(e);
					}
					else {
						this->FirrMat.coeffRef(matrixInd[0], BglobalNodeIdx) += this->FeIrr(Ai, Bi); // update matrix which will then multiplied by mua and fluenceRate
					}

					// Check if B is a dirichlet node for convection and thermal mass components
					if (matrixInd[1] >= 0) { // Ai and Bi are both valid positions so we add it to Kint and M and FirrElem
						this->Kint.coeffRef(matrixInd[0], matrixInd[1]) += this->KeInt(Ai, Bi);
						//Ktriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->KeInt(Ai, Bi)));
						this->M.coeffRef(matrixInd[0], matrixInd[1]) += this->Me(Ai, Bi);
						//Mtriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->Me(Ai, Bi)));
					}
					else if (matrixInd[1] < 0) { // valid row, but column is dirichlet node so we add to FirrElem... could be an if - else
						this->Fk(matrixInd[0]) += -this->KeInt(Ai, Bi) * this->Temp(BglobalNodeIdx);
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
	this->FirrMat.makeCompressed();

	if (!this->silentMode) {
		auto stopTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "Built the Matrices: " << duration.count() / 1000000.0 << std::endl;
	}
}

void FEM_Simulator::createFirr()
{
	/* This function only recreates the global FirrElem matrix. For when we change the fluence rate, but nothing else*/
	int nodeFace;
	int matrixInd[2];
	int elemNodeSize[3] = { this->Nn1d, this->Nn1d, this->Nn1d };
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
		std::cout << "Creating Firr Matrices" << std::endl;
	}

	int numElems = this->elementsPerAxis[0] * this->elementsPerAxis[1] * this->elementsPerAxis[2];
	int nNodes = this->nodesPerAxis[0] * this->nodesPerAxis[1] * this->nodesPerAxis[2];
	if (nNodes != (((this->Nn1d - 1) * elementsPerAxis[0] + 1) * ((this->Nn1d - 1) * elementsPerAxis[1] + 1) * ((this->Nn1d - 1) * elementsPerAxis[2] + 1))) {
		std::cout << "Nodes does not match: \n" << "Elems: " << numElems << "\nNodes: " << nNodes << std::endl;
	}

	this->initializeElementNodeSurfaceMap(); // link nodes of an element to face of the element
	this->initializeBoundaryNodes(); // Create a mapping between global node number and index in global matrices and locating dirichlet nodes
	this->initializeElementMatrices(1); // Initialize the element matrices assuming we are in the first layer
	int Nne = pow(this->Nn1d, 3); // number of nodes in an element is equal to the number of nodes in a single dimension cubed
	// Initialize matrices so that we don't have to resize them later
	this->FirrElem = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->FirrMat = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes);
	this->FirrMat.reserve(Eigen::VectorXi::Constant(nNodes, pow((this->Nn1d * 2 - 1), 3))); // at most 27 non-zero entries per column

	bool layerFlag = false;
	for (int e = 0; e < numElems; e++) {
		// When adding variable voxel height, we are preserving the nodal layout. The number of nodes in each x,y, and z axis is unchanged.
		// The only thing we are asssuming to change is the length along the z axis for elements after a certain threshold.

		this->ind2sub(e, this->elementsPerAxis, eSub);
		if ((eSub[2] >= this->elemsInLayer) && !layerFlag) {
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
			globalNodeIdx = globalNodeSub[0] + globalNodeSub[1] * this->nodesPerAxis[0] + globalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];

			nodeFace = this->determineNodeFace(globalNodeIdx);
			matrixInd[0] = this->nodeMap[globalNodeIdx];
			if (matrixInd[0] >= 0) { // Verify that the node we are working with is not a dirichlet node.
				for (int Bi = 0; Bi < Nne; Bi++) {
					this->ind2sub(Bi, elemNodeSize, BiSub);
					int BglobalNodeSub[3] = { eSub[0] * (this->Nn1d - 1) + BiSub[0], eSub[1] * (this->Nn1d - 1) + BiSub[1], eSub[2] * (this->Nn1d - 1) + BiSub[2] };
					BglobalNodeIdx = BglobalNodeSub[0] + BglobalNodeSub[1] * this->nodesPerAxis[0] + BglobalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];

					if (elemNFR) {// element-wise FluenceRate so we assume each node on the element has FluenceRate
						this->FirrElem(matrixInd[0]) += this->FeIrr(Ai, Bi) * this->FluenceRate(e);
					}
					else {//nodal FluenceRate so use as given
						this->FirrMat.coeffRef(matrixInd[0], BglobalNodeIdx) += this->FeIrr(Ai, Bi); // update matrix which will then multiplied by mua and fluenceRate
					}
				} // For loop through Bi
			} // If our node is not a dirichlet node
		} // For loop through Ai
	}
	if (!this->silentMode) {
		auto stopTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << "Built the Firr Matrix: " << duration.count() / 1000000.0 << std::endl;
	}
}

void FEM_Simulator::applyParameters()
{
	/* Here we assume constant tissue properties throughout the mesh. This allows us to
	multiply by tissue specific properties after the element construction, which means we can change
	tissue properties without having to reconstruct the matrices
	*/
	// Apply parameter specific multiplication for each global matrix.
	this->globK = this->Kint * this->TC + this->Kconv * this->HTC;
	this->globM = this->M * this->VHC; // M Doesn't have any additions so we just multiply it by the constant
	Eigen::VectorXf Firr;
	if (elemNFR) { // Using Element based NFR so we just use assignment
		Firr = this->FirrElem;
	}
	else { // to calculate Firr we post-multiply FirrMat by nodal irradiance
		Firr = this->FirrMat * this->FluenceRate;
	}
	this->globF = this->MUA * this->FirrMat * this->FluenceRate + this->Fconv * this->HTC + this->Fq + this->Fk * this->TC;
}

void FEM_Simulator::initializeModel()
{ 
	/* initializeModel gets the system ready to perform time stepping. This function needs to be called whenever
	the geometry of the tissue changes. This includes things like changing the number of nodes, changing the boundary
	conditions, changing the layers in the mesh, etc. This function also needs to be called if alpha or the time step 
	changes. 
	
	This function does not need to be called if we are only changing the irradiance, or the value of tissue properties
	like */
	this->createKMF();
	this->fluenceUpdate = false;

	auto startTime = std::chrono::high_resolution_clock::now();
	this->applyParameters();
	this->parameterUpdate = false;

	int nNodes = this->nodesPerAxis[0] * this->nodesPerAxis[1] * this->nodesPerAxis[2];
	/* PERFORMING TIME INTEGRATION USING EULER FAMILY */
	// Initialize d, v, and dTilde vectors
	this->dVec.resize(nNodes - dirichletNodes.size());
	this->vVec.resize(nNodes - dirichletNodes.size());
	
	// d vector gets initialized to what is stored in our Temp vector, ignoring Dirichlet Nodes
	this->dVec = this->Temp(validNodes);

	if (this->alpha < 1) { 
		// Perform the conjugate gradiant to compute the initial vVec value
		// This is a bit odd because if the user hasn't specified the initial fluence rate it will be 0 initially. 
		// And can mess up the first few timesteps
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> initSolver;
		initSolver.compute(this->globM);
		Eigen::VectorXf RHSinit = (this->globF) - this->globK * this->dVec;
		vVec = initSolver.solve(RHSinit);
	} // if we are using backwards Euler we can skip this initial computation of vVec. It is only
	// needed for explicit steps. 

	startTime = this->printDuration("Time Stepping Initialized: ", startTime);

	// Prepare solver for future iterations
	this->LHS = this->globM + this->alpha * this->deltaT * this->globK;
	this->LHS.makeCompressed();
	// These two steps form the cgSolver.compute() function. By calling them separately, we 
	// only ever need to call factorize when the tissue properties change.
	this->cgSolver.analyzePattern(this->LHS);
	this->cgSolver.factorize(this->LHS);
	if (this->cgSolver.info() != Eigen::Success) {
		std::cout << "Decomposition Failed" << std::endl;
	}
	startTime = this->printDuration("Initial Matrix Factorization Complete: ", startTime);
	// We are now ready to call single step
}

void FEM_Simulator::initializeSensorTemps(float duration) {
	// Create sensorTemperature Vectors and reserve sizes
	int nSensors = this->tempSensorLocations.size();
	if (nSensors == 0) { // if there weren't any sensors added, put one at 0,0,0
		nSensors = 1;
		this->tempSensorLocations.push_back({ 0,0,0 });
	}
	this->sensorTemps.resize(nSensors);
	for (int s = 0; s < nSensors; s++) {
		this->sensorTemps[s].resize(round(duration / deltaT) + 1);
	}
}

void FEM_Simulator::updateTemperatureSensors(int timeIdx) {
	/*TODO THIS DOES NOT WORK IF WE AREN'T USING LINEAR BASIS FUNCTIONS */
	int nSensors = this->tempSensorLocations.size();
	int Nne = pow(this->Nn1d, 3);
	// Input time = 0 information into temperature sensors
	//spacingLayer contains the distance between nodes in eacy layer.
	for (int s = 0; s < nSensors; s++) {
		std::array<float,3> sensorLocation = this->tempSensorLocations[s];
		// Determine the element (which for 1D linear elements is equivalent to the starting global node)
		// as well as the location in the bi-unit domain of that element, xi
		float xi[3];
		std::array<int, 3> elementLocation = positionToElement(sensorLocation,xi);
		std::array<int, 3> globalNodeStart;
		// For linear elements, there will be no change between globalNodeStart and elementLocation
		for (int i = 0; i < 3; i++) {
			globalNodeStart[i] = elementLocation[i] * (this->Nn1d - 1);
		}
		float tempValue = 0;	
		for (int Ai = 0; Ai < Nne; Ai++) { // iterate through each node in the element
			// adjust the global starting node based on the current node we should be visiting
			int globalNodeSub[3] = { globalNodeStart[0] + (Ai&1),globalNodeStart[1]+((Ai & 2) >> 1), globalNodeStart[2]+ ((Ai & 4) >> 2) };
			// convert subscript to index
			int globalNode = globalNodeSub[0] + globalNodeSub[1] * this->nodesPerAxis[0] + globalNodeSub[2] * this->nodesPerAxis[0] * this->nodesPerAxis[1];
			// add temperature contribution of node
			if (this->nodeMap[globalNode] >= 0) {//non-dirichlet node
				tempValue += this->calculateNA(xi, Ai) * this->dVec(this->nodeMap[globalNode]);
			}
			else { // dirichlet node
				tempValue += this->calculateNA(xi, Ai) * this->Temp(globalNode);
			}
			
		}
		this->sensorTemps[s][timeIdx] = tempValue;
	}

}

std::array<int, 3> FEM_Simulator::positionToElement(std::array<float, 3>& position, float xi[3]) {
	// Input should be a location where we want to place a sensor 
	float xSpacing = this->tissueSize[0] / float(this->elementsPerAxis[0]);
	float ySpacing = this->tissueSize[1] / float(this->elementsPerAxis[1]);
	float zSpacing = this->layerHeight / float(this->elemsInLayer);
	float zSpacing2 = (this->tissueSize[2] - this->layerHeight) / float(this->elementsPerAxis[2] - this->elemsInLayer);

	std::array<int, 3> elementLocation = { static_cast<int>(floor((position[0] + this->tissueSize[0] / 2.0f) / xSpacing)),
											static_cast<int>(floor((position[1] + this->tissueSize[1] / 2.0f) / ySpacing)),
											static_cast<int>(floor(position[2] / zSpacing)) };

	if (position[2] > this->layerHeight) {
		// This should compensate for the change in layer appropriately.
		elementLocation[2] = this->elemsInLayer + floor((position[2] - this->layerHeight) / zSpacing2);
	}
	// If the location of the element is the largest in a singular axis then we need to subtract one
	// and use the previous element position. This will effect locations on exact boundaries, where 
	// the 'floor' operation above won't work. 
	if (elementLocation[0] == this->elementsPerAxis[0]) elementLocation[0] -= 1;
	if (elementLocation[1] == this->elementsPerAxis[1]) elementLocation[1] -= 1;
	if (elementLocation[2] == this->elementsPerAxis[2]) elementLocation[2] -= 1;

	// Xi should be  avalue between -1 and 1 along each axis relating the placement of the sensor in that element.
	// Note that this only works because we assume cuboid elements (i.e. opposite faces are parallel, and each face is a rectangle). 
	// It should work for linear or quadratic basis functions given our assumptions. 
	xi[0] = -1 + ((position[0] + this->tissueSize[0] / 2.0f) / xSpacing - elementLocation[0]) * 2;
	xi[1] = -1 + ((position[1] + this->tissueSize[1] / 2.0f) / ySpacing - elementLocation[1]) * 2;
	xi[2] = -1 + ((position[2]) / zSpacing - elementLocation[2]) * 2;
	if (position[2] > this->layerHeight) {
		xi[2] = -1 + (this->elemsInLayer + (position[2] - this->layerHeight) / zSpacing2 - elementLocation[2]) * 2;
	}

	return elementLocation;

}

float FEM_Simulator::calculateNA(float xi[3], int Ai)
{
	/* Calculate the shape function output for given position in the element. */
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
	/* This function forms the building block for the derivative of the full shape function in 3D.
	This function assumes cuboid elements and works for linear or quadratic shape functions. */
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

Eigen::Vector3f FEM_Simulator::calculateNA_dot(float xi[3], int Ai)
{
	/* Calculate the derivative of the shape function with respect to all 3 axis. The result is a 3x1 vector. 
	*/
	Eigen::Vector3f NA_dot;
	int AiVec[3];
	int size[3] = { this->Nn1d,this->Nn1d,this->Nn1d };
	this->ind2sub(Ai, size, AiVec);
	NA_dot(0) = this->calculateNADotBase(xi[0], AiVec[0]) * this->calculateNABase(xi[1], AiVec[1]) * this->calculateNABase(xi[2], AiVec[2]);
	NA_dot(1) = this->calculateNADotBase(xi[1], AiVec[1]) * this->calculateNABase(xi[0], AiVec[0]) * this->calculateNABase(xi[2], AiVec[2]);
	NA_dot(2) = this->calculateNADotBase(xi[2], AiVec[2]) * this->calculateNABase(xi[0], AiVec[0]) * this->calculateNABase(xi[1], AiVec[1]);
	return NA_dot;
}

Eigen::Matrix3f FEM_Simulator::calculateJ(int layer)
{
	/* Builds the Jacobian to relate changes in the bi-unit domain to changes in the cartesian position. 
	This function takes advantage of the assumption that each element is a cuboid, and there aren't any orientation
	differences between the cartesian reference frame and bi-unit reference frame
	*/
	Eigen::Matrix3f J;
	//**** ASSUMING X-Y VOXEL SIZE IS CONSTANT THROUGHOUT VOLUME **********
	// we assume the z height can change once in the volume. 
	float deltaX = this->tissueSize[0] / float(this->elementsPerAxis[0]);
	float deltaY = this->tissueSize[1] / float(this->elementsPerAxis[1]);
	float deltaZ = 0;
	if (layer == 1) {
		deltaZ = this->layerHeight / float(this->elemsInLayer);
	}
	else if (layer == 2) {
		deltaZ = (this->tissueSize[2] - this->layerHeight) / float(this->elementsPerAxis[2] - this->elemsInLayer);
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

Eigen::Matrix2f FEM_Simulator::calculateJs(int dim, int layer)
{

	/* Builds the surface Jacobian to relate changes in the bi-unit domain to changes in the cartesian position.
	This function takes advantage of the assumption that each element is a cuboid, and there aren't any orientation
	differences between the cartesian reference frame and bi-unit reference frame
	*/

	// dim should be +-{1,2,3}. The dimension indiciates the axis of the normal vector of the plane. 
	// +1 is equivalent to (1,0,0) normal vector. -3 is equivalent to (0,0,-1) normal vector. 
	// We assume the values of xi correspond to the values of the remaining two axis in ascending order.
	// If dim = 2, then xi[0] is for the x-axis and xi[1] is for the z axis. 

	//**** ASSUMING X-Y VOXEL SIZE IS CONSTANT THROUGHOUT VOLUME **********
	// we assume the z height can change once in the volume. 
	float deltaX = this->tissueSize[0] / float(this->elementsPerAxis[0]);
	float deltaY = this->tissueSize[1] / float(this->elementsPerAxis[1]);
	float deltaZ = 0;
	if (layer == 1) {
		deltaZ = this->layerHeight / float(this->elemsInLayer);
	}
	else if (layer == 2) {
		deltaZ = (this->tissueSize[2] - this->layerHeight) / float(this->elementsPerAxis[2] - this->elemsInLayer);
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
	/* Perform Gaussian Quadrature numerical integration. 
	points: the number of points used for the integration
	dim: determines if the integration is over all 3 dimensions or just 2
	param1: passed into the function to integrate
	param2: passed into the function to integrate 
	*/
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
				float xi[3] = { static_cast<float>(dim / abs(dim)), zeros[i], zeros[j] };
				output += (this->*func)(xi, param1, param2) * weights[i] * weights[j];
			}
			else if (abs(dim) == 2) { // we are in the x-z plane
				float xi[3] = { zeros[i], static_cast<float>(dim / abs(dim)), zeros[j] };
				output += (this->*func)(xi, param1, param2) * weights[i] * weights[j];
			}
			else if (abs(dim) == 3) { // we are in the x-y plane
				float xi[3] = { zeros[i], zeros[j], static_cast<float>(dim / abs(dim)) };
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

	float deltaX = this->tissueSize[0] / float(this->elementsPerAxis[0]);
	float deltaY = this->tissueSize[1] / float(this->elementsPerAxis[1]);
	float deltaZ = this->tissueSize[2] / float(this->elementsPerAxis[2]);
	int sub[3];
	this->ind2sub(globalNode, this->nodesPerAxis, sub);
	position[0] = sub[0] * deltaX;
	position[1] = sub[1] * deltaY;
	position[2] = sub[2] * deltaZ;
}

float FEM_Simulator::calcKintAB(float xi[3], int Ai, int Bi)
{
	/* Calculates the value of function within the integral of the weak form when building the local
	matrices. This function will get integrated to build the local matrices
	*/
	float KABfunc = 0;
	Eigen::Vector3f NAdotA;
	Eigen::Vector3f NAdotB;
	Eigen::Matrix3f J = this->J;
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
	/* Calculates the value of function within the integral of the weak form when building the local
	matrices. This function will get integrated to build the local matrices
	*/
	float MABfunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3f J = this->J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	// The 1 should be replaced with this->VHC if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	MABfunc = (NAa * NAb) * J.determinant() * 1; // matrix math
	return MABfunc;
}

float FEM_Simulator::calcFintAB(float xi[3], int Ai, int Bi)
{
	/* Calculates the value of function within the integral of the weak form when building the local
	matrices. This function will get integrated to build the local matrices
	*/
	float FintFunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3f J = this->J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	// The 1 should be replaced with this->MUA if we change how the elemental matrices are computed
	// Right now we are computing matrices as parameter agnostic and then multiplying by the parameter
	FintFunc = 1 * (NAa * NAb) * J.determinant();
	// Output of this still needs to get multiplied by the FluenceRate at node Bi
	return FintFunc;
}

float FEM_Simulator::calcFqA(float xi[3], int Ai, int dim)
{
	/* Calculates the value of function within the integral of the weak form when building the local
	matrices. This function will get integrated to build the local matrices
	*/
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
	FjFunc = (NAa * this->heatFlux) * Js.determinant();
	return FjFunc;
}

float FEM_Simulator::calcFconvA(float xi[3], int Ai, int dim)
{
	/* Calculates the value of function within the integral of the weak form when building the local
	matrices. This function will get integrated to build the local matrices
	*/
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
	/* Calculates the value of function within the integral of the weak form when building the local
	matrices. This function will get integrated to build the local matrices
	*/
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
	/* This function has 3 tasks. 
	1. Locate all nodes that a dirichlet nodes and put them into the diricheltNodes vector.
	2. All nodes that are not dirichlet nodes are added to the validNodes vector
	3. We map create a mapping between global node indicies and the index into the global K/M/F matrices. 
	*/
	// we only need to scan nodes on the surface. Since we are assuming a cuboid this is easy to predetermine
	// Create mapping
	this->validNodes.clear();
	this->dirichletNodes.clear();

	int nNodes = this->nodesPerAxis[0] * this->nodesPerAxis[1] * this->nodesPerAxis[2];
	int positionCounter = 0;
	bool validNode = true;
	this->nodeMap = std::vector<int>(nNodes, -1); // initialize the mapping to -1. -1 indicates the node passed in is a dirichlet node.
	for (int i = 0; i < nNodes; i++) {
		int nodeFace = this->determineNodeFace(i);
		int nodeSub[3];
		this->ind2sub(i, this->nodesPerAxis, nodeSub);
		validNode = true;
		// Determine if the node lies on a boundary and then determine what kind of boundary
		if (nodeFace != 0) { // This check saves a lot time since most nodes are not on a surface.
			for (int f = 0; f < 6; f++) {
				if ((nodeFace >> f) & 1) { // Node lies on face f
					if (this->boundaryType[f] == HEATSINK) { // heatFlux boundary
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
	this->ind2sub(globalNode, this->nodesPerAxis, nodeSub);
	if (nodeSub[2] == 0) { 
		// Nodes on the top of the tissue have z-element = 0
		output += TOP;
	}
	if (nodeSub[2] == (this->nodesPerAxis[2] - 1)) {
		// Nodes on the bottom of the tissue have z-element = nodesPerAxis[2]-1
		output += BOTTOM;
	}
	if (nodeSub[0] == (this->nodesPerAxis[0] - 1)) {
		// Nodes on the front of the tissue have x-element = nodesPerAxis[0] - 1
		output += FRONT;
	}
	if (nodeSub[1] == (this->nodesPerAxis[0] - 1)) {
		// Nodes on the right of the tissue have y-element = nodesPerAxis[1] - 1
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
	int elementsPerAxis[3]; 
	if (((Temp.size() - 1) % (this->Nn1d - 1) != 0)|| ((Temp[0].size() - 1) % (this->Nn1d - 1) != 0) || ((Temp[0][0].size() - 1) % (this->Nn1d - 1) != 0)) {
		std::cout << "Invalid Node dimensions given the number of nodes in a single elemental axis" << std::endl;
	}
	elementsPerAxis[0] = (Temp.size() - 1) / (this->Nn1d - 1); // Temp contains the temperature at the nodes, so we need to subtract 1 to get the elements
	elementsPerAxis[1] = (Temp[0].size() - 1) / (this->Nn1d - 1);
	elementsPerAxis[2] = (Temp[0][0].size() - 1) / (this->Nn1d - 1);
	this->setElementsPerAxis(elementsPerAxis);
}

void FEM_Simulator::setTemp(Eigen::VectorXf &Temp)
{	
	//TODO make sure Temp is the correct size
	if (this->nodesPerAxis[0] * this->nodesPerAxis[1] * this->nodesPerAxis[2] != Temp.size()) {
		throw std::runtime_error("Total number of elements in Temp does not match current Node size.");
	}
	this->Temp = Temp;
}

std::vector<std::vector<std::vector<float>>> FEM_Simulator::getTemp()
{
	std::vector<std::vector<std::vector<float>>>
		TempOut(this->nodesPerAxis[0], std::vector<std::vector<float>>(this->nodesPerAxis[1], std::vector<float>(this->nodesPerAxis[2])));

	for (int i = 0; i < this->nodesPerAxis[0]; i++) {
		for (int j = 0; j < this->nodesPerAxis[1]; j++) {
			for (int k = 0; k < this->nodesPerAxis[2]; k++) {
				TempOut[i][j][k] = Temp(i + j * (this->nodesPerAxis[0]) + k * (this->nodesPerAxis[0] * this->nodesPerAxis[1]));
			}
		}
	}

	return TempOut;
}

void FEM_Simulator::setFluenceRate(std::vector<std::vector<std::vector<float>>> FluenceRate)
{
	this->FluenceRate = Eigen::VectorXf::Zero(FluenceRate.size() * FluenceRate[0].size() * FluenceRate[0][0].size());
	// Convert nested vectors into a single column Eigen Vector. 
	for (int i = 0; i < FluenceRate.size(); i++) // associated with x and is columns of matlab matrix
	{
		for (int j = 0; j < FluenceRate[0].size(); j++) // associated with y and is rows of matlab matrix
		{
			for (int k = 0; k < FluenceRate[0][0].size(); k++) // associated with z and is depth of matlab matrix
			{
				this->FluenceRate(i + j * FluenceRate.size() + k * FluenceRate.size() * FluenceRate[0].size()) = FluenceRate[i][j][k];
			}
		}
	}

	if ((FluenceRate.size() == this->elementsPerAxis[0]) && (FluenceRate[0].size() == this->elementsPerAxis[1]) && (FluenceRate[0][0].size() == this->elementsPerAxis[2])) {
		this->elemNFR = true;
	}
	else if ((FluenceRate.size() == this->nodesPerAxis[0]) && (FluenceRate[0].size() == this->nodesPerAxis[1]) && (FluenceRate[0][0].size() == this->nodesPerAxis[2])) {
		this->elemNFR = false;
	}
	else {
		std::cout << "NFR must have the same number of entries as the node space or element space" << std::endl;
		throw std::invalid_argument("NFR must have the same number of entries as the node space or element space");
	}

	this->fluenceUpdate = true;
}

void FEM_Simulator::setFluenceRate(Eigen::VectorXf& FluenceRate)
{
	this->FluenceRate = FluenceRate;
	//TODO Check for element or nodal FluenceRate;
	this->elemNFR = false;
	this->fluenceUpdate = true;
}

void FEM_Simulator::setFluenceRate(float laserPose[6], float laserPower, float beamWaist)
{
	this->FluenceRate = Eigen::VectorXf::Zero(this->nodesPerAxis[0]* this->nodesPerAxis[1]* this->nodesPerAxis[2]);
	float lambda = 10.6 * pow(10, -4); // wavelength of laser in cm
	// ASSUMING THERE IS NO ORIENTATION SHIFT ON THE LASER
	//TODO: account for orientation shift on the laser
	// I(x,y,z) = 2*P/(pi*w^2) * exp(-2*(x^2 + y^2)/w^2 - mua*z)

	float irr = 0;
	float width = 0; 
	float xPos = -this->tissueSize[0] / 2;
	float xStep = this->tissueSize[0] / this->elementsPerAxis[0];
	float yPos = -this->tissueSize[1] / 2;
	float yStep = this->tissueSize[1] / this->elementsPerAxis[1];
	float zPos = 0;
	float zStep = this->layerHeight / this->elemsInLayer;

	for (int i = 0; i < this->nodesPerAxis[0]; i++) {
		yPos = -this->tissueSize[1] / 2;
		for (int j = 0; j < this->nodesPerAxis[1]; j++) {
			zPos = 0;
			zStep = this->layerHeight / this->elemsInLayer;
			for (int k = 0; k < this->nodesPerAxis[2]; k++) {
				if (k >= this->elemsInLayer) {
					// if we have passed the layer size
					zStep = (tissueSize[2] - this->layerHeight) / (this->elementsPerAxis[2] - this->elemsInLayer);
				}
				// calculate beam width at depth
				width = beamWaist * std::sqrt(1 + pow((lambda * (zPos + laserPose[2]) / (std::acos(-1) * pow(beamWaist, 2))), 2));
				// calculate laser irradiance
				irr = 2 * laserPower / (std::acos(-1) * pow(width, 2)) * std::exp(-2 * (pow((xPos - laserPose[0]), 2) + pow((yPos - laserPose[1]), 2)) / pow(width,2) - this->MUA * zPos);
				// set laser irradiance
				this->FluenceRate(i + j * this->nodesPerAxis[0] + k * this->nodesPerAxis[0] * this->nodesPerAxis[1]) = irr;
				// increase z pos
				zPos = zPos + zStep;
			}
			// increase y pos
			yPos = yPos + yStep;
		}
		// increase x pos 
		xPos = xPos + xStep;
	}

	this->fluenceUpdate = true;
}

void FEM_Simulator::setTissueSize(float tissueSize[3]) {
	
	// if our initial layer configuration would become outside our tissue size
	// then we need to resize our layer configuration.
	if (this->layerHeight >= tissueSize[2]) {
		this->layerHeight = tissueSize[2];
		this->elemsInLayer = this->elementsPerAxis[2];
	}
	for (int i = 0; i < 3; i++) {
		this->tissueSize[i] = tissueSize[i];
	}
	
	this->setJ();
}

void FEM_Simulator::setLayer(float layerHeight, int elemsInLayer) {
	if ((elemsInLayer > this->elementsPerAxis[2])||(elemsInLayer < 0)) {
		std::cout << "Invalid layer size. The layer must be equal"
			<< " to or less than the total number of elements in the z direction and greater than 0" << std::endl;
		throw std::runtime_error("Layer Size must be equal to or less than the toal number of elements in the z direction and greater than 0");
	}
	if ((layerHeight < 0) || (layerHeight > this->tissueSize[2])) {
		std::cout << "Invalid layer height. The layer dimension must be less than or equal to the tissue size " 
			<< "and greater than zero" << std::endl;
		throw std::runtime_error("Invalid layer height. The layer dimension must be less than or equal to the tissue size and greater than zero");
	}
	if ((layerHeight == 0) != (elemsInLayer == 0)) {
		std::cout << "Layer Height must be 0 if layer size is 0 and vice versa" << std::endl;
		throw std::runtime_error("Layer Height must be 0 if layer size is 0 and vice versa");
	}
	if ((layerHeight == this->tissueSize[2]) != (elemsInLayer == this->elementsPerAxis[2])) {
		std::cout << "Layer Height must be the tissue height if layer size is the grid size and vice versa" << std::endl;
		throw std::runtime_error("Layer Height must be the tissue height if layer size is the grid size and vice versa");
	}
	this->layerHeight = layerHeight;
	this->elemsInLayer = elemsInLayer;
	this->setJ(1);
}

void FEM_Simulator::setTC(float TC) {
	this->TC = TC;
	this->parameterUpdate = true;
}

void FEM_Simulator::setVHC(float VHC) {
	this->VHC = VHC;
	this->parameterUpdate = true;
}

void FEM_Simulator::setMUA(float MUA) {
	this->MUA = MUA;
	this->parameterUpdate = true;
}

void FEM_Simulator::setHTC(float HTC) {
	this->HTC = HTC;
	this->parameterUpdate = true;
}

void FEM_Simulator::setFlux(float heatFlux)
{
	this->heatFlux = heatFlux;
}

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	this->ambientTemp = ambientTemp;
}

void FEM_Simulator::setElementsPerAxis(int elementsPerAxis[3]) {
	for (int i = 0; i < 3; i++) {
		this->elementsPerAxis[i] = elementsPerAxis[i];
		this->nodesPerAxis[i] = elementsPerAxis[i] * (this->Nn1d - 1) + 1;
	}
}

void FEM_Simulator::setNodesPerAxis(int nodesPerAxis[3]) {
	for (int i = 0; i < 3; i++) {
		this->elementsPerAxis[i] = nodesPerAxis[i] - 1;
		this->nodesPerAxis[i] = nodesPerAxis[i];
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

std::chrono::steady_clock::time_point FEM_Simulator::printDuration(const std::string& message, std::chrono::steady_clock::time_point startTime) {
	if (!this->silentMode) {
		auto stopTime = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
		std::cout << message << duration.count() / 1000000.0 << std::endl;
		startTime = stopTime;
		
	}
	return startTime;
}