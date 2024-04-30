#include "FEM_Simulator.h"
#include <iostream>

const int FEM_Simulator::A[8][3] = {{-1, -1, -1},{1,-1,-1},{-1,1,-1},{1,1,-1}, {-1,-1,1},{1,-1,1},{-1,1,1}, { 1,1,1 } };

FEM_Simulator::FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA, float HTC)
{
	this->setInitialTemperature(Temp);
	this->setTissueSize(tissueSize);
	this->setTC(TC);
	this->setVHC(VHC);
	this->setMUA(MUA);
	this->setHTC(HTC);
}

void FEM_Simulator::performTimeStepping()
{
	auto startTime = std::chrono::high_resolution_clock::now();
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
		this->sensorTemps[s].resize(ceil(this->tFinal / deltaT) + 1);
	}

	// Solve Euler Family 
	// Initialize d vector
	Eigen::VectorXf dVec(nNodes - dirichletNodes.size());
	Eigen::VectorXf vVec(nNodes - dirichletNodes.size());
	Eigen::VectorXf dTilde(nNodes - dirichletNodes.size());
	int counter = 0;
	for (int n : validNodes) {
			int nodeSub[3];
			ind2sub(n, this->nodeSize, nodeSub);
			dVec(counter) = this->Temp[nodeSub[0]][nodeSub[1]][nodeSub[2]];
			vVec(counter) = 0;
			counter++;
	}

	this->updateTemperatureSensors(0, dVec);
	
	auto stopTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
	std::cout << "D initialized: " << duration.count() / 1000000.0 << std::endl;
	startTime = stopTime;

	// Perform TimeStepping
	// Eigen documentation says using Lower|Upper gives the best performance for the solver with a full matrix. 
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> solver;
	Eigen::SparseMatrix<float> fullM = this->M + this->alpha * deltaT * this->K;
	solver.compute(fullM);
	if (solver.info() != Eigen::Success) {
		std::cout << "Decomposition Failed" << std::endl;
	}

	stopTime = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
	std::cout << "Matrix Factorized: " << duration.count() / 1000000.0 << std::endl;
	startTime = stopTime;

	for (float t = 1; t <= (this->tFinal/this->deltaT); t ++) { 
		dTilde = dVec + (1 - this->alpha) * this->deltaT * vVec;	
		Eigen::VectorXf fullF = this->F - this->K * dTilde;
		Eigen::VectorXf vVec2 = solver.solve(fullF);
		if (solver.info() != Eigen::Success) {
			std::cout << "Issue With Solver" << std::endl;
		}
		dVec = dVec + this->deltaT * (this->alpha * vVec2 + (1 - this->alpha) * vVec);

		this->updateTemperatureSensors(t, dVec);
		vVec = vVec2;
	}

	stopTime = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
	std::cout << "Time Stepping Completed: " << duration.count() / 1000000.0 << std::endl;
	startTime = stopTime;

	// Adjust our Temp with new d vector
	counter = 0;
	for (int n : validNodes) {
			int nodeSub[3];
			ind2sub(n, this->nodeSize, nodeSub);
			this->Temp[nodeSub[0]][nodeSub[1]][nodeSub[2]] = dVec(counter);
			counter++;
	}

	stopTime = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
	std::cout << "Updated Temp Variable: " << duration.count()/1000000.0 << std::endl;
	startTime = stopTime;
}

void FEM_Simulator::reduceSparseMatrix(Eigen::SparseMatrix<float> oldMat, std::vector<int> rowsToRemove, Eigen::SparseMatrix<float> *newMat, Eigen::SparseMatrix<float> *suppMat, int nNodes) {
	// Assuming the newMat is already the appropriate size, and has had its elements reserved. 
	int rowCounter = 0;
	for (int row = 0; row < nNodes; row++) {
		int validColCounter = 0;
		int invalidColCounter = 0;
		if (std::find(rowsToRemove.begin(), rowsToRemove.end(), row) == rowsToRemove.end()) {// if row is NOT in our rows to remove, iterate over the columns
			for (int col = 0; col < nNodes; col++) {
				if (std::find(rowsToRemove.begin(), rowsToRemove.end(), col) == rowsToRemove.end()) { // the col is a valid entry
					if (oldMat.coeff(row, col) != 0) { // The location in the sparse matrix is NOT a zero
						// We add the value from old matrix to new matrix
						newMat->insert(rowCounter, validColCounter) = oldMat.coeff(row, col);
					} 
					// else is just do nothing
					validColCounter++; // have to increment column counter so we maintain correct position in new matrix. 
				} // end if col is valid
				else { // the col is not a valid entry, but the row is so fill the support matrix
					if (oldMat.coeff(row, col) != 0) { // The location in the sparse matrix is NOT a zero
						suppMat->insert(rowCounter, invalidColCounter) = oldMat.coeff(row, col);
					}
					invalidColCounter++;
				}
			} // end for each column
			rowCounter++; // increment row counter in our new matrix
		} // end if row is not in our rows to remove
	} // end for each row
} //reduceSparseMatrix

void FEM_Simulator::createKMF() {
	
	auto startTime = std::chrono::high_resolution_clock::now();

	int numElems = this->gridSize[0] * this->gridSize[1] * this->gridSize[2];
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];

	this->initializeBoundaryNodes();

	// Initialize matrices so that we don't have to resize them later
	this->F = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->M = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->M.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), 27)); // at most 27 non-zero entries per column
	this->K = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->K.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), 27)); // at most 27 non-zero entries per column

	// iterate through the non-dirichlet nodes. Any dirichlet nodes don't get a row entry in the matrices/vectors
#ifdef _OPENMP
#pragma omp parallel for 
#endif
	for (int row = 0; row < this->validNodes.size(); row++) {
		int nodeSub[3];
		int globalNode = this->validNodes[row]; // get our global node
		int nodeFace = this->determineNodeFace(globalNode); // determine what faces our node is on
		this->ind2sub(globalNode, this->nodeSize, nodeSub);

		Eigen::Matrix<float, 1, 27> Kconv = Eigen::Matrix<float, 27, 1>::Constant(0.0f); // The result the convection boundary has on K(row,col)
		// Handle Flux and Convection Boundaries 
		if (nodeFace > 0) { // This check saves a lot time since most nodes are not on a surface.
			for (int f = 0; f < 6; f++) { // Iterate through each face of the element
				if ((nodeFace >> f) & 1) { // Node lies on face f
					std::vector<int> localNodes = this->convertToLocalNode(globalNode, f);
					for (int Ai : localNodes) { // iterate through the localNodes associated with the current global node
						if ((this->boundaryType[f] == FLUX)) { // flux boundary
							this->F(row) += this->Fj(Ai, f);
						}
						else if (this->boundaryType[f] == CONVECTION) { // Convection Boundary
							this->F(row) += this->Fv(Ai, f);
							for (int Bi : elemNodeSurfaceMap[f]) {

								int BiGlobal = this->convertToGlobalNode(Bi, globalNode, Ai); // Need some conversion from Ai,Bi,n to global position of Bi
								int BiNeighbor = this->convertToNeighborIdx(BiGlobal, globalNode); // Need some conversion from Ai, Bi to neighbor of n
								if (this->nodeMap[BiGlobal] >= 0) { // Bi is not a dirichlet node
									this->K.coeffRef(row, this->nodeMap[BiGlobal]) += this->Fvu[f](Ai, Bi);
								}
								else { // Bi is a dirichlet node
									this->F(row) += -this->Fvu[f](Ai, Bi);
								}
							}
						} // ENDIf Convection Boundary
					} // END iterate through the elements that contain our global node
				} // ENDIF Node is face f
			} // iterate through faces


			// Handle special cases of K, M, and F;
			// determine which elements in a 8 element box exist if we assume our node is position 13 in the 8 element box

			int eOpts = 0b11111111; //binary number where a 1 indicates a valid element, LSB order
			eOpts &= ((nodeSub[2] == 0) ? 0b11110000 : 0b11111111); // valid elements if we are at top layer: 4,5,6,7
			eOpts &= ((nodeSub[2] == (this->nodeSize[2] - 1)) ? 0b00001111 : 0b11111111); // valid elements if we are at bottom layer: 0,1,2,3
			eOpts &= ((nodeSub[1] == 0) ? 0b11001100 : 0b11111111); // valid elements if we are at front wall: 2,3,6,7
			eOpts &= ((nodeSub[1] == (this->nodeSize[1] - 1)) ? 0b00110011 : 0b11111111); // valid elements if we are at back wall: 0,1,4,5
			eOpts &= ((nodeSub[0] == 0) ? 0b10101010 : 0b11111111); // valid elements if we are at left wall: 1,3,5,7
			eOpts &= ((nodeSub[0] == (this->nodeSize[1] - 1)) ? 0b01010101 : 0b11111111); // valid elements if we are at right wall: 0,2,4,6
			for (int e = 0; e < 8; e++) { // iterate through possible elements
				if ((eOpts >> e) & 1) {// if valid elements
					int eSub[3] = { nodeSub[0],nodeSub[1],nodeSub[2] };
					eSub[0] = (((e % 2) == 0) ? nodeSub[0] - 1 : nodeSub[0]); // local element 0,2,4,6 requires we shift the x
					eSub[1] = ((((e / 2) % 2) == 0) ? nodeSub[1] - 1 : nodeSub[1]); // local element 0,1,4,5 requires we shift the y
					eSub[2] = (((e / 4) == 0) ? nodeSub[2] - 1 : nodeSub[2]); // local element 0,1,2,3 requires we shift the z

					int Ai = 7 - e; // Our current node's local index is just 7-e -- assuming our current node is node 13 (0-26) in an 8 element box
					for (int Bi = 0; Bi < 8; Bi++) {
						int BiGlobal = this->convertToGlobalNode(Bi, globalNode, Ai);
						int BiSub[3];
						ind2sub(BiGlobal, this->nodeSize, BiSub);
						if (this->nodeMap[BiGlobal] >= 0) {
							this->K.coeffRef(row, this->nodeMap[BiGlobal]) += this->Ke(Ai, Bi);
							this->M.coeffRef(row, this->nodeMap[BiGlobal]) += this->Me(Ai, Bi);
						}
						else
						{
							this->F(row) += this->Ke(Ai, Bi) * this->Temp[BiSub[0]][BiSub[1]][BiSub[2]];
						}
						if (this->elemNFR) {
							this->F(row) += this->FeInt(Ai, Bi) * this->NFR[eSub[0]][eSub[1]][eSub[2]];
						}
						else {
							this->F(row) += this->FeInt(Ai, Bi) * this->NFR[BiSub[0]][BiSub[1]][BiSub[2]];
						}
					}
				}
			}
		}
		else
		{// we know the node does not lie on a face so we can assume it is surrounded by 26 other nodes
			int idx = 0;
			int neighbor = 0;
			int BiSub[3];
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < 3; j++) {
					for (int i = 0; i < 3; i++) {
						idx = i + 3 * j + 9 * k;
						neighbor = globalNode + (k - 1) * this->nodeSize[0] * this->nodeSize[1] + (j - 1) * this->nodeSize[0] + (i - 1);
						ind2sub(neighbor, this->nodeSize, BiSub);
						if (this->nodeMap[neighbor] >= 0) { // - check if any of the neighbors are dirichlet still
							this->K.insert(row, this->nodeMap[neighbor]) = this->Kn(idx);
							this->M.insert(row, this->nodeMap[neighbor]) = this->Mn(idx);
						}
						else { // if dirichlet we add K to the force vector 
							this->F(row) += this->Kn(idx) * this->Temp[BiSub[0]][BiSub[1]][BiSub[2]];
						}
						if (this->elemNFR) {
							// for elemental NFR we need to average the elementNFR's that the neighbor
							// and global node are both in. It will either be 1 element, 2 elements, 4 elements, or 8 elements
							// We are assuming that the elemental NFR is the average/center NFR experienced by the element.
							float interpNFR = 0;
							int biShift = nodeSub[0] - BiSub[0]; // 0: same index, 1: BiSub is left, -1: BiSub is right
							int bjShift = nodeSub[1] - BiSub[1]; // 0: same index, 1: BiSub is forward, -1: BiSub is back
							int bkShift = nodeSub[2] - BiSub[2]; // 0: smae index, 1: BiSub is up, -1: Bi Sub is down
							int eShift[3] = { 1 - abs(biShift), 1 - abs(bjShift), 1 - abs(bkShift) };
							for (int ei = 0; ei <= eShift[0]; ei++) {
								for (int ej = 0; ej <= eShift[1]; ej++) {
									for (int ek = 0; ek <= eShift[2]; ek++) {
										// the division is for the linear interpolation, we can just divide because we know each elemental NFR will contribute
										// equally to the node in question and we know that each voxel has the same width/length/height.
										// Because of uniform voxel sizes, the Jacobian is the same regardless of the element, meaning we don't have to 
										// worry about additional weights on the terms
										int eSub[3] = { BiSub[0] - ei - (biShift < 0), BiSub[1] - ej - (bjShift < 0), BiSub[2] - ek - (bkShift < 0) };
										interpNFR += NFR[eSub[0]][eSub[1]][eSub[2]] * 1 / float(pow(2, eShift[0] + eShift[1] + eShift[2]));
									}
								}
							}
							// Remember that FnInt contains the contributions the node 'neighbor' in each element shared with our global node
							// if we were using nodal NFR, then the NFR at node neighbor is constant. With elemental NFR we calculate the NFR
							// at the neighbor by averaging the NFRs of the shared nodes. 
							this->F(row) += this->FnInt(idx) * interpNFR;
						}
						else {
							this->F(row) += this->FnInt(idx) * NFR[BiSub[0]][BiSub[1]][BiSub[2]];
						}
					}
				}
			}

		}
	}

	auto stopTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
	std::cout << "Built the Matrices: " << duration.count() / 1000000.0 << std::endl;
}

void FEM_Simulator::createKMFelem()
{
	auto startTime = std::chrono::high_resolution_clock::now();

	int numElems = this->gridSize[0] * this->gridSize[1] * this->gridSize[2];
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];

	this->initializeBoundaryNodes();

	// Initialize matrices so that we don't have to resize them later
	this->F = Eigen::VectorXf::Zero(nNodes - this->dirichletNodes.size());
	this->M = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->M.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), 27)); // at most 27 non-zero entries per column
	this->K = Eigen::SparseMatrix<float>(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size());
	this->K.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), 27)); // at most 27 non-zero entries per column
	//std::vector<Eigen::Triplet<float>> Ktriplets;
	//Ktriplets.reserve(numElems * 10 * 10);
	//std::vector<Eigen::Triplet<float>> Mtriplets;
	//Mtriplets.reserve(numElems * 8 * 8);

	for (int e = 0; e < numElems; e++) {
		this->currElement.elementNumber = e;

		int eSub[3];
		this->ind2sub(e, this->gridSize, eSub);
		int elementGlobalNodes[8]; //global nodes for element e
		this->getGlobalNodesFromElem(e, elementGlobalNodes);

		/* We are assuming a uniform cuboid so we don't need this
		for (int Ai = 0; Ai < 8; Ai++) {// x,y,z coordinates for each of the global nodes
			this->getGlobalPosition(elementGlobalNodes[Ai], this->currElement.globalNodePositions[Ai]);
		}
		*/

		int nodeFace;
		int matrixInd[2];
		for (int Ai = 0; Ai < 8; Ai++) {
			nodeFace = this->determineNodeFace(elementGlobalNodes[Ai]);
			matrixInd[0] = this->nodeMap[elementGlobalNodes[Ai]];
			if (matrixInd[0] >= 0) { // Verify that the node we are working with is not a dirichlet node.
				int AiSub[3];
				this->ind2sub(elementGlobalNodes[Ai], this->nodeSize, AiSub);

				// Determine if the node lies on a boundary and then determine what kind of boundary
				bool dirichletFlag = false;
				bool fluxFlag = false;
				if (nodeFace > 0) { // This check saves a lot time since most nodes are not on a surface.
					for (int f = 0; f < 6; f++) { // Iterate through each face of the element
						if ((nodeFace >> f) & 1) { // Node lies on face f
							if ((this->boundaryType[f] == FLUX)) { // flux boundary
								this->F(matrixInd[0]) += this->Fj(Ai, f);
							}
							else if (this->boundaryType[f] == CONVECTION) { // Convection Boundary
								this->F(matrixInd[0]) += this->Fv(Ai, f);
								for (int Bi : elemNodeSurfaceMap[f]) {
									matrixInd[1] = this->nodeMap[elementGlobalNodes[Bi]];
									if (matrixInd[1] >= 0) {
										int AiBi = Bi * 8 + Ai; // had to be creative here to encode Ai and Bi in a single variable. We are using base 8. 
										// So if Bi is 1 and Ai is 7, the value is 15. 15 in base 8 is 17. 
										this->K.coeffRef(matrixInd[0], matrixInd[1]) += this->Fvu[f](Ai, Bi);
										//Ktriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->Fvu[f](Ai, Bi)));
									}
									else {
										this->F(matrixInd[0]) += -this->Fvu[f](Ai, Bi);
									}
								}
							}
						} // if Node is face f
					} // iterate through faces
				} // if node is a face

				// Now we will build the K, M and F matrice
				int BiSub[3];
				for (int Bi = 0; Bi < 8; Bi++) {
					// Sparse Matrix can't be filled through slicing, so we have to add each element individually by iterating over Ai and Bi
					ind2sub(elementGlobalNodes[Bi], this->nodeSize, BiSub); // get our B value is a subscript

					matrixInd[1] = this->nodeMap[elementGlobalNodes[Bi]];
					if (matrixInd[1] >= 0) { // Ai and Bi are both valid positions so we add it to K and M and F
						this->K.coeffRef(matrixInd[0], matrixInd[1]) += this->Ke(Ai, Bi);
						//Ktriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->Ke(Ai, Bi)));
						this->M.coeffRef(matrixInd[0], matrixInd[1]) += this->Me(Ai, Bi);
						//Mtriplets.push_back(Eigen::Triplet<float>(matrixInd[0], matrixInd[1], this->Me(Ai, Bi)));
						if (elemNFR) {// element-wise NFR so we assume each node on the element has NFR
							this->F(matrixInd[0]) += this->FeInt(Ai, Bi) * this->NFR[eSub[0]][eSub[1]][eSub[2]];
						}
						else {//nodal NFR so use as given
							this->F(matrixInd[0]) += this->FeInt(Ai, Bi) * this->NFR[BiSub[0]][BiSub[1]][BiSub[2]];
						}
					}
					else if (matrixInd[1] < 0) { // valid row, but column is dirichlet node so we add to F... could be an if - else
						this->F(matrixInd[0]) += -this->Ke(Ai, Bi) * this->Temp[BiSub[0]][BiSub[1]][BiSub[2]];
						if (elemNFR) { // element-wise NFR so we assume each node on the element has NFR
							this->F(matrixInd[0]) += this->FeInt(Ai, Bi) * this->NFR[eSub[0]][eSub[1]][eSub[2]];
						}
						else {//nodal NFR so use as given
							this->F(matrixInd[0]) += this->FeInt(Ai, Bi) * this->NFR[BiSub[0]][BiSub[1]][BiSub[2]];
						}
					} // if both are invalid we ignore, if column is valid but row is invalid we ignore
				} // For loop through Bi
			} // If our node is not a dirichlet node
		} // For loop through Ai
	}

	this->K.makeCompressed();
	this->M.makeCompressed();
	auto stopTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - startTime);
	std::cout << "Built the Matrices: " << duration.count() / 1000000.0 << std::endl;
}

void FEM_Simulator::updateTemperatureSensors(int timeIdx, Eigen::VectorXf& dVec) {
	int nSensors = this->tempSensorLocations.size();
	// Input time = 0 information into temperature sensors
	float spacing[3] = { this->tissueSize[0] / float(this->gridSize[0]), this->tissueSize[1] / float(this->gridSize[1]) , this->tissueSize[2] / float(this->gridSize[2]) };
	for (int s = 0; s < nSensors; s++) {
		std::array<float,3> sensorLocation = this->tempSensorLocations[s];
		int globalNodeOptions[3][2] = { {floor((sensorLocation[0] + this->tissueSize[0] / 2.0f) / spacing[0]),
										ceil((sensorLocation[0] + this->tissueSize[0] / 2.0f) / spacing[0])},
			{floor((sensorLocation[1] + this->tissueSize[1] / 2.0f) / spacing[1]),
										ceil((sensorLocation[1] + this->tissueSize[1] / 2.0f) / spacing[1])},
			{floor(sensorLocation[2] / spacing[2]),
										ceil(sensorLocation[2] / spacing[0])} };
		float tempValue = 0;
		float xi[3];
		xi[0] = (sensorLocation[0] - (globalNodeOptions[0][0] * spacing[0] - this->tissueSize[0] / 2.0f)) * 2 / spacing[0] - 1;
		xi[1] = (sensorLocation[1] - (globalNodeOptions[1][0] * spacing[1] - this->tissueSize[1] / 2.0f)) * 2 / spacing[1] - 1;;
		xi[2] = (sensorLocation[2] - (globalNodeOptions[2][0] * spacing[2] - this->tissueSize[2] / 2.0f)) * 2 / spacing[2] - 1;;
		for (int Ai = 0; Ai < 8; Ai++) {
			int globalNodeSub[3] = { globalNodeOptions[0][Ai & 1],globalNodeOptions[1][(Ai & 2) >> 1], globalNodeOptions[2][(Ai & 4) >> 2] };
			int globalNode = globalNodeSub[0] + globalNodeSub[1] * this->nodeSize[0] + globalNodeSub[2] * this->nodeSize[0] * this->nodeSize[1];
			if (this->nodeMap[globalNode] >= 0) {//non-dirichlet node
				tempValue += this->calculateNA(xi, Ai) * dVec(this->nodeMap[globalNode]);
			}
			else { // dirichlet node
				tempValue += this->calculateNA(xi, Ai) * this->Temp[globalNodeSub[0]][globalNodeSub[1]][globalNodeSub[2]];
			}
			
		}
		this->sensorTemps[s][timeIdx] = tempValue;
	}

}

float FEM_Simulator::calculateNA(float xi[3], int Ai)
{
	float output = 0;
	int temp[3] = { FEM_Simulator::A[Ai][0],FEM_Simulator::A[Ai][1], FEM_Simulator::A[Ai][2] };
	output = 1 / 8.0 * (1 + xi[0] * temp[0]) * (1 + xi[1] * temp[1]) * (1 + xi[2] * temp[2]);
	return output;
}

Eigen::Matrix3<float> FEM_Simulator::calculateJ()
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
	//**** ASSUMING VOXEL SIZE IS CONSTANT THROUGHOUT VOLUME **********
	float deltaX = this->tissueSize[0] / float(this->gridSize[0]);
	float deltaY = this->tissueSize[1] / float(this->gridSize[1]);
	float deltaZ = this->tissueSize[2] / float(this->gridSize[2]);
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

Eigen::Matrix2<float> FEM_Simulator::calculateJs(int dim)
{
	// dim should be +-{1,2,3}. The dimension indiciates the axis of the normal vector of the plane. 
	// +1 is equivalent to (1,0,0) normal vector. -3 is equivalent to (0,0,-1) normal vector. 
	// We assume the values of xi correspond to the values of the remaining two axis in ascending order.
	// If dim = 2, then xi[0] is for the x-axis and xi[1] is for the z axis. 

	//**** ASSUMING VOXEL SIZE IS CONSTANT THROUGHOUT VOLUME **********
	float deltaX = this->tissueSize[0] / float(this->gridSize[0]);
	float deltaY = this->tissueSize[1] / float(this->gridSize[1]);
	float deltaZ = this->tissueSize[2] / float(this->gridSize[2]);
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

Eigen::Vector3<float> FEM_Simulator::calculateNA_dot(float xi[3], int Ai)
{
	Eigen::Vector3<float> NA_dot;
	NA_dot(0) = FEM_Simulator::calculateNA_xi(xi, Ai);
	NA_dot(1) = FEM_Simulator::calculateNA_eta(xi, Ai);
	NA_dot(2) = FEM_Simulator::calculateNA_zeta(xi, Ai);
	return NA_dot;
}

float FEM_Simulator::calculateNA_xi(float xi[3], int Ai)
{
	float output = 0;
	int temp[3] = { FEM_Simulator::A[Ai][0], FEM_Simulator::A[Ai][1], FEM_Simulator::A[Ai][2] };
	output = 1 / 8.0 * temp[0] * (1 + xi[1] * temp[1]) * (1 + xi[2] * temp[2]);
	return output;
}

float FEM_Simulator::calculateNA_eta(float xi[3], int Ai)
{
	float output = 0;
	int temp[3] = { FEM_Simulator::A[Ai][0], FEM_Simulator::A[Ai][1], FEM_Simulator::A[Ai][2] };
	output = 1 / 8.0 * temp[1] * (1 + xi[0] * temp[0]) * (1 + xi[2] * temp[2]);
	return output;
}

float FEM_Simulator::calculateNA_zeta(float xi[3], int Ai)
{
	float output = 0;
	int temp[3] = { FEM_Simulator::A[Ai][0], FEM_Simulator::A[Ai][1], FEM_Simulator::A[Ai][2] };
	output = 1 / 8.0 * temp[2] * (1 + xi[0] * temp[0]) * (1 + xi[1] * temp[1]);
	return output;
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

void FEM_Simulator::getGlobalNodesFromElem(int elem, int nodes[8])
{
	int sub[3];
	this->ind2sub(elem, this->gridSize, sub);
	// This order is defined by the pattern of A in the .h file. 
	
	int xShift = 0;
	int yShift = 0;
	int zShift = 0;
	for (int Ai = 0; Ai < 8; Ai++ ){
		// This makes the order of the nodes based on A, so if we change A the node order should change
		xShift = (FEM_Simulator::A[Ai][0] + 1) / 2;
		yShift = (FEM_Simulator::A[Ai][1] + 1) / 2;
		zShift = (FEM_Simulator::A[Ai][2] + 1) / 2;
		nodes[Ai] = (sub[2] + zShift) * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + yShift) * this->nodeSize[0] + (sub[0] + xShift);
	}
	/*
	nodes[0] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0];
	nodes[1] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0] + 1;
	nodes[2] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0] + 1;
	nodes[3] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0];
	nodes[4] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0];
	nodes[5] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0] + 1;
	nodes[6] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0] + 1;
	nodes[7] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0];
	return;
	*/
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

Eigen::Vector<int,27> FEM_Simulator::getNodeNeighbors(int globalNode)
{
	//returns a 27 element vector, idicating the 27 neighbors of our node (including the node itself). 
	// if a node does not exist in that neighbor idx, there will be a -1 in its place. For example, if our node is global node 0
	// then the first 13 elements would be -1, and so on.  
	
	// if the order changes here, need to change convertToNeighborIdx()
	Eigen::Vector<int,27> neighbors = Eigen::Vector<int,27>::Constant(-1);
	int nodeSub[3];
	this->ind2sub(globalNode, this->nodeSize, nodeSub);
	// I feel there is a more efficient way to do these checks
	int kStart = ((nodeSub[2] == 0) ? 1 : 0); // start at 0 if we are not on top layer, else start at 1
	int kEnd = ((nodeSub[2] == (this->nodeSize[2] - 1)) ? 2 : 3); // end at 1 if we are on bottom layer, else end at 2.
	int jStart = ((nodeSub[1] == 0) ? 1 : 0);
	int jEnd = ((nodeSub[1] == (this->nodeSize[1] - 1)) ? 2 : 3);
	int iStart = ((nodeSub[0] == 0) ? 1 : 0);
	int iEnd = ((nodeSub[0] == (this->nodeSize[1] - 1)) ? 2 : 3);
	for (int k = kStart; k < kEnd; k++) {
		for (int j = jStart; j < jEnd; j++) {
			for (int i = iStart; i < iEnd; i++) {
				int idx = i + 3 * j + 9 * k;
				neighbors(idx) = globalNode + (k - 1) * this->nodeSize[0] * this->nodeSize[1] + (j - 1) * this->nodeSize[0] + (i - 1);
			}
		}
	}
	return neighbors;
}

std::vector<int> FEM_Simulator::convertToLocalNode(int globalNode, int f)
{
	// TO DO, make this function look at this->A for its boundary checking. 
	std::vector<int> localIndices;
	int nodeSub[3];
	this->ind2sub(globalNode, this->nodeSize, nodeSub);
	bool validFlag = false;
	for (int nodeOption : elemNodeSurfaceMap[f]) { // go through each option based on the current face
		switch (nodeOption) {
		case 0: // for 0 to be an option, the global node cant be on the positive x y or z boundary
			if (!((nodeSub[0] == this->nodeSize[0] - 1) || (nodeSub[1] == this->nodeSize[1] - 1) || (nodeSub[2] == this->nodeSize[2] - 1))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 1:// for 1 to be an option, the global node can't be on the negative x or positive y/z boundary
			if (!((nodeSub[0] == 0) || (nodeSub[1] == this->nodeSize[1] - 1) || (nodeSub[2] == this->nodeSize[2] - 1))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 2:// for 2 to be an option, the global node can't be on the negative y or positive x/z boundary
			if (!((nodeSub[0] == this->nodeSize[1] - 1) || (nodeSub[1] == 0) || (nodeSub[2] == this->nodeSize[2] - 1))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 3:// for 3 to be an option, the global node can't be on the negative x/y or positive z boundary
			if (!((nodeSub[0] == 0) || (nodeSub[1] == 0) || (nodeSub[2] == this->nodeSize[2] - 1))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 4:// for 4 to be an option, the global node can't be on the negative z or positive x/y boundary
			if (!((nodeSub[0] == this->nodeSize[0] - 1) || (nodeSub[1] == this->nodeSize[1] - 1) || (nodeSub[2] == 0))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 5:// for 5 to be an option, the global node can't be on the negative z/x or positive y boundary
			if (!((nodeSub[0] == 0) || (nodeSub[1] == this->nodeSize[1] - 1) || (nodeSub[2] == 0))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 6:// for 6 to be an option, the global node can't be on the negative y/z or positive x boundary
			if (!((nodeSub[0] == this->nodeSize[0] - 1) || (nodeSub[1] == 0) || (nodeSub[2] == 0))) {
				localIndices.push_back(nodeOption);
			}
			break;
		case 7:// for 7 to be an option, the global node can't be on the negative x/y/z boundary
			if (!((nodeSub[0] == 0) || (nodeSub[1] == 0) || (nodeSub[2] == 0))) {
				localIndices.push_back(nodeOption);
			}
			break;
		}
	}

	return localIndices;
}

int FEM_Simulator::convertToGlobalNode(int localNode, int globalReference, int localReference)
{
	// if structure of this->A changes, this function will be wrong. 
	int nodeSub[3];
	int refSub[3];
	int tempSize[3] = { 2,2,2 };
	ind2sub(localNode, tempSize, nodeSub);
	ind2sub(localReference, tempSize, refSub);
	int xShift = nodeSub[0] - refSub[0];
	int yShift = nodeSub[1] - refSub[1];
	int zShift = nodeSub[2] - refSub[2];
	int globalNode = globalReference + xShift + yShift * this->nodeSize[0] + zShift * this->nodeSize[1] * this->nodeSize[0];
	return globalNode;
}

int FEM_Simulator::convertToNeighborIdx(int globalNode, int globalReference)
{
	// getNodeNeighbors is used as reference to create this mapping. 
	int difference = globalNode - globalReference; // the addition keeps the value positive and lets us use modulos
	int kShift = (difference / (this->nodeSize[0]*this->nodeSize[1])); // provides a value between -1 and 1
	int jShift = (difference % (this->nodeSize[0] * this->nodeSize[1])) / this->nodeSize[0]; // provides a value between -1 and 1
	int iShift = (difference % this->nodeSize[0]); // provides a value between -1 and 1
	int neighborIdx = kShift*9 + jShift*3 + iShift + 13; // if globalNode == globalReference, that is idx 13. 
	return neighborIdx;
}

float FEM_Simulator::createKABFunction(float xi[3], int Ai, int Bi)
{
	float KABfunc = 0;
	Eigen::Vector3<float> NAdotA;
	Eigen::Vector3<float> NAdotB;
	Eigen::Matrix3<float> J = this->J;

	NAdotA = this->calculateNA_dot(xi, Ai);
	NAdotB = this->calculateNA_dot(xi, Bi);

	Eigen::Matrix3f Jinv = J.inverse();
	Eigen::Matrix3f Jinv2 = Jinv * Jinv.transpose();

	KABfunc = (NAdotA.transpose() * this->J.inverse() * this->J.inverse().transpose() * NAdotB); // matrix math
	KABfunc = float(J.determinant() * this->TC * KABfunc); // Type issues if this multiplication is done with the matrix math so i am doing it on its own line
	return KABfunc;
}

float FEM_Simulator::createMABFunction(float xi[3], int Ai, int Bi)
{
	float MABfunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3<float> J = this->J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);

	MABfunc = (NAa * NAb) * J.determinant() * this->VHC; // matrix math
	return MABfunc;
}

float FEM_Simulator::createFintFunction(float xi[3], int Ai, int Bi)
{
	float FintFunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3<float> J = this->J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	FintFunc = this->MUA * (NAa * NAb) * J.determinant();
	// Output of this still needs to get multiplied by the NFR at node Bi
	return FintFunc;
}

float FEM_Simulator::createFjFunction(float xi[3], int Ai, int dim)
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
	FjFunc = (NAa * this->Jn) * Js.determinant();
	return FjFunc;
}

float FEM_Simulator::createFvFunction(float xi[3], int Ai, int dim)
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
	FvFunc = (NAa * this->HTC) * this->ambientTemp * Js.determinant();
	return FvFunc;
}

float FEM_Simulator::createFvuFunction(float xi[3], int AiBi, int dim)
{
	float FvuFunc = 0;
	int Ai = AiBi % 8; // AiBi is passed in in base 8. the ones digit is Ai, the 8s digit is Bi.
	int Bi = AiBi / 8;
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

	FvuFunc += (NAa * NAb * this->HTC) * Js.determinant();
	
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

int FEM_Simulator::determineNodeFace(int globalNode)
{
	int output = INTERNAL;
	int nodeSub[3];
	this->ind2sub(globalNode, this->nodeSize, nodeSub);
	if (nodeSub[2] == 0) {
		output += TOP;
	}
	if (nodeSub[2] == (this->nodeSize[2] - 1)) {
		output += BOTTOM;
	}
	if (nodeSub[1] == 0) {
		output += FRONT;
	}
	if (nodeSub[0] == (this->nodeSize[0] - 1)) {
		output += RIGHT;
	}
	if (nodeSub[1] == (this->nodeSize[1] - 1)) {
		output += BACK;
	}
	if (nodeSub[0] == 0) {
		output += LEFT;
	}
	return output;
}

void FEM_Simulator::setInitialTemperature(std::vector<std::vector<std::vector<float>>> Temp) {
	this->Temp = Temp;
	int gridSize[3]; 
	gridSize[0] = Temp.size() - 1; // Temp contains the temperature at the nodes, so we need to subtract 1 to get the elements
	gridSize[1] = Temp[0].size() - 1;
	gridSize[2] = Temp[0][0].size() - 1;
	this->setGridSize(gridSize);
}

void FEM_Simulator::setNFR(std::vector<std::vector<std::vector<float>>> NFR)
{
	this->NFR = NFR;

	if ((NFR.size() == this->gridSize[0]) && (NFR[0].size() == this->gridSize[1]) && (NFR[0][0].size() == this->gridSize[2])) {
		this->elemNFR = true;
	}
	else if ((NFR.size() == this->nodeSize[0]) && (NFR[0].size() == this->nodeSize[1]) && (NFR[0][0].size() == this->nodeSize[2])) {
		this->elemNFR = false;
	}
	else {
		throw std::invalid_argument("NFR must have the same number of entries as the node space or element space");
	}
}

void FEM_Simulator::setTissueSize(float tissueSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->tissueSize[i] = tissueSize[i];
	}
	this->setJ();
}

void FEM_Simulator::setTC(float TC) {
	this->TC = TC;
	this->setKe();
	this->setKn();
}

void FEM_Simulator::setVHC(float VHC) {
	this->VHC = VHC;
	this->setMe();
	this->setMn();
}

void FEM_Simulator::setMUA(float MUA) {
	this->MUA = MUA;
	setFeInt();
	this->setFnInt();
}

void FEM_Simulator::setHTC(float HTC) {
	this->HTC = HTC;
	this->setFv();
	this->setFvu();
}

void FEM_Simulator::setJn(float Jn)
{
	this->Jn = Jn;
	this->setFj();
}

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	this->ambientTemp = ambientTemp;
	this->setFv();
}

void FEM_Simulator::setGridSize(int gridSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->gridSize[i] = gridSize[i];
		this->nodeSize[i] = gridSize[i] + 1;
	}
	this->setJ();
}

void FEM_Simulator::setNodeSize(int nodeSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->gridSize[i] = nodeSize[i] - 1;
		this->nodeSize[i] = nodeSize[i];
	}
	setJ();
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

void FEM_Simulator::setJ() {
	this->J = this->calculateJ();
	this->Js1 = this->calculateJs(1);
	this->Js2 = this->calculateJs(2);
	this->Js3 = this->calculateJs(3);
	this->setKe();
	this->setMe();
	this->setFeInt();
	this->setKn();
	this->setMn();
	this->setFnInt();
}

void FEM_Simulator::setBoundaryConditions(int BC[6])
{
	for (int i = 0; i < 6; i++) {
		this->boundaryType[i] = static_cast<boundaryCond>(BC[i]);
	}
	this->initializeBoundaryNodes();
}

void FEM_Simulator::setKe() {
	// Taking advantage of the fact that J is costant across element and TC is constant across elements
	this->Ke.setZero();
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			this->Ke(Ai, Bi) = this->integrate(&FEM_Simulator::createKABFunction, 2, 0, Ai, Bi);
		}
	}

}

void FEM_Simulator::setKn()
{
	this->Kn.setZero();
	for (int i = 0; i < 27; i++) {
		int nodeSub[3];
		int tempNodeSize[3] = { 3,3,3 };
		this->ind2sub(i, tempNodeSize, nodeSub);
		for (int e = 0; e < 8; e++) {
			int tempElemSize[3] = { 2,2,2 };
			int eSub[3];
			this->ind2sub(e, tempElemSize, eSub);
			bool firstCond = (nodeSub[0] == eSub[0]) || (nodeSub[0] - 1 == eSub[0]);
			bool secondCond = (nodeSub[1] == eSub[1]) || (nodeSub[1] - 1 == eSub[1]);
			bool thirdCond = (nodeSub[2] == eSub[2]) || (nodeSub[2] - 1 == eSub[2]);
			if (firstCond && secondCond && thirdCond) {
				int Ai = (nodeSub[0] - 1 == eSub[0]) + (nodeSub[1] - 1 == eSub[1])*2 + (nodeSub[2] - 1 == eSub[2])*4;
				int Bi = 7-e;
				this->Kn(i) += this->integrate(&FEM_Simulator::createKABFunction, 2, 0, Ai, Bi);
			}
		}
		
	}
}

void FEM_Simulator::setMe() {
	// Taking advantage of the fact that J is costant across element and VHC is constant across elements
	this->Me.setZero();
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			this->Me(Ai, Bi) = this->integrate(&FEM_Simulator::createMABFunction, 2, 0, Ai, Bi);
		}
	}
}

void FEM_Simulator::setMn()
{
	this->Mn.setZero();
	for (int i = 0; i < 27; i++) {
		int nodeSub[3];
		int tempNodeSize[3] = { 3,3,3 };
		this->ind2sub(i, tempNodeSize, nodeSub);
		for (int e = 0; e < 8; e++) {
			int size[3] = { 2,2,2 };
			int eSub[3];
			this->ind2sub(e, size, eSub);
			bool firstCond = (nodeSub[0] == eSub[0]) || (nodeSub[0] - 1 == eSub[0]);
			bool secondCond = (nodeSub[1] == eSub[1]) || (nodeSub[1] - 1 == eSub[1]);
			bool thirdCond = (nodeSub[2] == eSub[2]) || (nodeSub[2] - 1 == eSub[2]);
			if (firstCond && secondCond && thirdCond) {
				int Ai = (nodeSub[0] - 1 == eSub[0]) + (nodeSub[1] - 1 == eSub[1])*2 + (nodeSub[2] - 1 == eSub[2])*4;
				int Bi = 7-e;
				this->Mn(i) += this->integrate(&FEM_Simulator::createMABFunction, 2, 0, Ai, Bi);;
			}
		}
		
	}
}

void FEM_Simulator::setFeInt()
{
	this->FeInt.setZero();
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			this->FeInt(Ai, Bi) = this->integrate(&FEM_Simulator::createFintFunction, 2, 0, Ai, Bi);
		}
	}
}

void FEM_Simulator::setFnInt()
{
	this->FnInt.setZero();
	for (int i = 0; i < 27; i++) {
		int nodeSub[3];
		int tempNodeSize[3] = { 3,3,3 };
		this->ind2sub(i, tempNodeSize, nodeSub);
		for (int e = 0; e < 8; e++) {
			int size[3] = { 2,2,2 };
			int eSub[3];
			this->ind2sub(e, size, eSub);
			bool firstCond = (nodeSub[0] == eSub[0]) || (nodeSub[0] - 1 == eSub[0]);
			bool secondCond = (nodeSub[1] == eSub[1]) || (nodeSub[1] - 1 == eSub[1]);
			bool thirdCond = (nodeSub[2] == eSub[2]) || (nodeSub[2] - 1 == eSub[2]);
			if (firstCond && secondCond && thirdCond) {
				int Ai = (nodeSub[0] - 1 == eSub[0]) + (nodeSub[1] - 1 == eSub[1]) * 2 + (nodeSub[2] - 1 == eSub[2]) * 4;
				int Bi = 7 - e;
				this->FnInt(i) += this->integrate(&FEM_Simulator::createFintFunction, 2, 0, Ai, Bi);;
			}
		}

	}
}

void FEM_Simulator::setFj() {
	this->Fj.setZero();
	for (int f = 0; f < 6; f++) { // iterate through each face
		for (int Ai : this->elemNodeSurfaceMap[f]) { // Go through nodes on face surface 
			this->Fj(Ai,f) = this->integrate(&FEM_Simulator::createFjFunction, 2, this->dimMap[f], Ai, this->dimMap[f]); // calculate FjA
		}
	} // iterate through faces
}

void FEM_Simulator::setFv() {
	this->Fv.setZero();
	for (int f = 0; f < 6; f++) { // iterate through each face
		for (int Ai : this->elemNodeSurfaceMap[f]) { // Go through nodes on face surface 
			this->Fv(Ai,f) = this->integrate(&FEM_Simulator::createFvFunction, 2, this->dimMap[f], Ai, this->dimMap[f]); // calculate FjA
		}
	} // iterate through faces
}

void FEM_Simulator::setFvu() {
	for (int f = 0; f < 6; f++) {
		this->Fvu[f].setZero();
		for (int Ai : elemNodeSurfaceMap[f]) {
			for (int Bi : elemNodeSurfaceMap[f]) {
				int AiBi = Bi * 8 + Ai; // had to be creative here to encode Ai and Bi in a single variable. We are using base 8. 
				// So if Bi is 1 and Ai is 7, the value is 15. 15 in base 8 is 17. 
				this->Fvu[f](Ai,Bi) = this->integrate(&FEM_Simulator::createFvuFunction, 2, this->dimMap[f], AiBi, this->dimMap[f]);
			}
		}
	}
}