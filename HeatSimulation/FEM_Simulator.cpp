#include "FEM_Simulator.h"
#include <iostream>

const int FEM_Simulator::A[8][3] = {{-1, -1, -1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};

FEM_Simulator::FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA)
{
	this->Temp = Temp;
	this->gridSize[0] = Temp.size() - 1; // Temp contains the temperature at the nodes, so we need to subtract 1 to get the elements
	this->gridSize[1] = Temp[1].size() - 1;
	this->gridSize[2] = Temp[1][1].size() - 1;
	for (int i = 0; i < 3; i++) {
		this->tissueSize[i] = tissueSize[i];
		this->nodeSize[i] = gridSize[i] + 1;
	}
	this->TC = TC;
	this->VHC = VHC;
	this->MUA = MUA;
	// If we change assumption of uniform voxel size in cuboid this won't work anymore.
	this->setJ();
	
	this->initializeBoundaryNodes();
}

void FEM_Simulator::solveFEA(std::vector<std::vector<std::vector<float>>> NFR)
{
	//this->NFR = NFR;
	int numElems = this->gridSize[0] * this->gridSize[1] * this->gridSize[2];
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];
	this->initializeBoundaryNodes();
	Eigen::SparseMatrix<float> Kbar(nNodes, nNodes);
	Kbar.reserve(Eigen::VectorXi::Constant(nNodes, 27)); // there will be at most 27 non-zero entries per column 
	Eigen::SparseMatrix<float> Mbar(nNodes, nNodes);
	Eigen::VectorXf Fbar(nNodes); // Containts Fint, Fj, and Fd
	Fbar.setZero();


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
		for (int Ai = 0; Ai < 8; Ai++) {
			nodeFace = this->determineNodeFace(elementGlobalNodes[Ai]);

			int nodeSub[3];
			this->ind2sub(elementGlobalNodes[Ai], this->nodeSize, nodeSub);

			// Determine if the node lies on a boundary and then determine what kind of boundary
			bool dirichletFlag = false;
			bool fluxFlag = false;
			if (nodeFace > 0) { // This check saves a lot time since most nodes are not on a surface.
				for (int f = 0; f < 6; f++) {
					if ((nodeFace >> f) & 1) { // Node lies on a boundary
						if (this->boundaryType[f] == FLUX) { // flux boundary
							Fbar(elementGlobalNodes[Ai]) += this->integrate(&FEM_Simulator::createFjFunction, 2, this->dimMap[f], Ai, this->dimMap[f]);
						}
						else if (this->boundaryType[f] == CONVECTION) { // Convection Boundary
							Fbar(elementGlobalNodes[Ai]) += this->integrate(&FEM_Simulator::createFvFunction, 2, this->dimMap[f], Ai, this->dimMap[f]);
							for (int Bi : BSurfMap[f]) {
								int AiBi = Bi * 8 + Ai; // had to be creative here to encode Ai and Bi in a single variable. We are using base 8. 
								// So if Bi is 1 and Ai is 7, the value is 15. 15 in base 8 is 17. 
								Kbar.coeffRef(elementGlobalNodes[Ai], elementGlobalNodes[Bi]) += this->integrate(&FEM_Simulator::createFvuFunction, 2, this->dimMap[f], AiBi, this->dimMap[f]);
							}
						}
					} // if Node is face f
				} // iterate through faces
			} // if node is a face

			// Now we will build the K, M and F matrice
			for (int Bi = 0; Bi < 8; Bi++) {
				Kbar.coeffRef(elementGlobalNodes[Ai], elementGlobalNodes[Bi]) += this->integrate(&FEM_Simulator::createKABFunction, 2, 0, Ai, Bi);
				Mbar.coeffRef(elementGlobalNodes[Ai], elementGlobalNodes[Bi]) += this->integrate(&FEM_Simulator::createMABFunction, 2, 0, Ai, Bi);

				int BiSub[3];
				ind2sub(elementGlobalNodes[Bi], this->nodeSize, BiSub);
				Fbar(elementGlobalNodes[Ai]) += (NFR[BiSub[0]][BiSub[1]][BiSub[2]]) * this->integrate(&FEM_Simulator::createFintFunction, 2, 0, Ai, Bi);
			}
		}
	}

	// Remove the rows corresponding to Dirichlet Nodes
	Eigen::VectorXf F = Fbar(this->validNodes);
	// Remove unecessary members of Kbar
	Eigen::SparseMatrix<float> K(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size()); // K is the reduced Kbar matrix
	K.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), 27)); // there will be at most 27 non-zero entries per column 
	Eigen::SparseMatrix<float> tempF(nNodes - this->dirichletNodes.size(), this->dirichletNodes.size()); // This will store the output of the dirichletBoundaryValues
	this->reduceSparseMatrix(Kbar, this->dirichletNodes, &K, &tempF, nNodes);

	// Remove unecessary memebers of Mbar
	Eigen::SparseMatrix<float> M(nNodes - this->dirichletNodes.size(), nNodes - this->dirichletNodes.size()); // M is the reduced Mbar matrix
	M.reserve(Eigen::VectorXi::Constant(nNodes - this->dirichletNodes.size(), 27)); // at most 27 non-zero entries per column
	Eigen::SparseMatrix<float> dummy(nNodes - this->dirichletNodes.size(), this->dirichletNodes.size()); // we don't actually care about the columns of Mbar that are removed 
	this->reduceSparseMatrix(Mbar, this->dirichletNodes, &M, &dummy, nNodes);

	// Create Fdirichlet based on dirichlet boundaries
	for (int col = 0; col < tempF.outerSize(); ++col) // iterate through columns 
		for (Eigen::SparseMatrix<float>::InnerIterator it(tempF, col); it; ++it) // iterate through rows that are nonzero
		{	// col refers to the index in the dirichletNodes array
			int nodeSub[3];
			ind2sub(this->dirichletNodes[col], this->nodeSize, nodeSub); // convert the dirichletNodes index to a subscript
			float dirichletValue = this->Temp[nodeSub[0]][nodeSub[1]][nodeSub[2]]; // Find the temperature value at that location
			
			F(it.row()) -= it.value() * dirichletValue; // Subtract it from F
		}

	// Solve Euler Family 
	// Initialize d vector
	Eigen::VectorXf dVec(nNodes - dirichletNodes.size());
	Eigen::VectorXf vVec(nNodes - dirichletNodes.size());
	Eigen::VectorXf dTilde(nNodes - dirichletNodes.size());
	int counter = 0;
	for (int n = 0; n < nNodes; n++) {
		if (std::find(dirichletNodes.begin(), dirichletNodes.end(), n) == dirichletNodes.end()) { //not in the list 
			int nodeSub[3];
			ind2sub(n, this->nodeSize, nodeSub);
			dVec(counter) = this->Temp[nodeSub[0]][nodeSub[1]][nodeSub[2]];
			vVec(counter) = 0;
		}
	}
	// Perform TimeStepping
	
	for (float t = deltaT; t < this->tFinal; t += this->deltaT) {
		dTilde = dVec + (1 - this->alpha) * this->deltaT * vVec;	
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
		solver.compute(M + this->alpha * deltaT * K);
		Eigen::VectorXf vVec2 = solver.solve(F - K * dTilde);
		dVec = dVec + this->deltaT * (this->alpha * vVec2 + (1 - this->alpha) * vVec);

		vVec = vVec2;
	}

	// Adjust our Temp with new d vector
	for (int n = 0; n < nNodes; n++) {
		if (std::find(dirichletNodes.begin(), dirichletNodes.end(), n) == dirichletNodes.end()) { //not in the list 
			int nodeSub[3];
			ind2sub(n, this->nodeSize, nodeSub);
			this->Temp[nodeSub[0]][nodeSub[1]][nodeSub[2]] = dVec(counter);
		}
	}
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
			if (dim == 0) { // integrate across all 3 axis
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

	MABfunc = (NAa * NAb); // matrix math
	MABfunc = MABfunc * J.determinant() * this->VHC; // Type issues if this multiplication is done with the matrix math so i am doing it on its own line
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

	long nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];
	for (int i = 0; i < nNodes; i++) {
		int nodeFace = this->determineNodeFace(i);
		int nodeSub[3];
		this->ind2sub(i, this->nodeSize, nodeSub);

		// Determine if the node lies on a boundary and then determine what kind of boundary
		bool dirichletFlag = false;
		bool fluxFlag = false;
		if (nodeFace == 0) { // This check saves a lot time since most nodes are not on a surface.
			this->validNodes.push_back(i);
		}
		else {
			bool valid = true;
			for (int f = 0; f < 6; f++) {
				if ((nodeFace >> f) & 1) { // Node lies on a boundary
					if (this->boundaryType[f] == HEATSINK) { // flux boundary
						valid = false;
						this->dirichletNodes.push_back(i);
						break;
					}
				}
			}
			if (valid) { // if none of faces of the nodes are dirichlet boundaries
				this->validNodes.push_back(i);
			}
		}
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

void FEM_Simulator::setTissueSize(float tissueSize[3]) {
	for (int i = 0; i < 3; i++) {
		this->tissueSize[i] = tissueSize[i];
	}
	this->setJ();
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

void FEM_Simulator::setAmbientTemp(float ambientTemp) {
	this->ambientTemp = ambientTemp;
}

void FEM_Simulator::setGridSize(int gridSize[3]) {
	for (int i = 0; i < 3; i++) {
		gridSize[i] = gridSize[i];
		this->nodeSize[i] = gridSize[i] + 1;
	}
	this->setJ();
}

void FEM_Simulator::setNodeSize(int nodeSize[3]) {
	for (int i = 0; i < 3; i++) {
		gridSize[i] = nodeSize[i] - 1;
		this->nodeSize[i] = nodeSize[i];
	}
	setJ();
}

void FEM_Simulator::setJ() {
	this->J = this->calculateJ();
	this->Js1 = this->calculateJs(1);
	this->Js2 = this->calculateJs(2);
	this->Js3 = this->calculateJs(3);
}

void FEM_Simulator::setBoundaryConditions(int BC[6])
{
	for (int i = 0; i < 6; i++) {
		this->boundaryType[i] = static_cast<boundaryCond>(BC[i]);
	}
	this->initializeBoundaryNodes();
}