#include "FEM_Simulator.h"

const int FEM_Simulator::A[8][3] = {{-1, -1, -1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};

FEM_Simulator::FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, int tissueSize[3], float TC, float VHC, float MUA)
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
}

void FEM_Simulator::solveFEA(std::vector<std::vector<std::vector<float>>> NFR)
{
	int numElems = this->gridSize[0] * this->gridSize[1] * this->gridSize[2];
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];

	Eigen::SparseMatrix<float> K(nNodes, nNodes);
	Eigen::SparseMatrix<float> M(nNodes, nNodes);
	std::vector<float> F(nNodes); // Containts Fint, Fj, and Fd
	for (int e = 0; e < numElems; e++) {
		this->currElement.elementNumber = e;

		int eSub[3];
		this->ind2sub(e, this->gridSize, eSub);
		int elementGlobalNodes[8]; //global nodes for element e
		this->getGlobalNodesFromElem(e, elementGlobalNodes);

		for (int Ai = 0; Ai < 8; Ai++) {// x,y,z coordinates for each of the global nodes
			this->getGlobalPosition(elementGlobalNodes[Ai], this->currElement.globalNodePositions[Ai]);
		}

		int nodeFace;
		for (int Ai = 0; Ai < 8; Ai++) {
			nodeFace = this->determineNodeFace(elementGlobalNodes[Ai]);

			int nodeSub[3];
			this->ind2sub(elementGlobalNodes[Ai], this->nodeSize, nodeSub);

			// Determine if the node lies on a boundary and then determine what kind of boundary
			bool dirichletFlag = false;
			bool fluxFlag = false;
			bool convectionFlag = false;
			if (nodeFace > 0) { // This check saves a lot time since most nodes are not on a surface.
				for (int f = 0; f < 6; f++) {
					if ((nodeFace >> f) & 1) { // Node lies on a boundary
						if (this->boundaryType[f] == HEATSINK) { // dirichlet boundary
							dirichletFlag == true;
						}
						else if (this->boundaryType[f] == FLUX) { // flux boundary
							fluxFlag = true;
						}
						else if (this->boundaryType[f] == CONVECTION) { // Convection Boundary
							convectionFlag = true;
						}
					} // if Node is face f
				} // iterate through faces
			} // if node is a face

			// Now we will build the K, M and F matrices
			if (!dirichletFlag) { // if the node is not a dirichlet boundary
				// Calculate Fj, Fint, K, M

				for (int Bi = 0; Bi < 8; Bi++) {
					K[elementGlobalNodes[Ai]][elementGlobalNodes[Bi]] += this->integrate(&calculateKAB); // TODO: K
					M[elementGlobalNodes[Ai]][elementGlobalNodes[Bi]] += 0; // TODO: M
				}

				F[elementGlobalNodes[Ai]] += NFR[nodeSub[0]][nodeSub[1]][nodeSub[2]]; // TODO: Fint
				if (fluxFlag) {
					F[elementGlobalNodes[Ai]] += 0; // TODO: Fj
				}
				if (convectionFlag) {
					F[elementGlobalNodes[Ai]] += 0; // TODO: Fj
				}

			}
			else { // If Node is a dirichlet boundary
				// Calculate Fd
			}
		}
	}
	// Remove unecessary members of Fint and Fj
	// Remove unecessary members of Mbar and Kbar
	// Create Fdirichlet based on dirichlet boundaries

	// Solve Euler Family 
}


float FEM_Simulator::calculateNA(float xi[3], int Ai)
{
	float output = 0;
	int temp[3] = { FEM_Simulator::A[Ai][0],FEM_Simulator::A[Ai][1], FEM_Simulator::A[Ai][2] };
	output = 1 / 8.0 * (1 + xi[0] * temp[0]) * (1 + xi[1] * temp[1]) * (1 + xi[2] * temp[2]);
	return output;
}

void FEM_Simulator::calculateJ(float deltaX[3], Eigen::Matrix3<float> J)
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
	J(0, 0) = deltaX[0] / 2.0;
	J(0, 1) = 0;
	J(0, 2) = 0;
	J(1, 0) = 0;
	J(1, 1) = deltaX[0]/2.0;
	J(1, 2) = 0;
	J(2, 0) = 0;
	J(2, 1) = 0;
	J(2, 2) = deltaX[2] / 2.0;

}

void FEM_Simulator::calculateJs(float deltaX[2], int dim, Eigen::Matrix2<float> Js)
{
	// dim should be +-{1,2,3}. The dimension indiciates the axis of the normal vector of the plane. 
	// +1 is equivalent to (1,0,0) normal vector. -3 is equivalent to (0,0,-1) normal vector. 
	// We assume the values of xi correspond to the values of the remaining two axis in ascending order.
	// If dim = 2, then xi[0] is for the x-axis and xi[1] is for the z axis. 
	int direction = dim / abs(dim);
	dim = abs(dim);
	Js(0, 1) = 0;
	Js(1, 0) = 0;
	if (dim == 1) {
		Js(0, 0) = deltaX[1] / 2.0;
		Js(1, 1) = deltaX[2] / 2.0;
	}
	else if (dim == 2) {
		Js(0, 0) = deltaX[0] / 2.0;
		Js(1, 1) = deltaX[2] / 2.0;
	}
	else if (dim == 3) {
		Js(0, 0) = deltaX[0] / 2.0;
		Js(1, 1) = deltaX[1] / 2.0;
	}

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

void FEM_Simulator::calculateNA_dot(float xi[3], int Ai, Eigen::Vector3<float> NA_dot)
{
	NA_dot(0) = FEM_Simulator::calculateNA_xi(xi, Ai);
	NA_dot(1) = FEM_Simulator::calculateNA_eta(xi, Ai);
	NA_dot(2) = FEM_Simulator::calculateNA_zeta(xi, Ai);
	return;
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

float FEM_Simulator::integrate(std::function<float(float[3], int, int)> fun, int points, int dim, int Ai, int Bi)
{
	std::vector<float> zeros;
	std::vector<float> weights;
	float output = 0;
	// TODO: Replace with automatic determination of zeros and weights
	if (points == 1) {
		zeros.push_back(0.0);
		weights.push_back(2.0);
	}
	else if (points == 2) {
		zeros.push_back(-sqrt(1 / 3.0));
		zeros.push_back(sqrt(1 / 3.0));
		weights.push_back(1.0);
		weights.push_back(1.0);
	}
	else if (points == 3) {
		zeros.push_back(-sqrt(3 / 5.0));
		zeros.push_back(0.0);
		zeros.push_back(sqrt(3 / 5.0));
		weights.push_back(5 / 9.0);
		weights.push_back(8 / 9.0);
		weights.push_back(5 / 9.0);
	}

	for (int i = 0; i < points; i++) {
		for (int j = 0; j < points; j++) {
			if (dim == 0) {
				for (int k = 0; k < points; k++) {
					float xi[3] = { zeros[i], zeros[j], zeros[k] };
					output += fun(xi, Ai, Bi) * weights[i] * weights[j] * weights[k];
				}
			} if (abs(dim) == 1) { // we are in the y-z plane
				float xi[3] = { dim / abs(dim), zeros[i], zeros[j] };
				output += fun(xi, Ai, Bi) * weights[i] * weights[j];
			}
			else if (abs(dim) == 2) { // we are in the x-z plane
				float xi[3] = { zeros[i], dim / abs(dim), zeros[j] };
				output += fun(xi, Ai, Bi) * weights[i] * weights[j];
			}
			else if (abs(dim) == 3) { // we are in the x-y plane
				float xi[3] = { zeros[i], zeros[j], dim / abs(dim) };
				output += fun(xi, Ai, Bi) * weights[i] * weights[j];
			}
		}
	}
}

void FEM_Simulator::getGlobalNodesFromElem(int elem, int nodes[8])
{
	int sub[3];
	this->ind2sub(elem, this->gridSize, sub);
	// This order is defined by the pattern of A in the .h file. 
	// TODO: Make the A ordering explicit
	nodes[0] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0];
	nodes[1] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0] + 1;
	nodes[2] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0] + 1;
	nodes[3] = sub[2] * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0];
	nodes[4] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0];
	nodes[5] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + sub[1] * this->nodeSize[0] + sub[0] + 1;
	nodes[6] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0] + 1;
	nodes[7] = (sub[2] + 1) * (this->nodeSize[0] * this->nodeSize[1]) + (sub[1] + 1) * this->nodeSize[0] + sub[0];
	return;

}

void FEM_Simulator::getGlobalPosition(int globalNode, float position[3])
{
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

float FEM_Simulator::calculateKAB(float xi[3])
{

	float KAB = integrate(&createKABFunction, 2, 0);
	return KAB;
}

float FEM_Simulator::createKABFunction(float xi[3], int Ai, int Bi)
{
	float KABfunc = 0;
	Eigen::Vector3<float> NAdotA;
	Eigen::Vector3<float> NAdotB;
	Eigen::Matrix3<float> J;

	this->calculateNA_dot(xi, Ai, NAdotA);
	this->calculateNA_dot(xi, Bi, NAdotB);
	this->calculateJ(this->currElement.globalNodePositions, J);

	KABfunc = (NAdotA.transpose() * J.inverse() * J.inverse().transpose() * NAdotB); // matrix math
	KABfunc = KABfunc * J.determinant() * this->TC; // Type issues if this multiplication is done with the matrix math so i am doing it on its own line
	return KABfunc;
}

float FEM_Simulator::createMABFunction(float xi[3], int Ai, int Bi)
{
	float MABfunc = 0;
	float NAa;
	float NAb;
	Eigen::Matrix3<float> J;

	NAa = this->calculateNA(xi, Ai);
	NAb = this->calculateNA(xi, Bi);
	this->calculateJ(xi, this->currElement.globalNodePositions, J);

	MABfunc = (NAa * NAb); // matrix math
	MABfunc = MABfunc * J.determinant() * this->VHC; // Type issues if this multiplication is done with the matrix math so i am doing it on its own line
	return MABfunc;
}

void FEM_Simulator::setBoundaryConditions(int BC[6])
{
	for (int i = 0; i < 6; i++) {
		this->boundaryType[i] = static_cast<boundaryCond>(i);
	}
	this->initializeBoundaryNodes();
}


void FEM_Simulator::initializeBoundaryNodes()
{
	// we only need to scan nodes on the surface. Since we are assuming a cuboid this is easy to predetermine
	int nNodes = this->nodeSize[0] * this->nodeSize[1] * this->nodeSize[2];
	int face = 0;
	for (int n = 0; n < (this->nodeSize[0] * this->nodeSize[1]); n++) { // Nodes on the bottom surface
	}
	for (int n = (this->nodeSize[0] * this->nodeSize[1]); n) {}
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

