#include "MatrixBuilder.h"


MatrixBuilder::MatrixBuilder()
{
}

MatrixBuilder::MatrixBuilder(std::vector<Node> nodeList, std::vector<Element> elemList)
{
}

MatrixBuilder::MatrixBuilder(std::string filename)
{
}


void MatrixBuilder::resetMatrices()
{
	// The number of non-zeros per column assuming hexahedral elements
	int nRelatedNodes = pow((nN1D_ * 2 - 1), 3);
	//  number of non-dirichlet nodes
	int nValidNodes = validNodes_.size();
	// total number of nodes
	int nNodes = nodeList_.size();
	// Initialize matrices so that we don't have to resize them later
	FintElem_ = Eigen::SparseMatrix<float>(nValidNodes, elemList_.size());
	FintElem_.reserve(Eigen::VectorXi::Constant(nNodes, nN1D_*nN1D_*nN1D_));

	Fint_ = Eigen::SparseMatrix<float>(nValidNodes, nNodes);
	Fint_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes)); 
	
	Fconv_ = Eigen::SparseMatrix<float>(nValidNodes, nNodes);
	Fconv_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes));

	Fk_ = Eigen::SparseMatrix<float>(nValidNodes, nNodes);
	Fint_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes));

	Fflux_ = Eigen::VectorXf::Zero(nValidNodes);
	Fq_ = Eigen::VectorXf::Zero(nValidNodes);

	// M and K will be sparse matrices because nodes are shared by relatively few elements
	M_ = Eigen::SparseMatrix<float>(nValidNodes, nValidNodes);
	M_.reserve(Eigen::VectorXi::Constant(nValidNodes, nRelatedNodes)); // at most 27 non-zero entries per column
	K_ = Eigen::SparseMatrix<float>(nValidNodes, nValidNodes);
	K_.reserve(Eigen::VectorXi::Constant(nValidNodes, nRelatedNodes)); // at most 27 non-zero entries per column
	// The Kconv matrix may also be able to be initialized differently since we know that it will only have values on the boundary ndoes.
	Q_ = Eigen::SparseMatrix<float>(nValidNodes, nValidNodes);
	Q_.reserve(Eigen::VectorXi::Constant(nValidNodes, nRelatedNodes)); // at most 27 non-zero entries per column
}

void MatrixBuilder::buildMatrices()
{
	resetMatrices();

	long nElem = elemList_.size();
	for (int elemIdx = 0; elemIdx < nElem; elemIdx++)
	{
		Element elem = elemList_[elemIdx];
		applyElement(elem, elemIdx);
		applyBoundary(elem);
	}
}

void MatrixBuilder::applyElement(Element elem, long elemIdx)
{
	int nodesPerElem = elem.nodes.size();
	long matrixRow = 0;
	long matrixCol = 0;
	Eigen::MatrixXf Me = calculateMe(elem);
	Eigen::MatrixXf Fe = calculateMe(elem);
	Eigen::MatrixXf Ke = calculateKe(elem);

	for (int A = 0; A < nodesPerElem; A++)
	{
		Node currNode = nodeList_[elem.nodes[A]];
		if (!currNode.isDirichlet)
		{
			// handle node-neighbor interactions
			matrixRow = nodeMap_[elem.nodes[A]];
			for (int B = 0; B < nodesPerElem; B++)
			{
				long neighborIdx = elem.nodes[B];
				matrixCol = nodeMap_[neighborIdx];
				Node neighbor = nodeList_[neighborIdx];
				// Add effect of nodal fluence rate 
				Fint_.coeffRef(matrixRow, neighborIdx) += Fe(A, B);
				// Add effect of element fluence rate
				FintElem_.coeffRef(matrixRow, elemIdx) += Fe(A, B);
				if (!neighbor.isDirichlet)
				{
					// add effect of mass matrix
					M_.coeffRef(matrixRow, matrixCol) += Me(A, B);
					// add effect of conductivity matrix
					K_.coeffRef(matrixRow, matrixCol) += Ke(A, B);
				}
				else
				{
					// add conductivity as a forcing effect
					Fk_.coeffRef(matrixRow, neighborIdx) -= Ke(A, B);
				}
			} // for each neighbor node
		} // if node is not dirichlet
	} // for each node in element
}

void MatrixBuilder::applyBoundary(Element elem)
{
	int nodesPerElem = elem.nodes.size();
	long matrixRow = 0;
	long matrixCol = 0;
	// handle boundary conditions of element
	for (int f; f < 6; f++) // iterate over faces
	{
		if (elem.faceBoundary[f] == BoundaryType::CONVECTION)
		{
			Eigen::VectorXf Feflux = calculateFeq(elem, f, 1);
			Eigen::VectorXf FeConv = calculateFeConv(elem, f);
			for (int A = 0; A < nodesPerElem; A++)
			{
				matrixRow = nodeMap_[elem.nodes[A]];
				// the portion of convection due to ambient temperature that acts like a constant flux boundary. 
				// needs to be multiplied by htc and ambient temp to be the correct value.
				Fq_(matrixRow) += Feflux(A);
				for (int B = 0; B < nodesPerElem; B++)
				{
					long neighborIdx = elem.nodes[B];
					matrixCol = nodeMap_[neighborIdx];
					Node neighbor = nodeList_[neighborIdx];
					if (!neighbor.isDirichlet)
					{
						// add effect of node temperature on convection
						Q_.coeffRef(matrixRow, matrixCol) += FeConv(A, B);
					}
					else
					{
						// Add effect of node temperature as forcing function
						Fconv_.coeffRef(matrixRow, neighborIdx) -= FeConv(A, B);
					}
				}
			}
		}
		else if (elem.faceBoundary[f] == BoundaryType::FLUX)
		{
			Eigen::VectorXf Feflux = calculateFeq(elem, f, 1);
			for (int A = 0; A < nodesPerElem; A++)
			{
				matrixRow = nodeMap_[elem.nodes[A]];
				this->Fflux_(matrixRow) += Feflux(A);
			}
		}
	}
}

float MatrixBuilder::calculateHexFunction1D(float xi, int A)
{
	/* This function calculates the building block for the basis functions given the number of nodes in 1 Dimension
	* It uses the equation \prod^Nn1d_{B = 1; B != A} (xi - xi^B)/(xi^A - xi^B)
	*/
	float output = 1.0f;
	// This will produce a value of -1,1 for 2-node elements, and -1,0,1 for 3-node elements
	float xiA = -1 + A * (2 / float(nN1D_ - 1));
	// Get the product of the linear functions to build polynomial shape function in 1D
	for (int i = 0; i < nN1D_; i++) { // for 2-node elements its a single product.
		if (i != A) {
			float xiB = -1 + i * (2 / float(nN1D_ - 1)); //same as above
			output *= (xi - xiB) / (xiA - xiB);
		}
	}
	return output;
}

float MatrixBuilder::calculateHexFunction3D(const std::array<float,3>& xi, int A)
{
	/* Calculate the shape function output for given position in the element. */
	float output = 1.0f;
	std::array<long, 3> AiVec;
	std::array<long,3> size = { nN1D_,nN1D_,nN1D_ };
	//2-node ex: 0-(0,0,0), 1-(1,0,0), 2-(0,1,0), 3-(1,1,0), 4-(0,0,1), 5-(1,0,1), 6-(0,1,1), 7-(1,1,1)
	AiVec = ind2sub(A, size);
	for (int i = 0; i < 3; i++) {
		// multiply each polynomial shape function in 1D across all 3 dimensions
		output *= calculateHexFunction1D(xi[i], AiVec[i]);
	}
	return output;
}

float MatrixBuilder::calculateHexFunctionDeriv1D(float xi, int Ai) {
	/* This function forms the building block for the derivative of the full shape function in 3D.
	This function assumes cuboid elements and works for linear or quadratic shape functions. */
	float output = 0;
	if (nN1D_ == 2) {
		if (Ai == 0) {
			output = -1 / 2.0f;
		}
		else if (Ai == 1) {
			output = 1 / 2.0f;
		}
	}
	else if (nN1D_ == 3) {
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

Eigen::Vector3f MatrixBuilder::calculateHexFunctionDeriv3D(const std::array<float, 3>& xi, int Ai)
{
	/* Calculate the derivative of the shape function with respect to all 3 axis. The result is a 3x1 vector.
	*/
	Eigen::Vector3f NA_dot;
	std::array<long, 3> AiVec;
	std::array<long,3> size = { nN1D_,nN1D_,nN1D_ };
	AiVec = ind2sub(Ai, size);
	NA_dot(0) = calculateHexFunctionDeriv1D(xi[0], AiVec[0]) * calculateHexFunction1D(xi[1], AiVec[1]) * calculateHexFunction1D(xi[2], AiVec[2]);
	NA_dot(1) = calculateHexFunctionDeriv1D(xi[1], AiVec[1]) * calculateHexFunction1D(xi[0], AiVec[0]) * calculateHexFunction1D(xi[2], AiVec[2]);
	NA_dot(2) = calculateHexFunctionDeriv1D(xi[2], AiVec[2]) * calculateHexFunction1D(xi[0], AiVec[0]) * calculateHexFunction1D(xi[1], AiVec[1]);
	return NA_dot;
}

float MatrixBuilder::calculateTetFunction3D(const std::array<float, 3>& xi, int A)
{ 
	switch (A) {
	case 0:
		return 1.0f - xi[0] - xi[1] - xi[2]; // N0
	case 1:
		return xi[0];                       // N1
	case 2:
		return xi[1];                       // N2
	case 3:
		return xi[2];                       // N3
	default:
		throw std::out_of_range("Bad node index in calculateTetFunction3D");
	}
}

Eigen::Vector3f MatrixBuilder::calculateTetFunctionDeriv3D(const std::array<float, 3>& xi, int A)
{
	Eigen::Vector3f dN;

	switch (A) {
	case 0:
		dN << -1.0f, -1.0f, -1.0f;
		break;
	case 1:
		dN << 1.0f, 0.0f, 0.0f;
		break;
	case 2:
		dN << 0.0f, 1.0f, 0.0f;
		break;
	case 3:
		dN << 0.0f, 0.0f, 1.0f;
		break;
	default:
		throw std::out_of_range("Bad node index in calculateTetFunctionDeriv3D");
	}

	return dN;
}


Eigen::MatrixXf MatrixBuilder::calculateMe(Element elem)
{
	Eigen::MatrixXf Me = Eigen::Matrix<float, 8, 8>::Zero();

	integrateHex8(elem, false, [&](const float N[8],
						const Eigen::Vector3f dN_dx[8],
						float w)
	{
		for (int a = 0; a < 8; ++a)
		{
			for (int b = 0; b < 8; ++b)
			{
				Me(a, b) += N[a] * N[b] * w;
			}
		}
	});

	return Me;
}

Eigen::MatrixXf MatrixBuilder::calculateKe(const Element& elem)
{
	Eigen::MatrixXf Ke = Eigen::Matrix<float, 8, 8>::Zero();

	integrateHex8(elem, true, [&](const float N[8], const Eigen::Vector3f dN_dx[8], float w)
		{
			// Assemble K_e
			for (int a = 0; a < 8; ++a)
				for (int b = 0; b < 8; ++b)
					Ke(a, b) += dN_dx[a].dot(dN_dx[b]) * w;
		});

	return Ke;
}

Eigen::MatrixXf MatrixBuilder::calculateFeq(const Element& elem, int faceIndex, float q)
{
	Eigen::Matrix<float, 8, 1> Feq = Eigen::Matrix<float, 8, 1>::Zero();

	integrateHexFace4(elem, faceIndex, [&](const float N_face[4], const int* nodesOnFace, float w)
		{
			for (int a = 0; a < 4; ++a)
				Feq(nodesOnFace[a]) += N_face[a] * q * w;
		});

	return Feq;
}

Eigen::MatrixXf MatrixBuilder::calculateFeConv(const Element& elem, int faceIndex)
{
	Eigen::Matrix<float, 8, 8> Qe = Eigen::Matrix<float, 8, 8>::Zero();

	// Integrate over the specified face
	integrateHexFace4(elem, faceIndex, [&](const float N[4], const int* nodesOnFace, float w) {
		// Local 4x4 face contribution
		Eigen::Matrix4f Qf = Eigen::Matrix4f::Zero();
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				Qf(i, j) = N[i] * N[j] * w;

		// Scatter local 4x4 into 8x8 element matrix
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				Qe(nodesOnFace[i], nodesOnFace[j]) += Qf(i, j);
		});

	return Qe;
}

Eigen::Matrix3f MatrixBuilder::calculateJ(const Element& elem, const std::array<float, 3>& xi)
{
	Eigen::Matrix3f J = Eigen::Matrix3f::Zero();

	for (int a = 0; a < 8; ++a)
	{
		Eigen::Vector3f dN = calculateHexFunctionDeriv3D(xi, a);
		J(0, 0) += dN(0) * nodeList_[elem.nodes[a]].x;
		J(0, 1) += dN(0) * nodeList_[elem.nodes[a]].y;
		J(0, 2) += dN(0) * nodeList_[elem.nodes[a]].z;

		J(1, 0) += dN(1) * nodeList_[elem.nodes[a]].x;
		J(1, 1) += dN(1) * nodeList_[elem.nodes[a]].y;
		J(1, 2) += dN(1) * nodeList_[elem.nodes[a]].z;

		J(2, 0) += dN(2) * nodeList_[elem.nodes[a]].x;
		J(2, 1) += dN(2) * nodeList_[elem.nodes[a]].y;
		J(2, 2) += dN(2) * nodeList_[elem.nodes[a]].z;
	}
	return J;
}

std::array<long, 3> MatrixBuilder::ind2sub(long idx,const std::array<long,3>& size) {
	std::array<long, 3> sub;
	
	sub[0] = (idx % size[0]);
	sub[1] = (idx % (size[0] * size[1])) / size[0];
	sub[2] = idx / (size[0] * size[1]);

	return sub;
}

void MatrixBuilder::setNodeList(std::vector<Node> nodeList)
{
	nodeList_ = nodeList;
}

void MatrixBuilder::setElementList(std::vector<Element> elemList)
{
	elemList_ = elemList;
}

