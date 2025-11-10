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
	int nRelatedNodes = pow((nN1D_ * 2 - 1), 3);
	int nValidNodes = validNodes_.size();
	int nNodes = nodeList_.size();
	// Initialize matrices so that we don't have to resize them later
	FirrElem_ = Eigen::VectorXf::Zero(nValidNodes);
	Fint_ = Eigen::SparseMatrix<float>(nValidNodes, nNodes);
	Fint_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes)); // at most 27 non-zero entries per column
	// TODO: make these three vectors sparse because they will only be non zero on the boundary nodes
	Fconv_ = Eigen::VectorXf::Zero(nValidNodes);
	Fk_ = Eigen::VectorXf::Zero(nValidNodes);
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

Eigen::Matrix<float, 8, 8> MatrixBuilder::calculateMe(Element elem)
{
	Me_ = Eigen::Matrix<float, 8, 8>::Zero();

	integrateHex8(elem, [&](const float N[8],
						const Eigen::Vector3f dN_dxi[8],
						float w)
	{
		for (int a = 0; a < 8; ++a)
		{
			for (int b = 0; b < 8; ++b)
			{
				Me_(a, b) += N[a] * N[b] * w;
			}
		}
	});

	return Me_;
}

Eigen::Matrix<float, 8, 8> MatrixBuilder::calculateKe(const Element& elem)
{
	Ke_ = Eigen::Matrix<float, 8, 8>::Zero();

	integrateHex8(elem, [&](const float N[8], const Eigen::Vector3f dN_dxi[8], float w)
		{
			// Compute derivatives in physical space
			Eigen::Matrix3f Jinv = J_.inverse();

			Eigen::Vector3f dN_dx[8];
			for (int a = 0; a < 8; ++a)
				dN_dx[a] = Jinv * dN_dxi[a];  // chain rule: dN/dx = J^-1 * dN/dxi

			// Assemble K_e
			for (int a = 0; a < 8; ++a)
				for (int b = 0; b < 8; ++b)
					Ke_(a, b) += dN_dx[a].dot(dN_dx[b]) * w;
		});

	return Ke_;
}

Eigen::Matrix<float, 8, 1> MatrixBuilder::calculateFeq(const Element& elem, int faceIndex, float q)
{
	Eigen::Matrix<float, 8, 1> Feq = Eigen::Matrix<float, 8, 1>::Zero();

	integrateHexFace4(elem, faceIndex, [&](const float N_face[4], const int* nodesOnFace, float w)
		{
			for (int a = 0; a < 4; ++a)
				Feq(nodesOnFace[a]) += N_face[a] * q * w;
		});

	return Feq;
}

void MatrixBuilder::calculateJ(const Element& elem, const std::array<float, 3>& xi)
{
	J_.setZero();

	for (int a = 0; a < 8; ++a)
	{
		Eigen::Vector3f dN = calculateHexFunctionDeriv3D(xi, a);
		J_(0, 0) += dN(0) * elem.nodes[a].x;
		J_(0, 1) += dN(0) * elem.nodes[a].y;
		J_(0, 2) += dN(0) * elem.nodes[a].z;

		J_(1, 0) += dN(1) * elem.nodes[a].x;
		J_(1, 1) += dN(1) * elem.nodes[a].y;
		J_(1, 2) += dN(1) * elem.nodes[a].z;

		J_(2, 0) += dN(2) * elem.nodes[a].x;
		J_(2, 1) += dN(2) * elem.nodes[a].y;
		J_(2, 2) += dN(2) * elem.nodes[a].z;
	}
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

