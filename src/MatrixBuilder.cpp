#include "MatrixBuilder.h"


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
	this->FirrElem = Eigen::VectorXf::Zero(nValidNodes);
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

void MatrixBuilder::calculateHexFunction1D(float xi, int A)
{
	/* This function calculates the building block for the basis functions given the number of nodes in 1 Dimension
	* It uses the equation \prod^Nn1d_{B = 1; B != A} (xi - xi^B)/(xi^A - xi^B)
	*/
	float output = 1.0f;
	// This will produce a value of -1,1 for 2-node elements, and -1,0,1 for 3-node elements
	float xiA = -1 + Ai * (2 / float(nN1D_ - 1));
	// Get the product of the linear functions to build polynomial shape function in 1D
	for (int i = 0; i < nN1D_; i++) { // for 2-node elements its a single product.
		if (i != Ai) {
			float xiB = -1 + i * (2 / float(nN1D_ - 1)); //same as above
			output *= (xi - xiB) / (xiA - xiB);
		}
	}
	return output;
}

void MatrixBuilder::calculateHexFunction3D(float[3] xi, int A)
{
	/* Calculate the shape function output for given position in the element. */
	float output = 1.0f;
	int AiVec[3];
	int size[3] = { nN1D_,nN1D_,nN1D_ };
	//2-node ex: 0-(0,0,0), 1-(1,0,0), 2-(0,1,0), 3-(1,1,0), 4-(0,0,1), 5-(1,0,1), 6-(0,1,1), 7-(1,1,1)
	ind2sub(Ai, size, AiVec); // convert element node A to a subscript (xi,eta,zeta)
	for (int i = 0; i < 3; i++) {
		// multiply each polynomial shape function in 1D across all 3 dimensions
		output *= calculateHexFunction1D(xi[i], AiVec[i]);
	}
	return output;
}