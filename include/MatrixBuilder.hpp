#pragma once
#include <vector> 
#include <array>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <stdexcept>
#include "Mesh.hpp"
#include "ShapeFunctions/HexLinear.hpp"
#include "ShapeFunctions/TetLinear.hpp"

class MatrixBuilder
{
public:

	MatrixBuilder(const Mesh& mesh) : mesh_(mesh) {}

	void buildMatrices()
	{
		if ((mesh_.order() == GeometricOrder::LINEAR) && (mesh_.elementShape() == Shape::HEXAHEDRAL))
		{
			buildMatricesT<ShapeFunctions::HexLinear>();
		}
		else if ((mesh_.order() == GeometricOrder::LINEAR) && (mesh_.elementShape() == Shape::TETRAHEDRAL))
		{
			buildMatricesT<ShapeFunctions::TetLinear>();
		}
	}

	void setNodeMap()
	{
		// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
		// A value of -1 at index i, indicates that global node i is a dirichlet node. 
		nodeMap_.resize(mesh_.nodes().size());
		std::fill(nodeMap_.begin(), nodeMap_.end(), 0);
		// First go through the boundary faces and set all nodes on a heatsink (dirichlet) face to -1
		for (BoundaryFace face : mesh_.boundaryFaces())
		{
			if (face.type == HEATSINK)
			{
				for (long n : face.nodes)
				{
					nodeMap_[n] = -1;
				}
			}
		}
		// Then go through all the nodes that aren't -1 and set them to an increasing value from 0 to n-1;
		// Also, store the index of the valid node. 
		nNonDirichlet_ = 0;
		for (int i = 0; i < mesh_.nodes().size(); i++)
		{
			if (nodeMap_[i] == 0)
			{
				nodeMap_[i] = nNonDirichlet_;
				validNodes_.push_back(i);
				nNonDirichlet_++;
			}
		}
	}

	template <typename ShapeFunc>
	void resetMatrices()
	{
		// The number of non-zeros per column this should be a safe number
		int nRelatedNodes = ShapeFunc::nNodes * ShapeFunc::nNodes * ShapeFunc::nNodes;
		//  number of non-dirichlet nodes
		// total number of nodes
		int nNodes = mesh_.nodes().size();
		int nodesPerElem = ShapeFunc::nNodes;
		// Initialize matrices so that we don't have to resize them later
		FintElem_ = Eigen::SparseMatrix<float>(nNonDirichlet_, mesh_.elements().size());
		FintElem_.reserve(Eigen::VectorXi::Constant(nNodes, nodesPerElem));

		Fint_ = Eigen::SparseMatrix<float>(nNonDirichlet_, nNodes);
		Fint_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes));

		Fconv_ = Eigen::SparseMatrix<float>(nNonDirichlet_, nNodes);
		Fconv_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes));

		Fk_ = Eigen::SparseMatrix<float>(nNonDirichlet_, nNodes);
		Fint_.reserve(Eigen::VectorXi::Constant(nNodes, nRelatedNodes));

		Fflux_ = Eigen::VectorXf::Zero(nNonDirichlet_);
		Fq_ = Eigen::VectorXf::Zero(nNonDirichlet_);

		// M and K will be sparse matrices because nodes are shared by relatively few elements
		M_ = Eigen::SparseMatrix<float>(nNonDirichlet_, nNonDirichlet_);
		M_.reserve(Eigen::VectorXi::Constant(nNonDirichlet_, nRelatedNodes)); // at most 27 non-zero entries per column
		K_ = Eigen::SparseMatrix<float>(nNonDirichlet_, nNonDirichlet_);
		K_.reserve(Eigen::VectorXi::Constant(nNonDirichlet_, nRelatedNodes)); // at most 27 non-zero entries per column
		// The Kconv matrix may also be able to be initialized differently since we know that it will only have values on the boundary ndoes.
		Q_ = Eigen::SparseMatrix<float>(nNonDirichlet_, nNonDirichlet_);
		Q_.reserve(Eigen::VectorXi::Constant(nNonDirichlet_, nRelatedNodes)); // at most 27 non-zero entries per column
	}


	template <typename ShapeFunc>
	void applyElement(Element elem, long elemIdx)
	{
		int nodesPerElem = ShapeFunc::nNodes;
		long matrixRow = 0;
		long matrixCol = 0;
		Eigen::MatrixXf Me = calculateMe<ShapeFunc>(elem);
		Eigen::MatrixXf Fe = calculateMe<ShapeFunc>(elem);
		Eigen::MatrixXf Ke = calculateKe<ShapeFunc>(elem);

		for (int A = 0; A < nodesPerElem; A++)
		{
			Node currNode = mesh_.nodes()[elem.nodes[A]];
			matrixRow = nodeMap_[elem.nodes[A]];
			if (matrixRow >= 0)
			{
				// handle node-neighbor interactions

				for (int B = 0; B < nodesPerElem; B++)
				{
					long neighborIdx = elem.nodes[B];
					matrixCol = nodeMap_[neighborIdx];
					Node neighbor = mesh_.nodes()[neighborIdx];
					// Add effect of nodal fluence rate 
					Fint_.coeffRef(matrixRow, neighborIdx) += Fe(A, B);
					// Add effect of element fluence rate
					FintElem_.coeffRef(matrixRow, elemIdx) += Fe(A, B);
					if (matrixCol >= 0) // non dirichlet node
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

	template <typename ShapeFunc>
	void applyBoundary(BoundaryFace face)
	{
		if (face.type == CONVECTION)
		{
			long matrixRow = 0;
			long matrixCol = 0;
			Element elem = mesh_.elements()[face.elemID];
			int nodesPerElem = ShapeFunc::nNodes;
			Eigen::VectorXf Feflux = calculateFeFlux<ShapeFunc>(elem, face.localFaceID, 1);
			Eigen::MatrixXf FeConv = calculateFeConv<ShapeFunc>(elem, face.localFaceID);
			for (int A = 0; A < nodesPerElem; A++)
			//TODO: Change this to only iterate over face nodes instead of all nodes
			{
				long matrixRow = nodeMap_[elem.nodes[A]];
				if (matrixRow >= 0)
				{
					// the portion of convection due to ambient temperature that acts like a constant flux boundary. 
					// needs to be multiplied by htc and ambient temp to be the correct value.
					Fq_(matrixRow) += Feflux(A);
					for (int B = 0; B < nodesPerElem; B++)
					{
						long neighborIdx = elem.nodes[B];
						matrixCol = nodeMap_[neighborIdx];
						Node neighbor = mesh_.nodes()[neighborIdx];
						if (matrixCol >= 0) // non dirichlet node
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
		}
		else if (face.type == FLUX)
		{
			Element elem = mesh_.elements()[face.elemID];
			long matrixRow = 0;
			Eigen::VectorXf Feflux = calculateFeFlux<ShapeFunc>(elem, face.localFaceID, 1);		
			for (int A = 0; A < ShapeFunc::nNodes; A++)
			//TODO: Change this to only iterate over face nodes instead of all nodes
			{
				matrixRow = nodeMap_[elem.nodes[A]];
				if (matrixRow >= 0)
					/*by adding here, we are implicitly stating that a positve flux value will add heat to the system
					* mathematically, we technically should be subtracting, but from a user perspective, it makes more sense
					* that a positive heat flux will add heat at the boundary. 
					*/ 
					Fflux_(matrixRow) += Feflux(A);
			}
		}
	}

	template <typename ShapeFunc>
	Eigen::MatrixXf calculateMe(const Element& elem) 
	{

		Eigen::MatrixXf Me = Eigen::MatrixXf::Zero(ShapeFunc::nNodes, ShapeFunc::nNodes);

		// simple 2-point Gauss integration in each axis (example)
		const std::vector<std::array<float, 3>>& gp = ShapeFunc::gaussPoints();
		const std::vector<std::array<float, 3>>& w = ShapeFunc::weights();

		for (int i = 0; i < ShapeFunc::nGP; i++)
		{
			std::array<float, 3> xi = gp[i];
			std::vector<Eigen::Vector3f> dNdxi(ShapeFunc::nNodes);
			std::array<float,ShapeFunc::nNodes> Nvals;
			for (int a = 0; a < ShapeFunc::nNodes; a++)
			{
				Nvals[a] = ShapeFunc::N(xi, a);
				dNdxi[a] = ShapeFunc::dNdxi(xi, a);
			}
			// Jacobian and determinant (simplified)
			Eigen::Matrix3f J = calculateJ<ShapeFunc>(elem, dNdxi);
			float detJ = J.determinant();
			float weight = w[i][0] * w[i][1] * w[i][2] * detJ;

			for (int a = 0; a < ShapeFunc::nNodes; a++)
				for (int b = 0; b < ShapeFunc::nNodes; b++)
					Me(a, b) += Nvals[a] * Nvals[b] * weight;
		}

		return Me;
	}

	template <typename ShapeFunc>
	Eigen::MatrixXf calculateKe(const Element& elem)
	{
		Eigen::MatrixXf Ke = Eigen::MatrixXf::Zero(ShapeFunc::nNodes, ShapeFunc::nNodes);

		const std::vector<std::array<float, 3>>& gp = ShapeFunc::gaussPoints();
		const std::vector<std::array<float, 3>>& w = ShapeFunc::weights();

		for (int i = 0; i < ShapeFunc::nGP; ++i)
		{
			const std::array<float, 3>& xi = gp[i];

			// Shape function derivatives in reference coordinates
			std::vector<Eigen::Vector3f> dNdxi(ShapeFunc::nNodes);
			for (int a = 0; a < ShapeFunc::nNodes; ++a)
				dNdxi[a] = ShapeFunc::dNdxi(xi, a);

			// Compute Jacobian and inverse
			Eigen::Matrix3f J = calculateJ<ShapeFunc>(elem, dNdxi);
			Eigen::Matrix3f Jinv = J.inverse();
			float detJ = J.determinant();

			// Derivatives in physical coordinates
			std::array<Eigen::Vector3f, ShapeFunc::nNodes> dNdx;
			for (int a = 0; a < ShapeFunc::nNodes; ++a)
				dNdx[a] = Jinv * dNdxi[a];

			// Weight for this Gauss point
			float weight = w[i][0] * w[i][1] * w[i][2] * detJ;

			// Assemble stiffness matrix
			for (int a = 0; a < ShapeFunc::nNodes; ++a)
				for (int b = 0; b < ShapeFunc::nNodes; ++b)
					Ke(a, b) += dNdx[a].dot(dNdx[b]) * weight;
				
		}

		return Ke;
	}

	template <typename ShapeFunc>
	Eigen::MatrixXf calculateFeFlux(const Element& elem, int faceIndex, float q)
	{
		Eigen::Matrix<float, ShapeFunc::nNodes, 1> Fe = Eigen::Matrix<float, ShapeFunc::nNodes, 1>::Zero();

		const auto& gp = ShapeFunc::faceGaussPoints(faceIndex);
		const auto& w = ShapeFunc::faceWeights(faceIndex);

		for (int i = 0; i < ShapeFunc::nFaceGP; ++i)
		{
			std::array<float, ShapeFunc::nFaceNodes> N_face;
			std::array<Eigen::Vector2f, ShapeFunc::nFaceNodes> dN_dxi_eta;

			for (int a = 0; a < ShapeFunc::nFaceNodes; ++a)
			{
				N_face[a] = ShapeFunc::N_face(gp[i], a, faceIndex);
				dN_dxi_eta[a] = ShapeFunc::dNdxi_face(gp[i], a, faceIndex);
			}

			// Compute 2x3 surface Jacobian
			Eigen::Matrix<float, 2, 3> JFace = Eigen::Matrix<float, 2, 3>::Zero();
			for (int a = 0; a < ShapeFunc::nFaceNodes; ++a)
			{
				const Node& n = mesh_.nodes()[elem.nodes[ShapeFunc::faceConnectivity[faceIndex][a]]];
				Eigen::Vector3f nodePos(n.x, n.y, n.z);
				JFace += dN_dxi_eta[a] * nodePos.transpose();
			}

			// Surface determinant: norm of cross product of rows
			float detJ = (JFace.row(0).cross(JFace.row(1))).norm();

			float weight = w[i][0] * w[i][1] * detJ;

			for (int a = 0; a < ShapeFunc::nFaceNodes; ++a)
				Fe(ShapeFunc::faceConnectivity[faceIndex][a]) += N_face[a] * q * weight;
		}

		return Fe;
	}

	template <typename ShapeFunc>
	Eigen::MatrixXf calculateFeConv(const Element& elem, int faceIndex)
	{
		constexpr int nNodes = ShapeFunc::nNodes;
		constexpr int nFaceNodes = ShapeFunc::nFaceNodes;

		Eigen::Matrix<float, nNodes, nNodes> Qe = Eigen::Matrix<float, nNodes, nNodes>::Zero();

		const auto& gp = ShapeFunc::faceGaussPoints(faceIndex);
		const auto& w = ShapeFunc::faceWeights(faceIndex);

		for (int i = 0; i < ShapeFunc::nFaceGP; ++i)
		{
			std::array<float, nFaceNodes> N_face;
			std::array<Eigen::Vector2f, nFaceNodes> dN_dxi_eta;

			for (int a = 0; a < nFaceNodes; ++a)
			{
				N_face[a] = ShapeFunc::N_face(gp[i], a, faceIndex);
				dN_dxi_eta[a] = ShapeFunc::dNdxi_face(gp[i], a, faceIndex); // returns 2D derivatives in 
			}

			// Get node positions for this face
			std::vector<Node> faceNodes;
			for (int a = 0; a < nFaceNodes; ++a)
				faceNodes.push_back(mesh_.nodes()[elem.nodes[ShapeFunc::faceConnectivity[faceIndex][a]]]);

			// Compute surface Jacobian
			Eigen::Matrix<float, 2, 3> JFace = Eigen::Matrix<float, 2, 3>::Zero();
			for (int a = 0; a < nFaceNodes; ++a)
			{
				Eigen::Vector3f nodePos(faceNodes[a].x, faceNodes[a].y, faceNodes[a].z);
				//Eigen::MatrixXf var = dN_dxi_eta[a] * nodePos.transpose();
				//std::cout << "J" << a << ":\n" << var << std::endl;
				JFace += dN_dxi_eta[a] * nodePos.transpose();
				//std::cout << "Jface:\n" << JFace << std::endl;;
			}

			float detJ = (JFace.row(0).cross(JFace.row(1))).norm();

			float weight = w[i][0] * w[i][1] * detJ;

			// Assemble local face contribution
			for (int a = 0; a < nFaceNodes; ++a)
			{
				int globalA = ShapeFunc::faceConnectivity[faceIndex][a];
				for (int b = 0; b < nFaceNodes; ++b)
				{
					int globalB = ShapeFunc::faceConnectivity[faceIndex][b];
					Qe(globalA, globalB) += N_face[a] * N_face[b] * weight;
				}
			}
		}

		return Qe;
	}
	
	template <typename ShapeFunc>
	Eigen::Matrix3f calculateJ(const Element& elem, const std::vector<Eigen::Vector3f> dNdxi)
	{
		Eigen::Matrix3f J = Eigen::Matrix3f::Zero();

		for (int a = 0; a < ShapeFunc::nNodes; ++a)
		{
			const Node& n = mesh_.nodes()[elem.nodes[a]];
			Eigen::Vector3f nodePos(n.x, n.y, n.z);
			J += dNdxi[a] * nodePos.transpose();
			/*J(0, 0) += dN(0) * mesh_.nodes()[elem.nodes[a]].x;
			J(0, 1) += dN(0) * mesh_.nodes()[elem.nodes[a]].y;
			J(0, 2) += dN(0) * mesh_.nodes()[elem.nodes[a]].z;

			J(1, 0) += dN(1) * mesh_.nodes()[elem.nodes[a]].x;
			J(1, 1) += dN(1) * mesh_.nodes()[elem.nodes[a]].y;
			J(1, 2) += dN(1) * mesh_.nodes()[elem.nodes[a]].z;

			J(2, 0) += dN(2) * mesh_.nodes()[elem.nodes[a]].x;
			J(2, 1) += dN(2) * mesh_.nodes()[elem.nodes[a]].y;
			J(2, 2) += dN(2) * mesh_.nodes()[elem.nodes[a]].z;*/
		}
		return J;
	}

	const std::vector<long>& nodeMap() { return nodeMap_; }
	const std::vector<long>& validNodes() { return validNodes_; }
	long nNonDirichlet() { return nNonDirichlet_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& M() { return M_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& K() { return K_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Q() { return Q_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Fint() { return Fint_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& FintElem() { return FintElem_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Fconv() { return Fconv_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Fk() { return Fk_; }
	const Eigen::VectorXf& Fflux() { return Fflux_; }
	const Eigen::VectorXf& Fq() { return Fq_; }

private:

	const Mesh& mesh_;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<long> nodeMap_;
	long nNonDirichlet_ = 0;
	std::vector<long> validNodes_;

	//Eigen::Matrix3f J_; // Jacobian of our current element

	Eigen::SparseMatrix<float, Eigen::RowMajor> M_; // Thermal Mass Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> K_; // Thermal Conductivity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> Q_; // Convection Matrix -- should have the same structure as Me just gets scaled by htc instead of vhc

	// Internal nodal heat generation (aka laser). Its size is nNodes x nNodes. Becomes a vector once post multiplied by a 
	// vector dictating the fluence experienced at each node. 
	Eigen::SparseMatrix<float, Eigen::RowMajor> Fint_; 

	// forcing function due to irradiance when using elemental fluence rate. Its size is nNodes x nElems. Becomes vector once post multiplied
	// by a vector dictating the fluence experienced by each element
	Eigen::SparseMatrix<float, Eigen::RowMajor> FintElem_; 

	// forcing function due to convection on dirichlet node. Size is nNodes x nNodes. Becomes a vector once post multiplied
	// by a vector specifying the fixed temperature at each element.
	Eigen::SparseMatrix<float, Eigen::RowMajor> Fconv_;

	// Forcing Function due to conductivity matrix on dirichlet nodes. Stored as a matrix but becomes vector once multiplied by nodal temperatures
	Eigen::SparseMatrix<float, Eigen::RowMajor> Fk_; 

	Eigen::VectorXf Fflux_; // forcing function due to constant heatFlux boundary
	Eigen::VectorXf Fq_; // forcing function due to ambient temperature

	template <typename ShapeFunc>
	void buildMatricesT()
	{
		setNodeMap();
		resetMatrices<ShapeFunc>();
		std::vector<Element> elements = mesh_.elements();
		long nElem = mesh_.elements().size();
		for (int elemIdx = 0; elemIdx < mesh_.elements().size(); elemIdx++)
		{
			Element elem = elements[elemIdx];
			applyElement<ShapeFunc>(elem, elemIdx);
		}
		for (int f = 0; f < mesh_.boundaryFaces().size(); f++)
		{
			applyBoundary<ShapeFunc>(mesh_.boundaryFaces()[f]);
		}
		// compress all sparse matrices
		K_.makeCompressed();
		M_.makeCompressed();
		Q_.makeCompressed();
		Fint_.makeCompressed();
		FintElem_.makeCompressed();
		Fconv_.makeCompressed();
		Fk_.makeCompressed();
	}

};