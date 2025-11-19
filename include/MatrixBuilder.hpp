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

struct GlobalMatrices
{
	std::vector<long> nodeMap_;
	long nNonDirichlet_ = 0;
	std::vector<long> validNodes_;
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
};

class MatrixBuilder
{
public:

	MatrixBuilder() {}
	MatrixBuilder(const Mesh& mesh) { setMesh(mesh); }
	~MatrixBuilder() {} // Destructor should not clear mesh because we don't allocate it in the class


	/*
	* @brief sets the mesh and then calls buildMatrices()
	* @param mesh a constant reference to a mesh object
	*/
	void buildMatrices(const Mesh& mesh)
	{
		setMesh(mesh);
		buildMatrices();
	}


	/*
	* @brief selects the ShapeFunction to use to build the matrices based on the mesh
	*/
	void buildMatrices()
	{
		if (!mesh_)
			throw std::runtime_error("Mesh not set");

		if ((mesh_->order() == GeometricOrder::LINEAR) && (mesh_->elementShape() == Shape::HEXAHEDRAL))
		{
			buildMatricesT<ShapeFunctions::HexLinear>();
		}
		else if ((mesh_->order() == GeometricOrder::LINEAR) && (mesh_->elementShape() == Shape::TETRAHEDRAL))
		{
			buildMatricesT<ShapeFunctions::TetLinear>();
		}
	}

	void setMesh(const Mesh& mesh)
	{
		mesh_ = &mesh;
	}

	/*
	* @brief goes through the mesh and labels dirichlet nodes so that we can properly size our global matrices and build them
	*/
	void setNodeMap()
	{
		// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
		// A value of -1 at index i, indicates that global node i is a dirichlet node. 
		nodeMap_.resize(mesh_->nodes().size());
		std::fill(nodeMap_.begin(), nodeMap_.end(), 0);
		// First go through the boundary faces and set all nodes on a heatsink (dirichlet) face to -1
		for (BoundaryFace face : mesh_->boundaryFaces())
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
		for (int i = 0; i < mesh_->nodes().size(); i++)
		{
			if (nodeMap_[i] == 0)
			{
				nodeMap_[i] = nNonDirichlet_;
				validNodes_.push_back(i);
				nNonDirichlet_++;
			}
		}
	}

	/*
	* @brief sets the size of all the global matrices based on the mesh so that we aren't resizing while building
	*/
	template <typename ShapeFunc>
	void resetMatrices()
	{
		// The number of non-zeros per column this should be a safe number
		int nRelatedNodes = ShapeFunc::nNodes * ShapeFunc::nNodes * ShapeFunc::nNodes;
		//  number of non-dirichlet nodes
		// total number of nodes
		int nNodes = mesh_->nodes().size();
		int nodesPerElem = ShapeFunc::nNodes;
		// Initialize matrices so that we don't have to resize them later
		FintElem_ = Eigen::SparseMatrix<float>(nNonDirichlet_, mesh_->elements().size());
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

	/*
	* @brief applies the contribution of the given element to the global matrices
	* @param elem of type Element
	* @param elemIdx the index of the element in the mesh element list
	*/
	template <typename ShapeFunc>
	void applyElement(Element elem, long elemIdx)
	{
		int nodesPerElem = ShapeFunc::nNodes;
		long matrixRow = 0;
		long matrixCol = 0;
		Eigen::MatrixXf Me = calculateIntNaNb<ShapeFunc>(elem);
		Eigen::MatrixXf Fe = calculateIntNaNb<ShapeFunc>(elem);
		Eigen::MatrixXf Ke = calculateIntdNadNb<ShapeFunc>(elem);

		for (int A = 0; A < nodesPerElem; A++)
		{
			Node currNode = mesh_->nodes()[elem.nodes[A]];
			matrixRow = nodeMap_[elem.nodes[A]];
			if (matrixRow >= 0)
			{
				// handle node-neighbor interactions

				for (int B = 0; B < nodesPerElem; B++)
				{
					long neighborIdx = elem.nodes[B];
					matrixCol = nodeMap_[neighborIdx];
					Node neighbor = mesh_->nodes()[neighborIdx];
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


	/*
	* @brief applies the contribution of the given element face to the global matrices
	* @param face of type BoundaryFace
	*/
	template <typename ShapeFunc>
	void applyBoundary(BoundaryFace face)
	{
		if (face.type == CONVECTION)
		{
			long matrixRow = 0;
			long matrixCol = 0;
			Element elem = mesh_->elements()[face.elemID];
			int nodesPerElem = ShapeFunc::nNodes;
			Eigen::VectorXf Feflux = calculateFaceIntNa<ShapeFunc>(elem, face.localFaceID, 1);
			Eigen::MatrixXf FeConv = calculateFaceIntNaNb<ShapeFunc>(elem, face.localFaceID);
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
						Node neighbor = mesh_->nodes()[neighborIdx];
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
			Element elem = mesh_->elements()[face.elemID];
			long matrixRow = 0;
			Eigen::VectorXf Feflux = calculateFaceIntNa<ShapeFunc>(elem, face.localFaceID, 1);		
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

	/*
	* @brief calculates the integral (N * N' |J| dV) for the given element
	* @param elem current element
	* 
	* This function is used to calculate the Thermal Mass Matrix of a given element (Me) as well
	* as the local function Fe used for nodal or elemental heat generation
	*/
	template <typename ShapeFunc>
	Eigen::MatrixXf calculateIntNaNb(const Element& elem) 
	{

		Eigen::MatrixXf Me = Eigen::MatrixXf::Zero(ShapeFunc::nNodes, ShapeFunc::nNodes);

		const std::vector<std::array<float, 3>>& w = ShapeFunc::weights();

		for (int i = 0; i < ShapeFunc::nGP; i++)
		{
			Eigen::Matrix<float, 3, ShapeFunc::nNodes> dNdxi = dNdxiCache_[i];

			// Jacobian and determinant (simplified)
			Eigen::Matrix3f J = calculateJ<ShapeFunc>(elem, dNdxi);
			float detJ = J.determinant();
			float weight = w[i][0] * w[i][1] * w[i][2] * detJ;


			Me += NaNbCache_[i] * weight;
		}

		return Me;
	}

	/*
	* @brief calculates the integral ( (Jinv * dN/dxi)^T * (Jinv * dN/dxi) |J| dV) for the given element
	* @param elem current element
	*
	* This function is used to calculate the Conductivity matrix of an element (Ke)
	*/
	template <typename ShapeFunc>
	Eigen::MatrixXf calculateIntdNadNb(const Element& elem)
	{
		Eigen::MatrixXf Ke = Eigen::MatrixXf::Zero(ShapeFunc::nNodes, ShapeFunc::nNodes);

		const std::vector<std::array<float, 3>>& w = ShapeFunc::weights();

		for (int i = 0; i < ShapeFunc::nGP; ++i)
		{
			Eigen::Matrix<float, 3, ShapeFunc::nNodes> dNdxi = dNdxiCache_[i];

			// Compute Jacobian and inverse
			Eigen::Matrix3f J = calculateJ<ShapeFunc>(elem, dNdxi);
			Eigen::Matrix3f Jinv = J.inverse();
			float detJ = J.determinant();

			// Derivatives in physical coordinates: 3 × nNodes
			Eigen::Matrix<float, 3, ShapeFunc::nNodes> dNdx = Jinv * dNdxi;
			// Weight
			float weight = w[i][0] * w[i][1] * w[i][2] * detJ;
			// Ke contribution: (nNodes × 3)(3 × nNodes) = nNodes × nNodes
			Ke += dNdx.transpose() * dNdx * weight;
				
		}

		return Ke;
	}

	/*
	* @brief calculates the integral (N * q J_s| dS) for the given face
	* @param elem current element
	* @param faceIndex face on the element
	* @param q if we want to scale by a coefficient or input value (pretty much unused)
	*
	* This function is used to calculate the influence of Neumann boundary conditions
	*/
	template <typename ShapeFunc>
	Eigen::MatrixXf calculateFaceIntNa(const Element& elem, int faceIndex, float q)
	{
		Eigen::Matrix<float, ShapeFunc::nNodes, 1> Fe = Eigen::Matrix<float, ShapeFunc::nNodes, 1>::Zero();

		//const auto& gp = ShapeFunc::faceGaussPoints(faceIndex);
		const auto& w = ShapeFunc::faceWeights(faceIndex);

		for (int i = 0; i < ShapeFunc::nFaceGP; ++i)
		{
			Eigen::Vector<float, ShapeFunc::nFaceNodes> N_face = NFaceCache_[faceIndex][i];
			Eigen::Matrix<float, 2, ShapeFunc::nFaceNodes> dN_dxi_eta = dNdxiFaceCache_[faceIndex][i];

			// Compute 2x3 surface Jacobian
			Eigen::Matrix<float, 2, 3> JFace = calculateJFace<ShapeFunc>(elem,faceIndex,dN_dxi_eta);

			// Surface determinant: norm of cross product of rows
			float detJ = (JFace.row(0).cross(JFace.row(1))).norm();

			float weight = w[i][0] * w[i][1] * detJ;

			for (int a = 0; a < ShapeFunc::nFaceNodes; ++a)
				Fe(ShapeFunc::faceConnectivity[faceIndex][a]) += N_face[a] * q * weight;
		}

		return Fe;
	}

	/*
	* @brief calculates the integral (N * N' |J_s| dS) for the given face
	* @param elem current element
	* @param faceIndex face on the element
	*
	* This function is used to calculate the influence of Neumann boundary conditions that are influenced
	* by a value at each node. For example convection is influenced by temperature at each node so this would
	* be used to calculate \int (N*h*T |J| dS) 
	*/
	template <typename ShapeFunc>
	Eigen::MatrixXf calculateFaceIntNaNb(const Element& elem, int faceIndex)
	{
		constexpr int nNodes = ShapeFunc::nNodes;
		constexpr int nFaceNodes = ShapeFunc::nFaceNodes;

		Eigen::Matrix<float, nNodes, nNodes> Qe = Eigen::Matrix<float, nNodes, nNodes>::Zero();

		const auto& w = ShapeFunc::faceWeights(faceIndex);

		for (int i = 0; i < ShapeFunc::nFaceGP; ++i)
		{
			Eigen::Vector<float, ShapeFunc::nFaceNodes> N_face = NFaceCache_[faceIndex][i];
			Eigen::Matrix<float, 2, ShapeFunc::nFaceNodes> dN_dxi_eta = dNdxiFaceCache_[faceIndex][i];

			// Compute 2x3 surface Jacobian
			Eigen::Matrix<float, 2, 3> JFace = calculateJFace<ShapeFunc>(elem, faceIndex, dN_dxi_eta);

			// Surface determinant: norm of cross product of rows
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
	
	/*
	* @brief precomputes the volume jacobian for an element
	* @param elem is of type Element and is the current element we are working on
	* @param dNdxi is the shape function derivative over the volume evaluated at the current gaussian integration point
	*/
	template <typename ShapeFunc>
	Eigen::Matrix3f calculateJ(const Element& elem, const Eigen::Matrix<float,3,Eigen::Dynamic>& dNdxi)
	{
		Eigen::Matrix<float, ShapeFunc::nNodes, 3> nodePoses;
		for (int a = 0; a < ShapeFunc::nNodes; ++a)
		{
			const Node& n = mesh_->nodes()[elem.nodes[a]];
			nodePoses.row(a) << n.x, n.y, n.z;
		}
		Eigen::Matrix3f J = dNdxi * nodePoses;
		return J;
	}

	/*
	* @brief precomputes the face jacobian for a face on an element
	* @param elem is of type Element and is the current element we are working on
	* @param faceIndex is the face number of the element
	* @param dNdxi is the shape function derivative over the surface evaluated at the current gaussian integration point
	*/
	template <typename ShapeFunc>
	Eigen::Matrix<float, 2, 3> calculateJFace(const Element& elem, int faceIndex, const Eigen::Matrix<float, 2, Eigen::Dynamic>& dNdxi)
	{
		Eigen::Matrix<float, ShapeFunc::nFaceNodes, 3> nodePoses;
		for (int a = 0; a < ShapeFunc::nFaceNodes; ++a)
		{
			const Node& n = mesh_->nodes()[elem.nodes[ShapeFunc::faceConnectivity[faceIndex][a]]];
			nodePoses.row(a) << n.x, n.y, n.z;
		}
		Eigen::Matrix<float, 2, 3> JFace = dNdxi * nodePoses;
		return JFace;
	}

	/*
	* @brief precompute the shape functions at each gauss point, their derivatves, and outer products
	* 
	* Because we are assuming each element in the mesh will be the same element type (e.g. hex linear)
	* we can compute the value of the shape functions at each gaussian integration point ahead of time. 
	* The shape functions are defined in the parametric domain, meaning different element sizes or warps won't
	* affect the shape functions. The differences in element lengths or sizes will be imparted with the Jacobian
	* which isn't precomputed.
	* 
	* If we ever change to allow multiple element types within a mesh, we won't be able to precompute and 
	* we will have to calculate them in the integration loops above.
	*/
	template <typename ShapeFunc>
	void precomputeShapeFunctions()
	{
		const std::vector<std::array<float, 3>>& gp = ShapeFunc::gaussPoints();
		NCache_.clear();
		NaNbCache_.clear();
		dNdxiCache_.clear();
		NFaceCache_.clear();
		dNdxiFaceCache_.clear();
		// pre-compute volume shape functions
		for (int i = 0; i < ShapeFunc::nGP; i++)
		{
			const auto& xi = gp[i];
			Eigen::Matrix<float,3,ShapeFunc::nNodes> dNdxi;
			Eigen::VectorXf Nvals(ShapeFunc::nNodes);
			for (int a = 0; a < ShapeFunc::nNodes; a++)
			{
				Nvals(a) = ShapeFunc::N(xi, a);
				dNdxi.col(a) = ShapeFunc::dNdxi(xi, a);
			}
			NaNbCache_.push_back(Nvals * Nvals.transpose());
			NCache_.push_back(Nvals);
			dNdxiCache_.push_back(dNdxi);
		}

		// pre-compute surface shape functions
		for (int f = 0; f < ShapeFunc::nFaces; f++)
		{
			const auto& gpFace = ShapeFunc::faceGaussPoints(f);

			std::vector<Eigen::VectorXf> temporaryNFace(ShapeFunc::nFaceGP);
			//std::vector<Eigen::Matrix<float, 2, ShapeFunc::nFaceNodes>> temporarydNdxi(ShapeFunc::nFaceGP);
			std::vector<Eigen::Matrix<float, 2, Eigen::Dynamic>> temporarydNdxi(ShapeFunc::nFaceGP);
			std::vector<Eigen::Matrix<float, ShapeFunc::nFaceNodes, ShapeFunc::nFaceNodes>> temporaryNaNbFace(ShapeFunc::nFaceGP);
			for (int i = 0; i < ShapeFunc::nFaceGP; ++i)
			{
				Eigen::Vector<float, ShapeFunc::nFaceNodes> N_face;
				Eigen::Matrix<float,2,ShapeFunc::nFaceNodes> dN_dxi_eta;

				for (int a = 0; a < ShapeFunc::nFaceNodes; ++a)
				{
					N_face(a) = ShapeFunc::N_face(gpFace[i], a, f);
					dN_dxi_eta.col(a) = ShapeFunc::dNdxi_face(gpFace[i], a, f);
				}
				temporaryNFace[i] = N_face;
				temporaryNaNbFace[i] = N_face * N_face.transpose();
				temporarydNdxi[i] = dN_dxi_eta;
			}
			NFaceCache_.push_back(temporaryNFace);
			dNdxiFaceCache_.push_back(temporarydNdxi);
		}
	}

	// public getters
	const std::vector<long>& nodeMap() const { return nodeMap_; }
	const std::vector<long>& validNodes() const { return validNodes_; }
	long nNonDirichlet() const { return nNonDirichlet_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& M() const { return M_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& K() const { return K_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Q() const { return Q_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Fint() const { return Fint_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& FintElem() const { return FintElem_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Fconv() const { return Fconv_; }
	const Eigen::SparseMatrix<float, Eigen::RowMajor>& Fk() const { return Fk_; }
	const Eigen::VectorXf& Fflux() const { return Fflux_; }
	const Eigen::VectorXf& Fq() const { return Fq_; }

private:

	const Mesh* mesh_ = nullptr;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<long> nodeMap_;
	long nNonDirichlet_ = 0;
	std::vector<long> validNodes_;

	/* Cached shape functions which are element independent */ 
	// volume shape functions
	std::vector<Eigen::VectorXf> NCache_; // Stores the shape functions for each node at each gauss point
	std::vector<Eigen::MatrixXf> NaNbCache_; // Stores the multiplication of N*N' for element shape functions at each gauss point
	std::vector<Eigen::Matrix<float, 3, Eigen::Dynamic>> dNdxiCache_; // Stores the shape function derivaties for each node at each gauss point
	// surface shape functions
	// outer std::vector is faceIdx, inner std::vector is gauss point, Eigen::Vector is face shape functions at gauss point on face
	// size is nFace x nGP x nFaceNodes
	std::vector<std::vector<Eigen::VectorXf>> NFaceCache_; 
	// Stores multiplication of N*N' for each face shape function at each gauss point
	std::vector<std::vector<Eigen::MatrixXf>> NaNbFaceCache_;
	// outer std::vector is faceIdx, inner std::vector is gauss point, Eigen::Matrix is face derivative shape functions at gauss point on face
	// size is nFace x nGp x (2 x nFaceNodes)
	std::vector<std::vector<Eigen::Matrix<float, 2, Eigen::Dynamic>>> dNdxiFaceCache_;

	/*Cache of global matrices used in assembly*/
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


	/*
	* @brief templated version of buildMatrices() for specific element type
	*/
	template <typename ShapeFunc>
	void buildMatricesT()
	{
		setNodeMap();
		resetMatrices<ShapeFunc>();
		precomputeShapeFunctions<ShapeFunc>();
		std::vector<Element> elements = mesh_->elements();
		long nElem = mesh_->elements().size();
		for (int elemIdx = 0; elemIdx < mesh_->elements().size(); elemIdx++)
		{
			Element elem = elements[elemIdx];
			applyElement<ShapeFunc>(elem, elemIdx);
		}
		for (int f = 0; f < mesh_->boundaryFaces().size(); f++)
		{
			applyBoundary<ShapeFunc>(mesh_->boundaryFaces()[f]);
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