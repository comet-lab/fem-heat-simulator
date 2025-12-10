#pragma once
#include <algorithm>
#include <numeric>
#include <chrono>
#include <math.h>
#include <stdexcept>
#include "ShapeFunctions/HexLinear.hpp"
#include "ShapeFunctions/TetLinear.hpp"
#include "ShapeFunctions/TetQuadratic.hpp"
#include <iostream>
#include <string>

enum BoundaryType {
	FLUX,
	CONVECTION,
	HEATSINK
};

struct Node {
	float x;
	float y;
	float z;
};

struct Element {
	std::vector<long> nodes; // index in node list used in element
};

struct BoundaryFace {
	std::vector<long> nodes;
	long elemID;
	int localFaceID;
	BoundaryType type;
	double value = 0.0;  // e.g., h or q
};

enum GeometricOrder {
	LINEAR,
	QUADRATIC
};

enum Shape {
	TETRAHEDRAL,
	HEXAHEDRAL
};

class Mesh {

public:
	//EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Mesh();
	Mesh(std::vector<Node> nodes, std::vector<Element> elements, std::vector<BoundaryFace> boundaryFaces);

	void setOrderAndShape(); // based on nodes per element, we can determine what the order and shape are
	void setElements(std::vector<Element> elements); // sets elements and the calls setOrderAndShape
	void setNodes(std::vector<Node> nodes); // sets nodes
	void setBoundaryFaces(std::vector<BoundaryFace> boundaryFaces); // sets boundarFaces and then calls setNodeMap

	const std::vector<Node>& nodes() const { return nodes_; }
	const std::vector<Element>& elements() const { return elements_; }
	const std::vector<BoundaryFace>& boundaryFaces() const { return boundaryFaces_; }
	const GeometricOrder order() const { return order_; }
	const Shape elementShape() const { return elementShape_; }
	long findPosInMesh(const std::array<float, 3>& p, std::array<float,3>& xi) const;
	bool insideReferenceElement(std::array<float, 3> xi) const;
	void computeBoundingBoxes();
	std::array<float,3> computeXiCoordinates(const Eigen::Vector3f& p, const std::vector<long>& nodeList) const;
	template <typename ShapeFunc>
	std::array<float,3> computeXiCoordinatesT(const Eigen::Vector3f& p, const std::vector<long>& nodeList) const;

	static Mesh buildCubeMesh(std::array<float, 3> tissueSize, std::array<long, 3> nodesPerAxis, std::array<BoundaryType,6> bc);
	static Mesh buildCubeMesh(const std::vector<float>& xPos, const std::vector<float>& yPos, const std::vector<float>& zPos, std::array<BoundaryType, 6> bc);

	

private:
	

	struct ElementBoundingBox {
		float xmin, xmax;
		float ymin, ymax;
		float zmin, zmax;
	};

	std::vector<Node> nodes_; // nodes in mesh
	std::vector<Element> elements_; // elements in mesh
	std::vector<BoundaryFace> boundaryFaces_; // boundary faces in mesh
	std::vector<ElementBoundingBox> elementBoxes_;
	GeometricOrder order_ = LINEAR; 
	Shape elementShape_ = HEXAHEDRAL;
};

	template<typename ShapeFunc>
	inline std::array<float,3> Mesh::computeXiCoordinatesT(const Eigen::Vector3f& p, const std::vector<long>& nodeList) const
	{
		std::array<float, 3> xi = { 0.0f, 0.0f, 0.0f }; // initial guess
		for (int iter = 0; iter < 20; iter++) {
			// Compute shape functions Ni and their derivatives
			Eigen::Vector<float, ShapeFunc::nNodes> N;
			Eigen::Matrix<float, ShapeFunc::nNodes, 3> dNdxi;
			for (int A = 0; A < ShapeFunc::nNodes; A++)
			{
				N(A) = ShapeFunc::N(xi,A);
				dNdxi.row(A) = ShapeFunc::dNdxi(xi, A);
			}

			// Compute residual r = X(xi) - p
			Eigen::Matrix<float, 3, ShapeFunc::nNodes> nodePos;
			for (int A = 0; A < ShapeFunc::nNodes; A++) {
				nodePos.col(A) << nodes_[nodeList[A]].x, nodes_[nodeList[A]].y, nodes_[nodeList[A]].z;
			}
			Eigen::Vector3f r = (nodePos * N) - p;

			// Build Jacobian matrix J
			Eigen::Matrix3f J = nodePos * dNdxi;

			// Solve J * delta = -r
			Eigen::Vector3f delta = J.colPivHouseholderQr().solve(-r);

			xi[0] += delta(0);
			xi[1] += delta(1);
			xi[2] += delta(2);

			// Convergence?
			if (delta.norm() < 1e-10) break;
		}
		return xi;
	}
