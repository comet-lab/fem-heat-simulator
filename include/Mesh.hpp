#pragma once
#include <vector> 
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <chrono>
#include <stdexcept>

enum BoundaryType {
	NONE,
	HEATSINK,
	FLUX,
	CONVECTION
};

struct Node {
	double x;
	double y;
	double z;
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

// =============================
// FACE CONNECTIVITY NAMESPACE
// =============================
namespace FaceConnectivity {
	extern const std::array<std::array<int, 8>, 6> HEX20;
	extern const std::array<std::array<int, 3>, 4> TET4;
	extern const std::array<std::array<int, 6>, 4> TET10;
}

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
	
	Mesh();
	Mesh(std::vector<Node> nodes, std::vector<Element> elements, std::vector<BoundaryFace> boundaryFaces);

	void setOrderAndShape(); // based on nodes per element, we can determine what the order and shape are
	void setElements(std::vector<Element> elements); // sets elements and the calls setOrderAndShape
	void setNodes(std::vector<Node> nodes); // sets nodes
	void setBoundaryFaces(std::vector<BoundaryFace> boundaryFaces); // sets boundarFaces and then calls setNodeMap

	const std::vector<Node>& nodes() const { return nodes_; }
	const std::vector<Element>& elements() const { return elements_; }
	const std::vector<BoundaryFace>& boundaryFaces() const { return boundaryFaces_; }

private:
	std::vector<Node> nodes_; // nodes in mesh
	std::vector<Element> elements_; // elements in mesh
	std::vector<BoundaryFace> boundaryFaces_; // boundary faces in mesh
	GeometricOrder order_ = LINEAR; 
	Shape elementShape_ = HEXAHEDRAL;
};