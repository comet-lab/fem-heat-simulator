#include "Mesh.hpp"


Mesh::Mesh()
{
}

Mesh::Mesh(std::vector<Node> nodes, std::vector<Element> elements, std::vector<BoundaryFace> boundaryFaces)
{
	setNodes(nodes);
	setElements(elements);
	setBoundaryFaces(boundaryFaces);
    computeBoundingBoxes();
}

void Mesh::setNodes(std::vector<Node> nodes)
{
	nodes_ = nodes;
}

void Mesh::setBoundaryFaces(std::vector<BoundaryFace> boundaryFaces)
{
	boundaryFaces_ = boundaryFaces;
}

void Mesh::setElements(std::vector<Element> elements)
{
	elements_ = elements;
	setOrderAndShape();
}

void Mesh::setOrderAndShape()
{
	int nNodesPerElem = elements_[0].nodes.size();
	switch (nNodesPerElem)
	{
	case 4:
		order_ = GeometricOrder::LINEAR;
		elementShape_ = TETRAHEDRAL;
		break;
	case 8:
		order_ = LINEAR;
		elementShape_ = HEXAHEDRAL;
		break;
	case 10:
		order_ = QUADRATIC;
		elementShape_ = TETRAHEDRAL;
		break;
	case 20:
		order_ = QUADRATIC;
		elementShape_ = HEXAHEDRAL;
		break;
	default:
		throw std::runtime_error("Invalid number of nodes per element");
	}

	for (int e = 0; e < elements_.size(); e++)
	{
		if (elements_[e].nodes.size() != nNodesPerElem)
		{
			throw std::runtime_error("Each element should have the same number of nodes");
		}
	}
}

long Mesh::findPosInMesh(const std::array<float,3>& p, std::array<float,3>& xi) const
{
    for (int e = 0; e < elements_.size(); e++)
    {
        auto elem = elements_[e];
        auto& box = elementBoxes_[e];

        if (p[0] < box.xmin || p[0] > box.xmax ||
            p[1] < box.ymin || p[1] > box.ymax ||
            p[2] < box.zmin || p[2] > box.zmax)
            continue;
        // Now do parametric inversion
        Eigen::Vector3f pVec;
        pVec << p[0], p[1], p[2];
        xi = computeXiCoordinates(pVec, elem.nodes);

        if (insideReferenceElement(xi))
            return e;
    }

    return -1; // not found
}

bool Mesh::insideReferenceElement(std::array<float, 3> xi) const
{
    const float tol = 1e-6f;
    if (elementShape_ == TETRAHEDRAL) {
        // xi barycentric coords for tet: 
        if (xi[0] < -tol || xi[1] < -tol || xi[2] < -tol) return false;
        if (xi[0] + xi[1] + xi[2] > 1.0f + tol) return false;
        return true;
    }
    else {
        // Hex:
        if (xi[0] < -1.0f - tol || xi[0] > 1.0f + tol) return false;
        if (xi[1] < -1.0f - tol || xi[1] > 1.0f + tol) return false;
        if (xi[2] < -1.0f - tol || xi[2] > 1.0f + tol) return false;
        return true;
    }
}

void Mesh::computeBoundingBoxes() {
    elementBoxes_.resize(elements_.size());
    for (int e = 0; e < elements_.size(); e++) {
        auto elemNodes = elements_[e].nodes;
        auto& box = elementBoxes_[e];

        box.xmin = box.ymin = box.zmin = std::numeric_limits<float>::max();
        box.xmax = box.ymax = box.zmax = std::numeric_limits<float>::lowest();

        for (auto n : elemNodes) {
            auto node = nodes_[n];
            box.xmin = std::min(box.xmin, node.x);
            box.xmax = std::max(box.xmax, node.x);
            box.ymin = std::min(box.ymin, node.y);
            box.ymax = std::max(box.ymax, node.y);
            box.zmin = std::min(box.zmin, node.z);
            box.zmax = std::max(box.zmax, node.z);
        }
    }
}

std::array<float,3> Mesh::computeXiCoordinates(const Eigen::Vector3f& p,const std::vector<long>& nodeList) const
{
    if ((order_ == LINEAR) && (elementShape_ == TETRAHEDRAL))
        return computeXiCoordinatesT<ShapeFunctions::TetLinear>(p, nodeList);
    else if ((order_ == LINEAR) && (elementShape_ == HEXAHEDRAL))
        return  computeXiCoordinatesT<ShapeFunctions::HexLinear>(p, nodeList);
    else
        throw std::runtime_error("cannot compute xi coordinates for unknown shape-order combination");

}

Mesh Mesh::buildCubeMesh(std::array<float, 3> tissueSize, std::array<long, 3> nodesPerAxis, std::array<BoundaryType, 6> bc)
{
    std::vector<float> xPos(nodesPerAxis[0]); 
    std::vector<float> yPos(nodesPerAxis[1]);
    std::vector<float> zPos(nodesPerAxis[2]);
    float x = -tissueSize[0] / 2.0f;
    float dx = tissueSize[0] / (nodesPerAxis[0] - 1.0f);
    for (int i = 0; i < nodesPerAxis[0]; i++, x+=dx)
    {
        xPos[i] = x;
    }
    float y = -tissueSize[1] / 2.0f;
    float dy = tissueSize[1] / (nodesPerAxis[1] - 1.0f);
    for (int i = 0; i < nodesPerAxis[1]; i++, y += dy)
    {
        yPos[i] = y;
    }
    float z = 0;
    float dz = tissueSize[2] / (nodesPerAxis[2] - 1.0f);
    for (int i = 0; i < nodesPerAxis[2]; i++, z += dz)
    {
        zPos[i] = z;
    }
    return buildCubeMesh(xPos, yPos, zPos, bc);
}


Mesh Mesh::buildCubeMesh(const std::vector<float>& xPos, const std::vector<float>& yPos,const std::vector<float>& zPos, std::array<BoundaryType, 6> bc)
{
    const long Nx = xPos.size();
    const long Ny = yPos.size();
    const long Nz = zPos.size();

    const long numNodes = Nx * Ny * Nz;

    std::vector<Node> nodes;
    nodes.reserve(numNodes);
    for (long k = 0; k < Nz; k++) {
        for (long j = 0; j < Ny; j++) {
            for (long i = 0; i < Nx; i++) {
                nodes.push_back(Node{ xPos[i], yPos[j] , zPos[k] });
            }
        }
    }

    // Helper to convert (i,j,k) -> flattened node index
    auto nodeID = [&](long i, long j, long k) {
        return i + j * Nx + k * Nx * Ny;
        };

    std::vector<Element> elements;
    elements.reserve((Nx - 1) * (Ny - 1) * (Nz - 1));

    for (long k = 0; k < Nz - 1; k++) {
        for (long j = 0; j < Ny - 1; j++) {
            for (long i = 0; i < Nx - 1; i++) {

                Element elem;
                elem.nodes = {
                    nodeID(i,     j,     k),
                    nodeID(i + 1, j,     k),
                    nodeID(i + 1, j + 1, k),
                    nodeID(i,     j + 1, k),
                    nodeID(i,     j,     k + 1),
                    nodeID(i + 1, j,     k + 1),
                    nodeID(i + 1, j + 1, k + 1),
                    nodeID(i,     j + 1, k + 1)                    
                };
                elements.push_back(elem);
            }
        }
    }

    // Helper to convert (i,j,k) -> flattened element index
    auto elemID = [&](long i, long j, long k) {
        return i + j * (Nx - 1) + k * (Nx - 1) * (Ny - 1);
        };

    std::vector<BoundaryFace> boundaryFaces;

    // ---------- Top face (z = 0), local face 0 ----------
    long k0 = 0;
    for (long j = 0; j < Ny - 1; j++) {
        for (long i = 0; i < Nx - 1; i++) {
            BoundaryFace bf;
            bf.elemID = elemID(i, j, k0);
            bf.localFaceID = 0;
            bf.nodes = {
                nodeID(i,   j,   k0), // 0 
                nodeID(i, j + 1, k0), // 3
                nodeID(i + 1,   j + 1, k0), // 2
                nodeID(i + 1, j,   k0) // 1
            };
            bf.type = bc[0];
            boundaryFaces.push_back(bf);
        }
    }

    // ---------- Bottom face (z = max), local face 1 ----------
    long kTop = Nz - 2;
    for (long j = 0; j < Ny - 1; j++) {
        for (long i = 0; i < Nx - 1; i++) {
            BoundaryFace bf;
            bf.elemID = elemID(i, j, kTop);
            bf.localFaceID = 1;
            bf.nodes = {
                nodeID(i,   j,   kTop + 1), // 4
                nodeID(i + 1, j,   kTop + 1), // 5
                nodeID(i + 1,   j + 1, kTop + 1), // 6
                nodeID(i, j + 1, kTop + 1), //7
            };
            bf.type = bc[1];
            boundaryFaces.push_back(bf);
        }
    }

    // ---------- Back face (y = -half), local face 2 ----------
    long j0 = 0;
    for (long k = 0; k < Nz - 1; k++) {
        for (long i = 0; i < Nx - 1; i++) {
            BoundaryFace bf;
            bf.elemID = elemID(i, j0, k);
            bf.localFaceID = 2;
            bf.nodes = {
                nodeID(i,   j0,   k), // 0
                nodeID(i + 1, j0,   k), // 1
                nodeID(i + 1, j0,   k + 1), // 5
                nodeID(i,   j0,   k + 1) // 4
            };
            bf.type = bc[2];
            boundaryFaces.push_back(bf);
        }
    }

    // ---------- Front face (y = max), local face 3 ----------
    long jTop = Ny - 2;
    for (long k = 0; k < Nz - 1; k++) {
        for (long i = 0; i < Nx - 1; i++) {
            BoundaryFace bf;
            bf.elemID = elemID(i, jTop, k);
            bf.localFaceID = 3;
            bf.nodes = {
                nodeID(i,     jTop + 1,   k), // 3
                nodeID(i,     jTop + 1,   k + 1), // 7
                nodeID(i + 1, jTop + 1,   k + 1), // 6
                nodeID(i + 1,   jTop + 1,   k) // 2
            };
            bf.type = bc[3];
            boundaryFaces.push_back(bf);
        }
    }

    // ---------- Left face (x = -half), local face 4 ----------
    long i0 = 0;
    for (long k = 0; k < Nz - 1; k++) {
        for (long j = 0; j < Ny - 1; j++) {
            BoundaryFace bf;
            bf.elemID = elemID(i0, j, k);
            bf.localFaceID = 4;
            bf.nodes = {
                nodeID(i0,   j,     k), // 0
                nodeID(i0,   j,     k + 1), // 4
                nodeID(i0,   j + 1, k + 1), // 7 
                nodeID(i0,   j + 1,     k) // 3
            };
            bf.type = bc[4];
            boundaryFaces.push_back(bf);
        }
    }

    // ---------- Right face (x = max), local face 5 ----------
    long iTop = Nx - 2;
    for (long k = 0; k < Nz - 1; k++) {
        for (long j = 0; j < Ny - 1; j++) {
            BoundaryFace bf;
            bf.elemID = elemID(iTop, j, k);
            bf.localFaceID = 5;
            bf.nodes = {
                nodeID(iTop + 1,   j,     k), // 1
                nodeID(iTop + 1,   j + 1, k), // 2 
                nodeID(iTop + 1,   j + 1, k + 1), // 6
                nodeID(iTop + 1,   j,     k + 1) // 5
            };
            bf.type = bc[5];
            boundaryFaces.push_back(bf);
        }
    }

    return Mesh(nodes, elements, boundaryFaces);
}

