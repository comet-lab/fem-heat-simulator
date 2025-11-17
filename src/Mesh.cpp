#include "Mesh.hpp"


Mesh::Mesh()
{
}

Mesh::Mesh(std::vector<Node> nodes, std::vector<Element> elements, std::vector<BoundaryFace> boundaryFaces)
{
	setNodes(nodes);
	setElements(elements);
	setBoundaryFaces(boundaryFaces);
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

Mesh Mesh::buildCubeMesh(std::array<float, 3> tissueSize, std::array<long, 3> nodesPerAxis, std::array<BoundaryType, 6> bc)
{
    const long Nx = nodesPerAxis[0];
    const long Ny = nodesPerAxis[1];
    const long Nz = nodesPerAxis[2];

    const long numNodes = Nx * Ny * Nz;

    std::vector<Node> nodes;
    nodes.reserve(numNodes);

    float dx = tissueSize[0] / (Nx - 1);
    float dy = tissueSize[1] / (Ny - 1);
    float dz = tissueSize[2] / (Nz - 1);

    float z = 0;
    for (long k = 0; k < Nz; k++, z += dz) {
        float y = -tissueSize[1] / 2.0f;
        for (long j = 0; j < Ny; j++, y += dy) {
            float x = -tissueSize[0] / 2.0f;
            for (long i = 0; i < Nx; i++, x += dx) {
                nodes.push_back(Node{ x, y, z });
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
                    nodeID(i,   j,   k),
                    nodeID(i + 1, j,   k),
                    nodeID(i, j + 1, k),
                    nodeID(i + 1,   j + 1, k),
                    nodeID(i,   j,   k + 1),
                    nodeID(i + 1, j,   k + 1),
                    nodeID(i, j + 1, k + 1),
                    nodeID(i + 1,   j + 1, k + 1)
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
                nodeID(i,   j,   k0),
                nodeID(i + 1, j,   k0),
                nodeID(i, j + 1, k0),
                nodeID(i + 1,   j + 1, k0)
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
                nodeID(i,   j,   kTop + 1),
                nodeID(i + 1, j,   kTop + 1),
                nodeID(i, j + 1, kTop + 1),
                nodeID(i + 1,   j + 1, kTop + 1)
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
                nodeID(i,   j0,   k),
                nodeID(i + 1, j0,   k),
                nodeID(i + 1, j0,   k + 1),
                nodeID(i,   j0,   k + 1)
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
                nodeID(i,   jTop + 1,   k),
                nodeID(i + 1, jTop + 1,   k),
                nodeID(i + 1, jTop + 1,   k + 1),
                nodeID(i,   jTop + 1,   k + 1)
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
                nodeID(i0,   j,   k),
                nodeID(i0,   j + 1, k),
                nodeID(i0,   j + 1, k + 1),
                nodeID(i0,   j,   k + 1)
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
                nodeID(iTop + 1,   j,   k),
                nodeID(iTop + 1,   j + 1, k),
                nodeID(iTop + 1,   j + 1, k + 1),
                nodeID(iTop + 1,   j,   k + 1)
            };
            bf.type = bc[5];
            boundaryFaces.push_back(bf);
        }
    }

    return Mesh(nodes, elements, boundaryFaces);
}

