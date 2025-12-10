#include "MEX_Utility.hpp"
#include <set>

class MexFunction : public matlab::mex::Function {

    /*
    * Use case MEX_Heat_Simulation(meshInfo,thermalInfo,settings);
    *   thermalInfo - struct containing temperature, fluenceRate, MUA, VHC, TC, HTC, FLUX, AmbientTemp
    *   settings - struct containing dt, time, alpha, silentMode, useGPU, useAllCPUs, resetIntegration
    *   laserInfo - struct containing either fluenceRate or laserPose, laserPower, beamWaist
    *   meshInfo - struct containing either sensorLocations and [(xpos, ypos, zpos, boundaryCond) OR (nodes,elements,boundaryFaces)]
    */

private:
    enum VarPlacement {
        THERMAL_INFO, SETTINGS, LASER_INFO, MESH_INFO
    };

    const std::vector<std::string> thermalFields = { "temperature","MUA","VHC","TC","HTC","flux","ambientTemp" };
    const std::vector<std::string> settingsFields = { "dt","time","alpha" }; // optional fields {"silentMode","useGPU","useAllCPUs","resetIntegration"}
    const std::vector<std::string> laserFields1 = { "fluenceRate" };
    const std::vector<std::string> laserFields3 = { "laserPose","laserPower","beamWaist","wavelength"};
    const std::vector<std::string> meshFields4 = { "nodes","elements","boundaryFaces","sensorLocations" };
    const std::vector<std::string> meshFields5 = { "xpos","ypos","zpos","boundaryConditions","sensorLocations" };

    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    FEM_Simulator simulator;
    Mesh mesh;
    std::ostringstream stream;
    /* Optional Parameters that are stored as class variables for repeat calls */
    bool silentMode = true;
    bool useAllCPUs = false;
    bool useGPU = false;
    bool buildMatrices = true; // build matrices will build global matrices and reset time integration
    bool resetIntegration = false; // uses current global matrices but resets time integration
    bool multiStep = false;
    bool saveSurfaceData = false;

    /* Required parameters extracted from inputs */
    float MUA, TC, HTC, VHC, flux, ambientTemp, alpha, dt, beamWaist, wavelength;
    Eigen::VectorXf Temp, fluenceRate;
    std::vector<float> simTime, laserPower;
    Eigen::MatrixXf laserPose;
    std::vector<std::array<float, 3>> sLocations;

    std::chrono::steady_clock::time_point timeRef_ = std::chrono::steady_clock::now();
        

public:
    /* Constructor for the class. */
    MexFunction()
    {
        matlabPtr = getEngine();
    }

    ~MexFunction()
    {
        return;
    }

    /* This is the gateway routine for the MEX-file. */
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        resetTimer();
        checkArguments(outputs, inputs); // check arguments will check for valid inputs and set the necessary class variables
        stream.str("");
        stream << "MEX: Inputs Validated -- Setting Up Model" << std::endl;
        displayOnMATLAB(matlabPtr,stream,silentMode);

        try {
            // apply mesh which was already constructed
            simulator.setMesh(mesh);
            // set thermal model info 
            simulator.setTemp(Temp);
            simulator.setMUA(MUA);
            simulator.setTC(TC);
            simulator.setVHC(VHC);
            simulator.setHTC(HTC);
            simulator.setHeatFlux(flux);
            simulator.setAmbientTemp(ambientTemp);

            if (!multiStep)
                simulator.setFluenceRate(fluenceRate);
            else
                simulator.setFluenceRate(laserPose.row(0), laserPower[0], beamWaist, wavelength);
            
            printDuration("MEX: Applied ThermalModel to Simulator ");

            // set time stepping params
            simulator.setDt(dt);
            simulator.setAlpha(alpha);
            printDuration("MEX: Set time stepping rate ");
            
            // Set sensor locations
            simulator.setSensorLocations(sLocations);
            printDuration("MEX: Set sensor locations ");
        }
        catch (const std::exception& e) {
            stream << "MEX: Error in Setup: " << std::endl;
            displayOnMATLAB(matlabPtr,stream, silentMode);
            displayError(matlabPtr,e.what());
            return;
        }
        printDuration("MEX: Set all simulation parameters -- ");

        std::set<long> surfaceNodesSet;
        std::vector<long> surfaceNodes;
        if (saveSurfaceData)
        {
            for (BoundaryFace bf : mesh.boundaryFaces())
            {
                for (long n : bf.nodes)
                {
                    surfaceNodesSet.insert(n);
                }
            }
            surfaceNodes = std::vector<long>(surfaceNodesSet.begin(), surfaceNodesSet.end()); 
        }
        
        // Set parallelization
        Eigen::setNbThreads(1);
#ifdef _OPENMP
        stream << "MEX: OPEMMP Enabled" << std::endl;
        displayOnMATLAB(matlabPtr, stream, silentMode);
        if (useAllCPUs) { //useAllCPUs is true
            stream << "MEX: Using all CPUs" << std::endl;
            displayOnMATLAB(matlabPtr, stream, silentMode);
            Eigen::setNbThreads(omp_get_num_procs()/2);
        }
#endif
        stream << "MEX: Number of threads: " << Eigen::nbThreads() << std::endl;
        displayOnMATLAB(matlabPtr, stream, silentMode);

        // Create global K M and F 
        if (buildMatrices) { // only need to create the KMF matrices the first time
            stream << "MEX: Initializing simulator ... " << std::endl;
            displayOnMATLAB(matlabPtr, stream, silentMode);
            buildMatrices = false;
            resetIntegration = true;
            try
            {
                simulator.buildMatrices();
            }
            catch (const std::exception& e) {
                stream << "MEX: Error in building Matrices" << std::endl;
                displayOnMATLAB(matlabPtr, stream, silentMode);
                displayError(matlabPtr, e.what());
                return;
            }
            printDuration("MEX: Matrices Built -- ");
        }
        
        if (resetIntegration)
        {
            resetIntegration = false;
            if (useGPU)
            {   
                int isEnabled = simulator.enableGPU();
                if (isEnabled)
                    stream << "MEX: GPU Enabled ... " << std::endl;
                else
                    stream << "MEX: GPU Disabled ..." << std::endl;
            }
            else
            {
                simulator.disableGPU();
                stream << "MEX: GPU Disabled ..." << std::endl;
            }
            displayOnMATLAB(matlabPtr, stream, silentMode);
            simulator.initializeTimeIntegration();

            printDuration("MEX: Time Integration Initialized -- ");
        }

        // Perform time stepping
        int numTimePoints = simTime.size();
        if (numTimePoints == 1)
        { // treat this as if someone is going from 0 -> simTime. 
            numTimePoints = 2;
            simTime.push_back(simTime[0]);
            simTime[0] = 0;
        }
        stream << "Time Stepping for " << numTimePoints -1 << " steps (" << simTime[numTimePoints-1] << " seconds)" << std::endl;
        displayOnMATLAB(matlabPtr, stream, silentMode);
        std::vector<std::vector<float>> sensorTemps(numTimePoints); // save the sensor data at each discrete time point provided
        simulator.updateTemperatureSensors();
        sensorTemps[0] = simulator.sensorTemps(); // initialize the time = 0 sensor
        Eigen::MatrixXf surfaceTemps(surfaceNodes.size(),numTimePoints);
        if (saveSurfaceData)
        {
            auto& fullTemp = simulator.Temp();
            surfaceTemps.col(0) = fullTemp(surfaceNodes);
        }
        try 
        {             
            for (int i = 1; i < numTimePoints; i++) {
                if (multiStep)
                    simulator.setFluenceRate(laserPose.row(i), laserPower[i], beamWaist, wavelength);
                simulator.multiStep(simTime[i] - simTime[i - 1]);
                sensorTemps[i] = simulator.sensorTemps();
                if (saveSurfaceData)
                {
                    auto& fullTemp = simulator.Temp();
                    surfaceTemps.col(i) = fullTemp(surfaceNodes);
                }
            }
        }
        catch (const std::exception& e) {
            stream << "MEX: Error in performTimeStepping() " << std::endl;
            displayOnMATLAB(matlabPtr, stream, silentMode);
            displayError(matlabPtr,e.what());
            return;
        }
        printDuration("MEX: Time Stepping Complete -- ");

        // Have to convert the std::vector to a matlab array for output
        //display3DVector(this->simulator.Temp_, "Final Temp_: ");
        Eigen::VectorXf TFinal = simulator.Temp();
        matlab::data::TypedArray<float> finalTemp = convertEigenVectorToMatlabArray(TFinal);
        outputs[0] = finalTemp;
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float> sensorTempsOutput = factory.createArray<float>({ sensorTemps.size(), sensorTemps[0].size() });
        for (size_t i = 0; i < sensorTemps.size(); ++i) {
            for (size_t j = 0; j < sensorTemps[i].size(); ++j) {
                sensorTempsOutput[i][j] = sensorTemps[i][j];
            }
        }
        outputs[1] = sensorTempsOutput;
        if (saveSurfaceData)
        {
            matlab::data::TypedArray<float> surfaceTempsOutput = convertEigenMatrixToMatlabArray(surfaceTemps);
            outputs[2] = surfaceTempsOutput;
        }

        printDuration("MEX: End of Mex Function. Total Time was ");
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, FluenceRate, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if ((inputs.size() != 4) && (buildMatrices)) {
            displayError(matlabPtr, "Exactly 3 inputs are required on initialization");
        }
        else if ((inputs.size() < 3) || (inputs.size() > 4))
        {
            displayError(matlabPtr, "The number of inputs must be 3 or 4. If three inputs are provided, the mesh won't be reset. If 4 inputs are provided, the mesh will be reset.");
        }
        if (inputs.size() == 4)
        {
            // A mesh was provided so we set our build Matrices flag 
            buildMatrices = true;
            // we also check the input type and set the mesh
            if (inputs[VarPlacement::MESH_INFO].getType() != matlab::data::ArrayType::STRUCT) {
                displayError(matlabPtr, "Input #1 must be a struct (meshInfo).");
            }
            try
            {
                checkMeshInfo(inputs[VarPlacement::MESH_INFO]);
            }
            catch (const std::exception& e) {
                std::cerr << "Error checking mesh input: " << e.what() << "\n";
            }
        }
        if ((outputs.size() > 3) || (outputs.size() < 2)) {
            displayError(matlabPtr, "Must specify 2 or 3 outputs.");
        }
        if (outputs.size() == 3)
            saveSurfaceData = true;
        if (outputs.size() == 2)
            saveSurfaceData = false;
        // Check input 1 which is required
        if (inputs[VarPlacement::THERMAL_INFO].getType() != matlab::data::ArrayType::STRUCT) {
            displayError(matlabPtr, "Input #1 must be a struct (thermalInfo).");
        }
        checkThermalInfo(inputs[VarPlacement::THERMAL_INFO]);
        // Check input 2 which is required
        if (inputs[VarPlacement::SETTINGS].getType() != matlab::data::ArrayType::STRUCT) {
            displayError(matlabPtr, "Input #2 must be a struct (Settings).");
        }
        checkSettings(inputs[VarPlacement::SETTINGS]);

        // Check input 3 which is required
        if (inputs[VarPlacement::LASER_INFO].getType() != matlab::data::ArrayType::STRUCT) {
            displayError(matlabPtr, "Input #3 must be a struct (Settings).");
        }
        checkLaserInfo(inputs[VarPlacement::LASER_INFO]);
    }

    /* Checks input terms but also creates Mesh */
    void checkMeshInfo(matlab::data::StructArray meshInfo)
    {
        size_t nfields = meshInfo.getNumberOfFields();
        auto fields = meshInfo.getFieldNames();
        size_t total_num_of_elements = meshInfo.getNumberOfElements();
        std::vector<std::string> fieldNames(fields.begin(), fields.end());
        /* Produce error if structure has more than 1 element. */
        if (total_num_of_elements != 1) {
            displayError(matlabPtr, "meshInfo must consist of 1 element (struct).");
        }
        if (nfields == 3)
        { // build mesh from nodes, elements, boundary faces
            if (!checkFieldNames(meshFields4, fieldNames))
            {
                std::ostringstream oss;
                oss << "MeshInfo must have fields: ";
                for (int i = 0; i < meshFields4.size(); i++)
                    oss << meshFields4[i] << " ";
                displayError(matlabPtr, oss.str());
            }
            /* Extract nodes from input */
            matlab::data::TypedArray<double> nodesArr = meshInfo[0]["nodes"];
            std::vector<Node> nodes(nodesArr.getDimensions()[1]);

            //stream << "Dimension of nodes: (" << nodesArr.getDimensions()[0] << " x " << nodesArr.getDimensions()[1] << ") nodes" << std::endl;
            //displayOnMATLAB(matlabPtr, stream, false);
            for (int i = 0; i < nodesArr.getDimensions()[1]; i++)
            {
                nodes[i].x = nodesArr[0][i];
                nodes[i].y = nodesArr[1][i];
                nodes[i].z = nodesArr[2][i];
            }

            /* Extract Elements from input */
            matlab::data::TypedArray<double> elemArr = meshInfo[0]["elements"];
            long nodesPerElem = elemArr.getDimensions()[0];
            long numElems = elemArr.getDimensions()[1];
            std::vector<Element> elements(numElems);
            //stream << "Dimension of elements: (" << nodesPerElem << " x " << numElems << ") elements" << std::endl;
            //displayOnMATLAB(matlabPtr, stream, false);
            for (int e = 0; e < numElems; e++)
            {
                elements[e].nodes.resize(nodesPerElem);
                for (int n = 0; n < nodesPerElem; n++)
                {
                    elements[e].nodes[n] = (elemArr[n][e] - 1); // convert 1-based indexing to 0-based indexing.
                }
            }
            /* Extract boundary faces from input */
            matlab::data::StructArray boundaryFaceStruct = meshInfo[0]["boundaryFaces"];
            float numFaces = boundaryFaceStruct.getDimensions()[0];
            std::vector<BoundaryFace> boundaryFaces(numFaces);
            //stream << "Dimension of boundaryFaces: (" << numFaces << " x " << boundaryFaceStruct.getDimensions()[1] << ") faces" << std::endl;
            //displayOnMATLAB(matlabPtr, stream, false);
            for (int j = 0; j < numFaces; j++)
            {   
                // boundary face type
                matlab::data::TypedArray<double> typeArr = boundaryFaceStruct[j]["type"];
                int type = static_cast<int>(typeArr[0]);
                boundaryFaces[j].type = static_cast<BoundaryType>(type);
                // boundar face element ID
                matlab::data::TypedArray<double> elemArr = boundaryFaceStruct[j]["elemID"];
                boundaryFaces[j].elemID = static_cast<long>(elemArr[0]) - 1; // matlab is 1 indexing so we convert here to 0 indexing
                // boundary face local face ID
                matlab::data::TypedArray<double> localFaceArr = boundaryFaceStruct[j]["localFaceID"];
                boundaryFaces[j].localFaceID = static_cast<long>(localFaceArr[0]) - 1; // matlab is 1 indexing so we convert here to 0 indexing
                // boundary face nodes
                matlab::data::TypedArray<double> nodeIDArr = boundaryFaceStruct[j]["nodes"];
                for (int i = 0; i < nodeIDArr.getDimensions()[0]; i++)
                {
                    boundaryFaces[j].nodes.push_back(nodeIDArr[i] - 1);
                }
            }
            mesh = Mesh(nodes, elements, boundaryFaces);
        }
        else if (nfields == 4)
        { // build mesh from xpos, ypos, zpos using a meshgrid
            if (!checkFieldNames(meshFields5, fieldNames))
            {
                std::ostringstream oss;
                oss << "MeshInfo must have fields: ";
                for (int i = 0; i < meshFields5.size(); i++)
                    oss << meshFields5[i] << " ";
                displayError(matlabPtr, oss.str());
            }
            std::vector<float> xpos, ypos, zpos;
            xpos = convertMatlabArrayToFloatVector(meshInfo[0]["xpos"]);
            ypos = convertMatlabArrayToFloatVector(meshInfo[0]["ypos"]);
            zpos = convertMatlabArrayToFloatVector(meshInfo[0]["zpos"]);
            matlab::data::Array BC = meshInfo[0]["boundaryConditions"];
            std::array<BoundaryType, 6> boundaryConditions;
            if (BC.getDimensions()[0] != 6)
                displayError(matlabPtr, "BC should have 6 dimensions, one for each face of a cuboid");
            for (size_t i = 0; i < 6; i++)
            {
                // Get MATLAB numeric value, then cast to enum
                double val = BC[i]; // or BC[i].get<double>()
                boundaryConditions[i] = static_cast<BoundaryType>(static_cast<int>(val));
            }
            mesh = Mesh::buildCubeMesh(xpos,ypos,zpos,boundaryConditions);
        }
        else
        {
            displayError(matlabPtr,"meshInfo should contain either 3 fields (nodes, elements, boundaryFaces) or 4 fields (xpos,ypos,zpos,boundaryConditions)");
        }
    }

    void checkThermalInfo(matlab::data::StructArray thermalInfo)
    {
        size_t nfields = thermalInfo.getNumberOfFields();
        auto fields = thermalInfo.getFieldNames();
        size_t total_num_of_elements = thermalInfo.getNumberOfElements();
        std::vector<std::string> fieldNames(fields.begin(), fields.end());
        /* Produce error if structure has more than 1 element. */
        if (total_num_of_elements != 1) {
            displayError(matlabPtr, "thermalInfo must consist of 1 element (struct).");
        }
        if (!checkFieldNames(thermalFields, fieldNames))
        {
            std::ostringstream oss;
            oss << "ThermalInfo must have fields: ";
            for (int i = 0; i < thermalFields.size(); i++)
                oss << thermalFields[i] << " ";
            displayError(matlabPtr, oss.str());
        }
        // Temperature -- thermalFields[0]
        matlab::data::Array tempArr = thermalInfo[0]["temperature"];
        if ((tempArr.getDimensions()[0] == 1) // checking for column vector
            || (tempArr.getDimensions()[0] != mesh.nodes().size())) // same length as mesh 
        {
            long length = static_cast<long>(tempArr.getDimensions()[0]);
            long width = static_cast<long>(tempArr.getDimensions()[1]);
            std::ostringstream oss;
            oss << "Temperature vector has incorrect length -- should be (" << mesh.nodes().size() << ", 1)";
            oss << " but is (" << length << ", " << width << ")";
            displayError(matlabPtr, oss.str());
        }
        Temp = convertMatlabArrayToEigenVector(tempArr);
            
        // MUA -- thermalFields[2]
        matlab::data::TypedArray<double> muaArr = thermalInfo[0]["MUA"];
        if ((muaArr.getDimensions()[0] != 1) || (muaArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "MUA should be a scalar");
        MUA = static_cast<float>(muaArr[0]);

        // VHC -- thermalFields[3]
        matlab::data::TypedArray<double> vhcArr = thermalInfo[0]["VHC"];
        if ((vhcArr.getDimensions()[0] != 1) || (vhcArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "VHC should be a scalar");
        VHC = static_cast<float>(vhcArr[0]);

        // TC -- thermalFields[4]
        matlab::data::TypedArray<double> tcArr = thermalInfo[0]["TC"];
        if ((tcArr.getDimensions()[0] != 1) || (tcArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "TC should be a scalar");
        TC = static_cast<float>(tcArr[0]);

        // HTC -- thermalFields[5]
        matlab::data::TypedArray<double> htcArr = thermalInfo[0]["HTC"];
        if ((htcArr.getDimensions()[0] != 1) || (htcArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "HTC should be a scalar");
        HTC = static_cast<float>(htcArr[0]);

        // FLUX -- thermalFields[6]
        matlab::data::TypedArray<double> fluxArr = thermalInfo[0]["flux"];
        if ((fluxArr.getDimensions()[0] != 1) || (fluxArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "flux should be a scalar");
        flux = static_cast<float>(fluxArr[0]);
        // AMBIENTTEMP -- thermalFields[7]
        matlab::data::TypedArray<double> ambArr = thermalInfo[0]["ambientTemp"];
        if ((ambArr.getDimensions()[0] != 1) || (ambArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "ambientTemp should be a scalar");
        ambientTemp = static_cast<float>(ambArr[0]);
    }

    void checkSettings(matlab::data::StructArray settings)
    {
        size_t nfields = settings.getNumberOfFields();
        auto fields = settings.getFieldNames();
        size_t total_num_of_elements = settings.getNumberOfElements();
        std::vector<std::string> fieldNames(fields.begin(), fields.end());
        /* Produce error if structure has more than 1 element. */
        if (total_num_of_elements != 1) {
            displayError(matlabPtr, "settings must consist of 1 element (struct).");
        }
        if (!checkFieldNames(settingsFields, fieldNames))
        {
            std::ostringstream oss;
            oss << "settings must have fields: ";
            for (int i = 0; i < settingsFields.size(); i++)
                oss << settingsFields[i] << " ";
            displayError(matlabPtr, oss.str());
        }

        // dt
        matlab::data::TypedArray<double> dtArr = settings[0]["dt"];
        if ((dtArr.getDimensions()[0] != 1) || (dtArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "dt should be a scalar");
        dt = static_cast<float>(dtArr[0]);

        //alpha
        matlab::data::TypedArray<double> alphaArr = settings[0]["alpha"];
        if ((alphaArr.getDimensions()[0] != 1) || (alphaArr.getDimensions()[1] != 1))
            displayError(matlabPtr, "alpha should be a scalar");
        alpha = static_cast<float>(alphaArr[0]);

        // finaltime
        if (settings[0]["time"].getDimensions()[1] != 1)
            displayError(matlabPtr, "time should be a column vector");
        simTime = convertMatlabArrayToFloatVector(settings[0]["time"]);

        // Check sensor Locations
        matlab::data::TypedArray<double> sensorLocations = settings[0]["sensorLocations"];
        if (sensorLocations.getDimensions()[1] != 3)
            displayError(matlabPtr, "sensorLocations must be n x 3");
        sLocations.clear();
        for (int s = 0; s < sensorLocations.getDimensions()[0]; s++) {
            sLocations.push_back({ static_cast<float>(sensorLocations[s][0]),
                static_cast<float>(sensorLocations[s][1]),
                static_cast<float>(sensorLocations[s][2]) });
        }

        //optional settings
        if (hasField(fieldNames, "silentMode"))
        {
            matlab::data::Array smArr = settings[0]["silentMode"];
            silentMode = static_cast<bool>(smArr[0]);
            simulator.silentMode = silentMode;
        }
        if (hasField(fieldNames, "useGPU"))
        {
            matlab::data::Array gpuArr = settings[0]["useGPU"];
            useGPU = static_cast<bool>(gpuArr[0]);
        }
        if (hasField(fieldNames, "useAllCPUs"))
        {
            matlab::data::Array cpuArr = settings[0]["useAllCPUs"];
            useAllCPUs = static_cast<bool>(cpuArr[0]);
        }
        if (hasField(fieldNames, "resetIntegration"))
        {
            matlab::data::Array matArr = settings[0]["resetIntegration"];
            resetIntegration = static_cast<bool>(matArr[0]);
        }

    }

    void checkLaserInfo(matlab::data::StructArray laserInfo)
    {
        size_t nfields = laserInfo.getNumberOfFields();
        auto fields = laserInfo.getFieldNames();
        size_t total_num_of_elements = laserInfo.getNumberOfElements();
        std::vector<std::string> fieldNames(fields.begin(), fields.end());
        /* Produce error if structure has more than 1 element. */
        if (total_num_of_elements != 1) {
            displayError(matlabPtr, "settings must consist of 1 element (struct).");
        }
        if (nfields == 1)
        { // laser fluence rate was passed in
            if (!checkFieldNames(laserFields1, fieldNames))
            {
                std::ostringstream oss;
                oss << "LaserInfo must have fields: FluenceRate if only one field";
                displayError(matlabPtr, oss.str());
            }

            // FLUENCE Rate
            matlab::data::Array fluenceArr = laserInfo[0]["fluenceRate"];
            if ((fluenceArr.getDimensions()[0] == 1) // checking for column vector
                || ((fluenceArr.getDimensions()[0] != mesh.nodes().size()) // length != num nodes
                    && (fluenceArr.getDimensions()[0] != mesh.elements().size()))) // length != num elements
                displayError(matlabPtr, "FluenceRate vector should be a column vector with length equal to number of elements or number of nodes in mesh");
            fluenceRate = convertMatlabArrayToEigenVector(fluenceArr);

            multiStep = false;
        }
        else if (nfields == 4)
        {
            if (!checkFieldNames(laserFields3, fieldNames))
            {
                std::ostringstream oss;
                oss << "LaserInfo must have fields: laserPose, laserPower, beamWaist, and wavelength if using three fields";
                displayError(matlabPtr, oss.str());
            }
            
            // laser pose
            matlab::data::Array poseArr = laserInfo[0]["laserPose"];
            if ((poseArr.getDimensions()[0] != simTime.size()) || (poseArr.getDimensions()[1] != 6))
                displayError(matlabPtr, "laserPose must be n x 6, where n is the length of the time vector");
            laserPose = convertMatlabArrayToEigenMatrix(poseArr);

            // laser power
            matlab::data::Array laserPowerArr = laserInfo[0]["laserPower"];
            if ((laserPowerArr.getDimensions()[0] != simTime.size()) || (laserPowerArr.getDimensions()[1] != 1))
                displayError(matlabPtr, "laserPower must be n x 1, where n is the length of the time vector");
            laserPower = convertMatlabArrayToFloatVector(laserInfo[0]["laserPower"]);

            // Beam Waist
            matlab::data::TypedArray<double> waistArr = laserInfo[0]["beamWaist"];
            if ((waistArr.getDimensions()[0] != 1) || (waistArr.getDimensions()[1] != 1))
                displayError(matlabPtr, "beamWaist must be a scalar.");
            beamWaist = static_cast<float>(waistArr[0]);

            // wavelength
            matlab::data::TypedArray<double> wavelengthArr = laserInfo[0]["wavelength"];
            if ((wavelengthArr.getDimensions()[0] != 1) || (wavelengthArr.getDimensions()[1] != 1))
                displayError(matlabPtr, "wavelength must be a scalar.");
            wavelength = static_cast<float>(wavelengthArr[0]);

            multiStep = true;
        }
    }

    void resetTimer(){
        timeRef_ = std::chrono::steady_clock::now();
    }

    void printDuration(const std::string& message) {
        auto stopTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - timeRef_);
        stream << message << duration.count() / 1000000.0 << " s" << std::endl;	
        displayOnMATLAB(matlabPtr, stream, silentMode);
    }

// #ifdef USE_CUDA
//     void initializeGPU(){
//         if (!gpuHandle){
//             gpuHandle = new GPUTimeIntegrator();
//         }
//         simulator.buildMatrices();
//         gpuHandle->setAlpha(simulator.alpha_);
//         gpuHandle->setDeltaT(simulator.dt_);
//         gpuHandle->setModel(&simulator);
//         gpuHandle->initializeWithModel();
//     }

//     void multiStepGPU(float totalTime){
//         auto startTime = std::chrono::steady_clock::now();
//         int numSteps = round(totalTime / simulator.dt_);
//         simulator.initializeSensorTemps(numSteps);
//         simulator.updateTemperatureSensors(0);
//         for (int i = 1; i <= numSteps; i++) {
//             gpuHandle->singleStepWithUpdate();
//             simulator.updateTemperatureSensors(i);
//         }
//     }
// #endif 

};
