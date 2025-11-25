#include "MEX_Utility.hpp"

class MexFunction : public matlab::mex::Function {

    /*
    * Use case MEX_Heat_Simulation(meshInfo,thermalInfo,settings);
    *   meshInfo - struct containing either sensorLocations and [(xpos, ypos, zpos, boundaryCond) OR (nodes,elements,boundaryFaces)]
    *   thermalInfo - struct containing temperature, fluenceRate, MUA, VHC, TC, HTC, FLUX, AmbientTemp
    *   settings - struct containing dt, alpha, finalTime, silentMode, useGPU, useAllCPUs, createMatrices
    */

private:
    /*Inputs required : T0, FluenceRate, tissueSize, tfinal, deltat, tissueProperties, BC, Flux, ambientTemp, sensorLocations, beamWaist, time, laserPose, laserPower,
   OPTIONAL       : (layers), (useAllCPUs), (useGPU), (alpha), (silentMode), (Nn1d), (createAllMatrices), */
    enum VarPlacement {
        MESH_INFO, THERMAL_INFO, SETTINGS 
    };

    const std::vector<std::string> meshFields4 = { "nodes","elements","boundaryFaces","sensorLocations" };
    const std::vector<std::string> meshFields5 = { "xpos","ypos","zpos","boundaryConditions","sensorLocations"};
    const std::vector<std::string> thermalFields = { "temperature","fluenceRate","MUA","VHC","TC","HTC","flux","ambientTemp" };
    const std::vector<std::string> settingsFields = { "dt","alpha","finalTime" }; // optional fields {"silentMode","useGPU","useAllCPUs","createMatrices"}

    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    FEM_Simulator simulator;
    Mesh mesh;
    std::ostringstream stream;
    /* Default Parameters that are stored as class variables for repeat calls */
    bool silentMode = true;
    bool useAllCPUs = true;
    bool useGPU = true;
    bool createAllMatrices = true;
    std::chrono::steady_clock::time_point timeRef_ = std::chrono::steady_clock::now();
#ifdef USE_CUDA
    GPUTimeIntegrator* gpuHandle = nullptr;
#endif
        

public:
    /* Constructor for the class. */
    MexFunction()
    {
        matlabPtr = getEngine();
    }

    ~MexFunction()
    {
        #ifdef USE_CUDA
        delete gpuHandle;
        gpuHandle = nullptr;
        #endif
        return;
    }

    /* This is the gateway routine for the MEX-file. */
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        resetTimer();
        checkArguments(outputs, inputs); // mesh object gets created in checkArguments through checkMeshInfo
        stream.str("");
        stream << "MEX: Inputs Validated -- Setting Up Model" << std::endl;
        displayOnMATLAB(matlabPtr,stream,silentMode);
        float simDuration;
        try {
            // apply mesh which was already constructed
            simulator.setMesh(mesh);
            // convert thermal model info 
            matlab::data::StructArray thermalInfo = inputs[VarPlacement::THERMAL_INFO];
            Eigen::VectorXf Temp = convertMatlabArrayToEigenVector(thermalInfo[0]["temperature"]);
            Eigen::VectorXf fluenceRate = convertMatlabArrayToEigenVector(thermalInfo[0]["fluenceRate"]);
            matlab::data::TypedArray<double> muaArr = thermalInfo[0]["MUA"];
            float MUA = static_cast<float>(muaArr[0]);
            matlab::data::TypedArray<double> tcArr = thermalInfo[0]["TC"];
            float TC = static_cast<float>(tcArr[0]);
            matlab::data::TypedArray<double> vhcArr = thermalInfo[0]["VHC"];
            float VHC = static_cast<float>(vhcArr[0]);
            matlab::data::TypedArray<double> htcArr = thermalInfo[0]["HTC"];
            float HTC = static_cast<float>(htcArr[0]);
            matlab::data::TypedArray<double> fluxArr = thermalInfo[0]["flux"];
            float flux = static_cast<float>(fluxArr[0]);
            matlab::data::TypedArray<double> ambArr = thermalInfo[0]["ambientTemp"];
            float ambientTemp = static_cast<float>(ambArr[0]);

            simulator.setTemp(Temp);
            simulator.setFluenceRate(fluenceRate);
            simulator.setMUA(MUA);
            simulator.setTC(TC);
            simulator.setVHC(VHC);
            simulator.setHTC(HTC);
            simulator.setHeatFlux(flux);
            simulator.setAmbientTemp(ambientTemp);
            
            printDuration("MEX: Applied ThermalModel to Simulator ");

            // Get time step and final time
            matlab::data::StructArray settings = inputs[VarPlacement::SETTINGS];
            matlab::data::TypedArray<double> durArr = settings[0]["finalTime"];
            simDuration = static_cast<float>(durArr[0]);
            matlab::data::TypedArray<double> dtArr = settings[0]["dt"];
            float dt = static_cast<float>(dtArr[0]);
            matlab::data::TypedArray<double> alphaArr = settings[0]["alpha"];
            float alpha = static_cast<float>(alphaArr[0]);

            simulator.setDt(dt);
            simulator.setAlpha(alpha);
            printDuration("MEX: Set time stepping rate ");
            
            // Set sensor locations
            matlab::data::StructArray meshInfo = inputs[VarPlacement::MESH_INFO];
            matlab::data::TypedArray<double> sensorLocations = meshInfo[0]["sensorLocations"];
            std::vector<std::array<float, 3>> sLocations;
            for (int s = 0; s < sensorLocations.getDimensions()[0]; s++) {
                sLocations.push_back({ static_cast<float>(sensorLocations[s][0]), 
                    static_cast<float>(sensorLocations[s][1]),
                    static_cast<float>(sensorLocations[s][2]) });
            }
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
        if (createAllMatrices) { // only need to create the KMF matrices the first time
            createAllMatrices = true;
            try {
#ifdef USE_CUDA
                if (useGPU)
                {
                    stream << "MEX: GPU enabled " << std::endl;
                    displayOnMATLAB(stream);
                    initializeGPU();
                }
                else
#endif  
                {
                    stream << "MEX: GPU Disabled " << std::endl;
                    displayOnMATLAB(matlabPtr, stream, silentMode);
                    simulator.initializeModel();
                }
                /*stream << "Global matrices created" << std::endl;
                displayOnMATLAB(stream);*/
            }
            catch (const std::exception& e) {
                stream << "MEX: Error in building Matrices" << std::endl;
                displayOnMATLAB(matlabPtr, stream, silentMode);
                displayError(matlabPtr, e.what());
                return;
            }
            printDuration("MEX: Matrices Built -- ");
        }

        // Perform time stepping
        int numSteps = round(simDuration / simulator.dt()) + 1;
        stream << "Time Stepping for " << numSteps-1 << " steps (" << simDuration << " seconds)" << std::endl;
        displayOnMATLAB(matlabPtr, stream, silentMode);
        std::vector<std::vector<float>> sensorTemps(numSteps);
        try 
        { //
#ifdef USE_CUDA
            if (this->useGPU)
            {
                stream << "MEX: GPU time step " << std::endl;
                displayOnMATLAB(stream);
                multiStepGPU(simDuration);
            }
            else
#endif  
            {
                stream << "MEX: CPU time step " << std::endl;
                displayOnMATLAB(matlabPtr, stream, silentMode);
                simulator.updateTemperatureSensors();
                sensorTemps[0] = simulator.sensorTemps();
                for (int i = 1; i < numSteps; i++) {
                    simulator.singleStep();
                    simulator.updateTemperatureSensors();
                    sensorTemps[i] = simulator.sensorTemps();
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

        printDuration("MEX: End of Mex Function. Total Time was ");
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, FluenceRate, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() != 3) {
            displayError(matlabPtr, "Exactly 3 inputs are required");
        }
        if (outputs.size() != 2) {
            displayError(matlabPtr, "Exactly 2 outputs must be specified.");
        }

        if (inputs[VarPlacement::MESH_INFO].getType() != matlab::data::ArrayType::STRUCT) {
            displayError(matlabPtr, "Input #1 must be a struct (meshInfo).");
        }
        try
        {
            checkMeshInfo(inputs[VarPlacement::MESH_INFO]);
        }
        catch (const std::exception & e) {
            std::cerr << "Error checking mesh input: " << e.what() << "\n";
        }

        if (inputs[VarPlacement::THERMAL_INFO].getType() != matlab::data::ArrayType::STRUCT) {
            displayError(matlabPtr, "Input #2 must be a struct (thermalInfo).");
        }
        checkThermalInfo(inputs[VarPlacement::THERMAL_INFO]);

        if (inputs[VarPlacement::SETTINGS].getType() != matlab::data::ArrayType::STRUCT) {
            displayError(matlabPtr, "Input #1 must be a struct (Settings).");
        }
        checkSettings(inputs[VarPlacement::SETTINGS]);
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
        if (nfields == 4)
        { // build mesh from nodes, elements, boundary faces
            if (!checkFieldNames(meshFields4, fieldNames))
            {
                std::ostringstream oss;
                oss << "MeshInfo must have fields: ";
                for (int i = 0; i < meshFields4.size(); i++)
                    oss << meshFields4[i] << " ";
                displayError(matlabPtr, oss.str());
            }
            displayError(matlabPtr, "Have not implemented meshInfo with 3 fields yet");
        }
        else if (nfields == 5)
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

        if (meshInfo[0]["sensorLocations"].getDimensions()[1] != 3)
            displayError(matlabPtr, "sensorLocations must be n x 3");
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
        if ((thermalInfo[0]["temperature"].getDimensions()[0] == 1) // checking for column vector
            || (thermalInfo[0]["temperature"].getDimensions()[0] != mesh.nodes().size())) // same length as mesh 
        {
            long length = static_cast<long>(thermalInfo[0]["temperature"].getDimensions()[0]);
            long width = static_cast<long>(thermalInfo[0]["temperature"].getDimensions()[1]);
            std::ostringstream oss;
            oss << "Temperature vector has incorrect length -- should be (" << mesh.nodes().size() << ", 1)";
            oss << " but is (" << length << ", " << width << ")";
            displayError(matlabPtr, oss.str());
        }
            
        // FLUENCE Rate -- thermalFields[2]
        if ((thermalInfo[0]["fluenceRate"].getDimensions()[0] == 1) // checking for column vector
            || ((thermalInfo[0]["fluenceRate"].getDimensions()[0] != mesh.nodes().size()) // length != num nodes
                && (thermalInfo[0]["fluenceRate"].getDimensions()[0] != mesh.elements().size()))) // length != num elements
            displayError(matlabPtr, "FluenceRate vector should be a column vector with length equal to number of elements or number of nodes in mesh");
        // MUA -- thermalFields[3]
        if ((thermalInfo[0]["MUA"].getDimensions()[0] != 1) || (thermalInfo[0]["MUA"].getDimensions()[1] != 1))
            displayError(matlabPtr, "MUA should be a scalar");
        // VHC -- thermalFields[4]
        if ((thermalInfo[0]["VHC"].getDimensions()[0] != 1) || (thermalInfo[0]["VHC"].getDimensions()[1] != 1))
            displayError(matlabPtr, "VHC should be a scalar");
        // TC -- thermalFields[5]
        if ((thermalInfo[0]["TC"].getDimensions()[0] != 1) || (thermalInfo[0]["TC"].getDimensions()[1] != 1))
            displayError(matlabPtr, "TC should be a scalar");
        // HTC -- thermalFields[6]
        if ((thermalInfo[0]["HTC"].getDimensions()[0] != 1) || (thermalInfo[0]["HTC"].getDimensions()[1] != 1))
            displayError(matlabPtr, "HTC should be a scalar");
        // FLUX -- thermalFields[7]
        if ((thermalInfo[0]["flux"].getDimensions()[0] != 1) || (thermalInfo[0]["flux"].getDimensions()[1] != 1))
            displayError(matlabPtr, "flux should be a scalar");
        // AMBIENTTEMP -- thermalFields[8]
        if ((thermalInfo[0]["ambientTemp"].getDimensions()[0] != 1) || (thermalInfo[0]["ambientTemp"].getDimensions()[1] != 1))
            displayError(matlabPtr, "ambientTemp should be a scalar");
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
        if ((settings[0]["dt"].getDimensions()[0] != 1) || (settings[0]["dt"].getDimensions()[1] != 1))
            displayError(matlabPtr, "dt should be a scalar");
        //if (settings[0]["dt"][0] < 0)
        //    displayError(matlabPtr, "dt should be greater than 0");
        //alpha
        if ((settings[0]["alpha"].getDimensions()[0] != 1) || (settings[0]["alpha"].getDimensions()[1] != 1))
            displayError(matlabPtr, "alpha should be a scalar");
        //if ((settings[0]["alpha"][0] >= 0) || (settings[0]["alpha"][0] <= 1))
        //    displayError(matlabPtr, "alpha should be between 0 and 1");
        // finaltime
        if ((settings[0]["finalTime"].getDimensions()[0] != 1) || (settings[0]["finalTime"].getDimensions()[1] != 1))
            displayError(matlabPtr, "finalTime should be a scalar");
        /*if (settings[0]["finalTime"][0] < settings[0]["dt"])
            displayError(matlabPtr, "finalTime should be greater than the timestep");*/
        //optional settings
        if (hasField(fieldNames, "silentMode"))
        {
            matlab::data::Array smArr = settings[0]["silentMode"];
            silentMode = static_cast<bool>(smArr[0]);
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
        if (hasField(fieldNames, "createMatrices"))
        {
            matlab::data::Array matArr = settings[0]["createMatrices"];
            createAllMatrices = createAllMatrices || static_cast<bool>(matArr[0]);
        }

    }

    bool checkForMatrixReset(int Nn1d, std::vector<std::vector<std::vector<float>>>& T0, std::vector<std::vector<std::vector<float>>>& FluenceRate,
        float tissueSize[3],float layerHeight, int elemsInLayer,int boundaryType[6]) {
        // this function checks to see if we need to recreate the global element matrices
        createAllMatrices = true;


        return createAllMatrices;
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

#ifdef USE_CUDA
    void initializeGPU(){
        if (!gpuHandle){
            gpuHandle = new GPUTimeIntegrator();
        }
        simulator.buildMatrices();
        gpuHandle->setAlpha(simulator.alpha_);
        gpuHandle->setDeltaT(simulator.dt_);
        gpuHandle->setModel(&simulator);
        gpuHandle->initializeWithModel();
    }

    void multiStepGPU(float totalTime){
        auto startTime = std::chrono::steady_clock::now();
        int numSteps = round(totalTime / simulator.dt_);
        simulator.initializeSensorTemps(numSteps);
        simulator.updateTemperatureSensors(0);
        for (int i = 1; i <= numSteps; i++) {
            gpuHandle->singleStepWithUpdate();
            simulator.updateTemperatureSensors(i);
        }
    }
#endif 

};
