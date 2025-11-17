/* ========================================================================
* It compiles in MATLAB, but not in Visual Studio... idk.
 *=======================================================================*/

#include "mex.hpp"
#include "mexAdapter.hpp"
#include <string>
#include <memory>
#include <iostream>
#include "FEM_Simulator.h"

#ifdef USE_CUDA
#include "GPUTimeIntegrator.cuh"
#endif

 //using namespace matlab::mex;
 //using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
private:
    /*Inputs required : T0, FluenceRate, tissueSize, tfinal, deltat, tissueProperties, BC, Flux, ambientTemp, sensorLocations, beamWaist, time, laserPose, laserPower,
   OPTIONAL       : (layers), (useAllCPUs), (useGPU), (alpha), (silentMode), (Nn1d), (createAllMatrices), */
    enum VarPlacement {
        TEMPERATURE, FLUENCE_RATE, TISSUE_SIZE, SIM_DURATION, DELTAT, TISSUE_PROP, BC,
        FLUX, AMB_TEMP, SENSOR_LOC, LAYERS, USE_ALL_CPUS, USE_GPU, ALPHA, SILENT_MODE, NN1D, CREATE_MAT, 
    };

    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr;
    FEM_Simulator simulator;
    std::ostringstream stream;
    /* Default Parameters that are stored as class variables for repeat calls */
    bool silentMode = true;
    bool useAllCPUs = true;
    bool useGPU = true;
    bool createAllMatrices = true;
    float layerHeight = 1;
    int elemsInLayer = 1;
    int Nn1d = 2;
    std::chrono::steady_clock::time_point timeRef = std::chrono::steady_clock::now();
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
    /* Helper function to convert a matlab array to a std vector*/
    std::vector<std::vector<std::vector<float>>> convertMatlabArrayTo3DVector(const matlab::data::Array& matlabArray) {
        std::vector<std::vector<std::vector<float>>> result;

        // Get dimensions of the Matlab array
        size_t dim1 = matlabArray.getDimensions()[0];
        size_t dim2 = matlabArray.getDimensions()[1];
        size_t dim3 = matlabArray.getDimensions()[2];

        // Resize the result vector to match the dimensions of the Matlab array
        result.resize(dim1);
        for (size_t i = 0; i < dim1; ++i) {
            result[i].resize(dim2);
            for (size_t j = 0; j < dim2; ++j) {
                result[i][j].resize(dim3);
            }
        }

        // Iterate through the Matlab array and fill the result vector
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    result[i][j][k] = static_cast<float>(matlabArray[i][j][k]);
                }
            }
        }

        return result;
    }

    matlab::data::TypedArray<float> convertVectorToMatlabArray(const std::vector<std::vector<std::vector<float>>>& vec) {
        // Get dimensions of the input vector
        size_t dim1 = vec.size();
        size_t dim2 = (dim1 > 0) ? vec[0].size() : 0;
        size_t dim3 = (dim2 > 0) ? vec[0][0].size() : 0;

        matlab::data::ArrayFactory factory;
        // Create MATLAB array with appropriate dimensions
        matlab::data::TypedArray<float> matlabArray = factory.createArray<float>({ dim1, dim2, dim3 });

        // Fill MATLAB array with vector data
        for (size_t i = 0; i < dim1; ++i) {
            for (size_t j = 0; j < dim2; ++j) {
                for (size_t k = 0; k < dim3; ++k) {
                    matlabArray[i][j][k] = vec[i][j][k];
                }
            }
        }

        return matlabArray;
    }

    /* Helper function to generate an error message from given string,
     * and display it over MATLAB command prompt.
     */
    void displayError(std::string errorMessage)
    {
        matlab::data::ArrayFactory factory;
        matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({
          factory.createScalar(errorMessage) }));
    }

    /* Helper function to generate messages to matlab command window
    */
    void displayOnMATLAB(std::ostringstream& stream) {
        // Pass stream content to MATLAB fprintf function
        if (!this->silentMode) {
            matlab::data::ArrayFactory factory;
            matlabPtr->feval(u"fprintf", 0,
                std::vector<matlab::data::Array>({ factory.createScalar(stream.str()) }));
            // Clear stream buffer
            stream.str("");
        }
    }

    void display3DVector(const std::vector<std::vector<std::vector<float>>>& vec, std::string title) {
        // Get dimensions of the input vector
        size_t dim1 = vec.size();
        size_t dim2 = (dim1 > 0) ? vec[0].size() : 0;
        size_t dim3 = (dim2 > 0) ? vec[0][0].size() : 0;

        stream << "\n" << title << std::endl;
        for (int k = 0; k < dim3; k++) {
            for (int j = 0; j < dim2; j++) {
                for (int i = 0; i < dim1; i++) {
                    stream << vec[i][j][k] << ", ";
                    displayOnMATLAB(stream);
                }
                stream << std::endl;
            }
            stream << std::endl;
        }
        displayOnMATLAB(stream);
    }


    /* This is the gateway routine for the MEX-file. */
    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        resetTimer();
        checkArguments(outputs, inputs);
        stream.str("");
        stream << "MEX: Start of MEX function" << std::endl;
        displayOnMATLAB(stream);
        float simDuration;
        try {
            
            // Have to convert T0 and FluenceRate to std::vector<<<float>>>
            std::vector<std::vector<std::vector<float>>> T0 = convertMatlabArrayTo3DVector(inputs[VarPlacement::TEMPERATURE]);
            std::vector<std::vector<std::vector<float>>> FluenceRate = convertMatlabArrayTo3DVector(inputs[VarPlacement::FLUENCE_RATE]);
            printDuration("MEX: Converted MATLAB Matrices -- ");

            // Get tissue size
            float tissueSize[3];
            tissueSize[0] = inputs[VarPlacement::TISSUE_SIZE][0];
            tissueSize[1] = inputs[VarPlacement::TISSUE_SIZE][1];
            tissueSize[2] = inputs[VarPlacement::TISSUE_SIZE][2];
            
            // Get time step and final time
            simDuration = inputs[VarPlacement::SIM_DURATION][0];
            float deltaT = inputs[VarPlacement::DELTAT][0];

            // Extract tissue properties
            float MUA = inputs[VarPlacement::TISSUE_PROP][0];
            float TC = inputs[VarPlacement::TISSUE_PROP][1];
            float VHC_ = inputs[VarPlacement::TISSUE_PROP][2];
            float HTC = inputs[VarPlacement::TISSUE_PROP][3];

            // Get boundary conditions
            int boundaryType[6] = { 0,0,0,0,0,0 };
            //stream << "Boundary Conditions: ";
            for (int i = 0; i < 6; i++) {
               boundaryType[i] = inputs[VarPlacement::BC][i];
               //stream << boundaryType[i] << ", ";
            }
            //stream << std::endl;
            //displayOnMATLAB(stream);*/
            

            // get heatFlux condition
            float heatFlux = inputs[VarPlacement::FLUX][0];
            // get ambient temperature
            float ambientTemp = inputs[VarPlacement::AMB_TEMP][0];

            /* SET ALL PARAMETERS NECESSARY BEFORE CONSTRUCTING ELEMENTS*/
            // First check to see if certain variables have changed which would require us to reconstruct elements
            if (!this->createAllMatrices) {
                this->createAllMatrices = checkForMatrixReset(Nn1d, T0, FluenceRate, tissueSize, layerHeight, elemsInLayer, boundaryType);
            }

            // Set the type of basis functions we are using by setting nodes per dimension of an 
            this->simulator.Nn1d = Nn1d;
            this->simulator.setTemp(T0);
            // Set the FluenceRate
            this->simulator.setFluenceRate(FluenceRate);
            this->simulator.setTissueSize(tissueSize);
            // set the layer info
            this->simulator.setLayer(layerHeight, elemsInLayer);
            // set the time step
            this->simulator.setDeltaT(deltaT);
            // set the tissue properties
            this->simulator.setMUA(MUA);
            this->simulator.setTC(TC);
            this->simulator.setVHC(VHC_);
            this->simulator.setHTC(HTC);
            // set boundary conditions
            this->simulator.setBoundaryConditions(boundaryType);
            // set heatFlux
            this->simulator.setHeatFlux(heatFlux_);

            this->simulator.setAmbientTemp(ambientTemp);
        }
        catch (const std::exception& e) {
            stream << "MEX: Error in Setup: " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }
        
        // Set sensor locations
        auto sensorTempsInput = inputs[VarPlacement::SENSOR_LOC];
        std::vector<std::array<float, 3>> sensorTemps;
        for (int s = 0; s < sensorTempsInput.getDimensions()[0]; s++) {
            sensorTemps.push_back({ sensorTempsInput[s][0],sensorTempsInput[s][1] ,sensorTempsInput[s][2] });
        }
        try {
            this->simulator.setSensorLocations(sensorTemps);
        }
        catch (const std::exception& e){
            stream << "MEX: Error setting sensor locations " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }
        printDuration("MEX: Set all simulation parameters -- ");

        // Set parallelization
        Eigen::setNbThreads(1);
#ifdef _OPENMP
        stream << "MEX: OPEMMP Enabled" << std::endl;
        displayOnMATLAB(stream);
        if (useAllCPUs) { //useAllCPUs is true
            stream << "MEX: Using all CPUs" << std::endl;
            displayOnMATLAB(stream);
            Eigen::setNbThreads(omp_get_num_procs()/2);
        }
#endif
        stream << "MEX: Number of threads: " << Eigen::nbThreads() << std::endl;
        displayOnMATLAB(stream);
        // Create global K M and F 
        if (this->createAllMatrices) { // only need to create the KMF matrices the first time
            this->createAllMatrices = false;
            try {
#ifdef USE_CUDA
                if (this->useGPU)
                {
                    stream << "MEX: GPU enabled " <<std::endl;
                    displayOnMATLAB(stream);
                    initializeGPU();
                }
                else
#endif  
                {
                    stream << "MEX: GPU Disabled " << std::endl;
                    displayOnMATLAB(stream);
                    this->simulator.initializeModel();    
                }
                /*stream << "Global matrices created" << std::endl;
                displayOnMATLAB(stream);*/
            }
            catch (const std::exception& e) {
                stream << "MEX: Error in createKMF() " << std::endl;
                displayOnMATLAB(stream);
                displayError(e.what());
                return;
            }
            printDuration("MEX: Matrices Built -- ");
        }

        // Perform time stepping
        try { //
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
                displayOnMATLAB(stream);
                this->simulator.multiStep(simDuration);
            }
        }
        catch (const std::exception& e) {
            stream << "MEX: Error in performTimeStepping() " << std::endl;
            displayOnMATLAB(stream);
            displayError(e.what());
            return;
        }
        printDuration("MEX: Time Stepping Complete -- ");

        // Have to convert the std::vector to a matlab array for output
        //display3DVector(this->simulator.Temp_, "Final Temp_: ");
        std::vector<std::vector<std::vector<float>>> TFinal = this->simulator.TempAsVec();
        matlab::data::TypedArray<float> finalTemp = convertVectorToMatlabArray(TFinal);
        outputs[0] = finalTemp;
        matlab::data::ArrayFactory factory;
        matlab::data::TypedArray<float> sensorTempsOutput = factory.createArray<float>({ this->simulator.sensorTemps_.size(), this->simulator.sensorTemps_[0].size()});
        for (size_t i = 0; i < this->simulator.sensorTemps_.size(); ++i) {
            for (size_t j = 0; j < this->simulator.sensorTemps_[i].size(); ++j) {
                sensorTempsOutput[i][j] = this->simulator.sensorTemps_[i][j];
            }
        }
        outputs[1] = sensorTempsOutput;

        printDuration("MEX: End of Mex Function. Total Time was ");
    }

    /* This function makes sure that user has provided the proper inputs
    * Inputs: T0, FluenceRate, tissueSize TC, VHC, MUA, HTC, boundaryConditions
     */
    void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
        if (inputs.size() < 10) {
            displayError("At least 10 inputs required: T0, NFR, tissueSize, simDuration,"
                "deltaT tissueProperties, BC, Flux, ambientTemp, sensorLocations, "
                "(layers), (useAllCPUs), (useGPU), (alpha), (silentMode), (Nn1d), (createAllMatrices)");
        }
        if (outputs.size() > 2) {
            displayError("Too many outputs specified.");
        }
        if (outputs.size() < 2) {
            displayError("Not enough outputs specified.");
        }
        if (inputs[VarPlacement::TEMPERATURE].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("T0 must be an Array of type Single.");
        }
        if (inputs[VarPlacement::FLUENCE_RATE].getType() != matlab::data::ArrayType::SINGLE) {
            displayError("NFR must be an Array of type Single.");
        }
        if (!((inputs[VarPlacement::FLUENCE_RATE].getDimensions()[0] == inputs[VarPlacement::TEMPERATURE].getDimensions()[0])
            && (inputs[VarPlacement::FLUENCE_RATE].getDimensions()[1] == inputs[VarPlacement::TEMPERATURE].getDimensions()[1])
            && (inputs[VarPlacement::FLUENCE_RATE].getDimensions()[2] == inputs[VarPlacement::TEMPERATURE].getDimensions()[2]))
            &&
            !((inputs[VarPlacement::FLUENCE_RATE].getDimensions()[0] == (inputs[VarPlacement::TEMPERATURE].getDimensions()[0] - 1))
                && (inputs[VarPlacement::FLUENCE_RATE].getDimensions()[1] == (inputs[VarPlacement::TEMPERATURE].getDimensions()[1] - 1))
                && (inputs[VarPlacement::FLUENCE_RATE].getDimensions()[2] == (inputs[VarPlacement::TEMPERATURE].getDimensions()[2] - 1)))) {
            displayError("NFR must have either the same dimensions as Temp, or one less in each axis");
        }
        if ((inputs[VarPlacement::TISSUE_SIZE].getDimensions()[0] != 3) || (inputs[VarPlacement::TISSUE_SIZE].getDimensions()[1] != 1)) {
            displayError("Tissue Size must be 3 x 1");
        }
        /* This Check Doesnt work
        if (inputs[4][0] > inputs[3][0]) {
            displayError("deltaT must be less than the final time");
        }
        */
        if ((inputs[VarPlacement::TISSUE_PROP].getDimensions()[0] != 4) || (inputs[VarPlacement::TISSUE_PROP].getDimensions()[1] != 1)) {
            displayError("Tissue Properties must be 4 x 1: MUA_, TC_, VHC_, HTC_");
        }
        if ((inputs[VarPlacement::BC].getDimensions()[0] != 6) || (inputs[VarPlacement::BC].getDimensions()[1] != 1)) {
            displayError("Boundary Conditions Size must be 6 x 1");
        }
        if ((inputs[VarPlacement::BC].getType() != matlab::data::ArrayType::INT32)) {
            displayError("Boundary Conditions must be an int");
        }
        if ((inputs[VarPlacement::SENSOR_LOC].getDimensions()[1] != 3)) {
            displayError("Sensor Locations must be n x 3");
        }
        if (inputs.size() > VarPlacement::LAYERS) {
            layerHeight = inputs[VarPlacement::LAYERS][0];
            elemsInLayer = int(inputs[VarPlacement::LAYERS][1]);
        }
        else {
            elemsInLayer = inputs[VarPlacement::TEMPERATURE].getDimensions()[2];
            layerHeight = inputs[VarPlacement::TISSUE_SIZE][2];
        }
        if (inputs.size() > VarPlacement::USE_ALL_CPUS) {
            useAllCPUs = inputs[VarPlacement::USE_ALL_CPUS][0];
        }
        if (inputs.size() > VarPlacement::USE_GPU) {
            // Control GPU usage
            bool tempGPU = inputs[VarPlacement::USE_GPU][0];
            if (this->useGPU != tempGPU)
                this->createAllMatrices = true; // if we switch GPU usage createMatrices has to be true
            this->useGPU = tempGPU;
        }
        if (inputs.size() > VarPlacement::ALPHA) {
            this->simulator.alpha_ = inputs[VarPlacement::ALPHA][0];
        }
        if (inputs.size() > VarPlacement::SILENT_MODE) {
            this->silentMode = inputs[VarPlacement::SILENT_MODE][0];
            this->simulator.silentMode = this->silentMode;
        }
        if (inputs.size() > VarPlacement::NN1D) {
            Nn1d = inputs[VarPlacement::NN1D][0];
        }
        if (inputs.size() > VarPlacement::CREATE_MAT) {
            // this parameter is primarily here for debugging and timing tests. Not really practical for someone
            // to use this while they're running simulations
            this->createAllMatrices = this->createAllMatrices || inputs[VarPlacement::CREATE_MAT][0];
        }
    }

    bool checkForMatrixReset(int Nn1d, std::vector<std::vector<std::vector<float>>>& T0, std::vector<std::vector<std::vector<float>>>& FluenceRate,
        float tissueSize[3],float layerHeight, int elemsInLayer,int boundaryType[6]) {
        // this function checks to see if we need to recreate the global element matrices
        this->createAllMatrices = false;

        // if the number of nodes in 1 dimension for an element changes, we need to reconstruct the matrices
        if (this->simulator.Nn1d != Nn1d) {
            this->createAllMatrices = true;
        }
        // if the size of our temperature vector changes, we need to reconstruct the matrices
        if (T0.size() != this->simulator.nodesPerAxis[0]) this->createAllMatrices = true; 
        if (T0[0].size() != this->simulator.nodesPerAxis[1]) this->createAllMatrices = true;
        if (T0[0][0].size() != this->simulator.nodesPerAxis[2]) this->createAllMatrices = true;

        // if our tissue size has changed we need to reconstruct all matrices because our jacobian has changed
        if (tissueSize[0] != this->simulator.tissueSize[0]) this->createAllMatrices = true;
        if (tissueSize[1] != this->simulator.tissueSize[1]) this->createAllMatrices = true;
        if (tissueSize[2] != this->simulator.tissueSize[2]) this->createAllMatrices = true;

        //if layer height or layer size changed then again we have a jacobian change
        if (abs(layerHeight - this->simulator.layerHeight) > 0.0001) this->createAllMatrices = true;
        if (elemsInLayer != elemsInLayer) this->createAllMatrices = true;

        // check if any boundaries have changed
        for (int b = 0; b < 6; b++) {
            if (boundaryType[b] != this->simulator.boundaryType[b]) {
                this->createAllMatrices = true;
                break;
            }
        }
        
        if (this->createAllMatrices) {
            stream << "Need to recreate Matrices" << std::endl;
            displayOnMATLAB(stream);
        }

        return this->createAllMatrices;
    }

    void resetTimer(){
        timeRef = std::chrono::steady_clock::now();
    }

    void printDuration(const std::string& message) {
        auto stopTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (stopTime - timeRef);
        stream << message << duration.count() / 1000000.0 << " s" << std::endl;	
        displayOnMATLAB(stream);
    }

#ifdef USE_CUDA
    void initializeGPU(){
        if (!gpuHandle){
            gpuHandle = new GPUTimeIntegrator();
        }
        simulator.buildMatrices();
        gpuHandle->setAlpha(simulator.alpha_);
        gpuHandle->setDeltaT(simulator.deltaT_);
        gpuHandle->setModel(&simulator);
        gpuHandle->initializeWithModel();
    }

    void multiStepGPU(float totalTime){
        auto startTime = std::chrono::steady_clock::now();
        int numSteps = round(totalTime / simulator.deltaT_);
        simulator.initializeSensorTemps(numSteps);
        simulator.updateTemperatureSensors(0);
        for (int i = 1; i <= numSteps; i++) {
            gpuHandle->singleStepWithUpdate();
            simulator.updateTemperatureSensors(i);
        }
    }
#endif 

};
