#include "mex.hpp"
#include "mexAdapter.hpp"
#include <string>
#include <memory>
#include <iostream>
#include <algorithm>
#include "FEM_Simulator.h"

#ifdef USE_CUDA
#include "GPUTimeIntegrator.cuh"
#endif

/* Helper function to convert a matlab array to a std vector*/
inline std::vector<std::vector<std::vector<float>>> convertMatlabArrayTo3DVector(const matlab::data::Array& matlabArray) {
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

inline matlab::data::TypedArray<float> convertVectorToMatlabArray(const std::vector<std::vector<std::vector<float>>>& vec) {
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

/* Helper function to convert a matlab array to a std::vector<float>*/
inline std::vector<float> convertMatlabArrayToFloatVector(const matlab::data::Array& matlabArray) {
    // Get dimensions of the Matlab array
    size_t numElems = matlabArray.getNumberOfElements();
    // Create result vector to match the dimensions of the Matlab array
    std::vector<float> result(numElems);
    // Iterate through the Matlab array and fill the result vector
    for (size_t i = 0; i < numElems; ++i) {
        result[i] = static_cast<float>(matlabArray[i]);
    }
    return result;
}

inline std::vector<float> convertMatlabArrayToFloatVector(const matlab::data::Array& matlabArray) {
    // Get dimensions of the Matlab array
    size_t numElems = matlabArray.getNumberOfElements();
    // Create result vector to match the dimensions of the Matlab array
    std::vector<float> result(numElems);
    // Iterate through the Matlab array and fill the result vector
    for (size_t i = 0; i < numElems; ++i) {
        result[i] = static_cast<float>(matlabArray[i]);
    }
    return result;
}

Eigen::VectorXf convertMatlabArrayToEigenVector(const matlab::data::Array& matlabArray) {
    // Get dimensions of the Matlab array
    size_t numElems = matlabArray.getNumberOfElements();
    // Create the result vector to match the dimensions of the Matlab array
    Eigen::VectorXf result(numElems);
    // Iterate through the Matlab array and fill the result vector
    for (size_t i = 0; i < numElems; ++i) {
        result(i) = static_cast<float>(matlabArray[i]);
    }
    return result;
}

/* Helper function to generate an error message from given string,
 * and display it over MATLAB command prompt.
 */
inline void displayError(std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr, std::string errorMessage)
{
    matlab::data::ArrayFactory factory;
    matlabPtr->feval(u"error", 0, std::vector<matlab::data::Array>({
      factory.createScalar(errorMessage) }));
}

/* Helper function to generate messages to matlab command window
*/
inline void displayOnMATLAB(std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr, std::ostringstream& stream, bool silentMode) {
    // Pass stream content to MATLAB fprintf function
    if (!silentMode) {
        matlab::data::ArrayFactory factory;
        matlabPtr->feval(u"fprintf", 0,
            std::vector<matlab::data::Array>({ factory.createScalar(stream.str()) }));
        // Clear stream buffer
        stream.str("");
    }
}

bool strEquals(const std::string& a, const std::string& b)
{
    if (a.size() != b.size()) return false;

    return std::equal(a.begin(), a.end(), b.begin(),
        [](char c1, char c2) {
            return std::tolower(c1) == std::tolower(c2);
        }
    );
}

bool checkFieldNames(const std::vector<std::string>& requiredNames, const std::vector<std::string>& givenNames)
{
    bool allMatching = true;
    for (int i = 0; i < requiredNames.size(); i++) /* Check the field names*/
    {
        bool match hasField(givenNames, requiredNames[0]);
        allMatching |= match;
    }
    return allMatching;
}

bool hasField(const std::vector<std::string>& fieldNames, const std::string& target)
{
    for (const auto& fn : fieldNames) {
        if (fn == target) {
            return true;
        }
    }
    return false;
}
//inline void display3DVector(const std::vector<std::vector<std::vector<float>>>& vec, std::string title) {
//    // Get dimensions of the input vector
//    size_t dim1 = vec.size();
//    size_t dim2 = (dim1 > 0) ? vec[0].size() : 0;
//    size_t dim3 = (dim2 > 0) ? vec[0][0].size() : 0;
//
//    stream << "\n" << title << std::endl;
//    for (int k = 0; k < dim3; k++) {
//        for (int j = 0; j < dim2; j++) {
//            for (int i = 0; i < dim1; i++) {
//                stream << vec[i][j][k] << ", ";
//                displayOnMATLAB(stream);
//            }
//            stream << std::endl;
//        }
//        stream << std::endl;
//    }
//    displayOnMATLAB(stream);
//}