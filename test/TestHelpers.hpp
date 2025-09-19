#pragma once
#include <Eigen/Sparse>
#include <gtest/gtest.h>

inline void compareTwoMats(const Eigen::SparseMatrix<float,Eigen::RowMajor>& mat1,
                           const Eigen::SparseMatrix<float,Eigen::RowMajor>& mat2, float epsilon = 0.000001)
{
    for (int k = 0; k < mat1.outerSize(); ++k) {
        Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator itA(mat1, k);
        Eigen::SparseMatrix<float,Eigen::RowMajor>::InnerIterator itB(mat2, k);

        for (; itA && itB; ++itA, ++itB) {
            EXPECT_EQ(itA.row(), itB.row()) << "Row mismatch at outer " << k;
            EXPECT_EQ(itA.col(), itB.col()) << "Col mismatch at outer " << k;
            ASSERT_NEAR(itA.value(), itB.value(),epsilon) 
                << "Value mismatch at (" << itA.row() << "," << itA.col() << ")";
        }
    }
}

inline void compareTwoVectors(const Eigen::VectorXf& vec1, const Eigen::VectorXf& vec2, float epsilon = 0.000001)
{
    for (int k = 0; k < vec1.size(); ++k) {
        ASSERT_NEAR(vec1(k), vec2(k), epsilon)<< "Value mismatch at (" << k << ")";
    }
}