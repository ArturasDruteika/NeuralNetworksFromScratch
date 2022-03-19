#include <iostream>
#include <vector>
#include "../../headers/activations/relu.h"

std::vector<std::vector<double>> relu(std::vector<std::vector<double>> inputMatrix) {
    std::vector<std::vector<double>> tempMatrix;

    for (int i = 0; i < inputMatrix.size(); i++) {
        std::vector<double> rowVector;
        rowVector.reserve(inputMatrix[0].size());
        for (int j = 0; j < inputMatrix[0].size(); j++) {
            rowVector.push_back(std::max(0.0, inputMatrix[i][j]));
        }

        tempMatrix.push_back(rowVector);
    }

    return tempMatrix;
}

