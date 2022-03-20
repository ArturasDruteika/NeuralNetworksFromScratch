#include <vector>
#include <cmath>
#include "../../headers/activations/sigmoid.h"

std::vector<std::vector<double>> sigmoid(std::vector<std::vector<double>> inputMatrix)
{
    std::vector<std::vector<double>> resultMatrix;

    for (int i = 0; i < inputMatrix.size(); i++)
    {
        std::vector<double> rowVector;
        rowVector.reserve(inputMatrix[0].size());

        for (int j = 0; j < inputMatrix[0].size(); j++)
        {
            rowVector.push_back(1.0 / (1.0 + exp(inputMatrix[i][j])));
        }

        resultMatrix.push_back(rowVector);
    }

    return resultMatrix;
}