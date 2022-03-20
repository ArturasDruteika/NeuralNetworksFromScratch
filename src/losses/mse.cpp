#include <vector>
#include <cmath>

#include "../../headers/losses/mse.h"

double mseLoss(const std::vector<double> &labels, const std::vector<std::vector<double>> &predictionMatrix)
{
    double totalSquaredDifference = 0;

    for (int i = 0; i < predictionMatrix.size(); i++)
    {
        totalSquaredDifference += pow(labels[i] - predictionMatrix[i][0], 2);
    }

    return totalSquaredDifference / predictionMatrix.size();
}