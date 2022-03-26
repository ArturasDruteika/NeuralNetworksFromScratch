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

std::vector<std::vector<double>> msePrime(const std::vector<double> &labels, const std::vector<std::vector<double>> &predictionMatrix)
{
    std::vector<std::vector<double>> grad;
    grad.reserve(labels.size());

    for (int i = 0; i < predictionMatrix.size(); i++)
    {
        grad.push_back({2 * (predictionMatrix[i][0] - labels[i]) / labels.size()});
    }

    return grad;

}