#include <iostream>
#include <random>
#include <vector>
#include "../../headers/layers/linear.h"


Linear::Linear(int inputTensorShape, int outputTensorShape)
{
    inputShape = inputTensorShape;
    outputShape = outputTensorShape;

    weightsMatrix = this->initializeWeights();
    biases = this->initializeBiases();
}

std::vector<std::vector<double>> Linear::initializeWeights() const
{
    std::vector<std::vector<double>> layerWeights;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int n = 0; n < this->outputShape; n++)
    {
        std::vector<double> rowVector;
        rowVector.reserve(this->inputShape);

        for (int i = 0; i < this->inputShape; i++)
        {
            rowVector.push_back(0.1 * dis(gen));
        }

        layerWeights.push_back(rowVector);
    }

    return layerWeights;
}

std::vector<double> Linear::initializeBiases() const
{
    std::vector<double> layerBiases;
    layerBiases.reserve(this->outputShape);

    for (int i = 0; i < this->outputShape; i++)
    {
        layerBiases.push_back(0.0);
    }

    return layerBiases;

}

std::vector<std::vector<double>> Linear::transposeMatrix(std::vector<std::vector<double>> matrixToTranspose)
{
    std::vector<std::vector<double>> transposedMatrix;

    for (int i = 0; i < matrixToTranspose[0].size(); i++)
    {
        std::vector<double> rowVector;
        rowVector.reserve(matrixToTranspose[0].size());

        for (auto &j: matrixToTranspose)
        {
            rowVector.push_back(j[i]);
        }

        transposedMatrix.push_back(rowVector);
    }

    return transposedMatrix;
}

std::vector<std::vector<double>> Linear::dotProduct(const std::vector<std::vector<double>> &firstMatrix,
                                                    const std::vector<std::vector<double>> &secondMatrix)
{
    std::vector<std::vector<double>> outputMatrix;
    std::vector<std::vector<double>> transposedSecondMatrix = transposeMatrix(secondMatrix);

    for (int i = 0; i < firstMatrix.size(); i++)
    {
        std::vector<double> rowVector;
        rowVector.reserve(secondMatrix[0].size());

        for (int j = 0; j < transposedSecondMatrix[0].size(); j++)
        {
            double result = 0;

            for (int k = 0; k < firstMatrix[0].size(); k++)
            {
                result += firstMatrix[i][k] * transposedSecondMatrix[k][j];
            }

            rowVector.push_back(result);
        }

        outputMatrix.push_back(rowVector);
    }

    return transposeMatrix(outputMatrix);
}

std::vector<std::vector<double>> Linear::addBias(std::vector<std::vector<double>> &inputMatrix)
{
    std::vector<std::vector<double>> output;

    for (int i = 0; i < inputMatrix.size(); i++)
    {
        std::vector<double> rowVector;
        rowVector.reserve(inputMatrix[0].size());

        for (int j = 0; j < inputMatrix[0].size(); j++)
        {
            rowVector.push_back(inputMatrix[i][j] + this->biases[i]);
        }

        output.push_back(rowVector);
    }

    return output;
}

void Linear::updateTrainableParameters(const std::vector<std::vector<double>> &weightsError,
                                       const std::vector<std::vector<double>> &outputError,
                                       double learningRate)
{
    for (int i = 0; i < weightsError.size(); i++)
    {
        for (int j = 0; j < weightsError[0].size(); j++)
        {
            this->weightsMatrix[i][j] = this->weightsMatrix[i][j] - learningRate * weightsError[i][j];
        }

        this->biases[i] = this->biases[i] - learningRate * outputError[i][0];
    }
}

void Linear::forward(std::vector<std::vector<double>> &inputMatrix)
{
    std::vector<std::vector<double>> output = dotProduct(this->weightsMatrix, inputMatrix);
    this->outputMatrix = addBias(output);
}


void Linear::backward(std::vector<std::vector<double>> &outputError,
                      const std::vector<std::vector<double>> &inputMatrix,
                      double learningRate)
{
    std::vector<std::vector<double>> inputError = dotProduct(outputError, this->weightsMatrix);
    std::vector<std::vector<double>> transposedInputMatrix = transposeMatrix(inputMatrix);

    outputError = transposeMatrix(outputError);

    std::vector<std::vector<double>> weightsError = dotProduct(transposedInputMatrix, outputError);

    updateTrainableParameters(weightsError, outputError, learningRate);


}

Linear::~Linear() = default;

