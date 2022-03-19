#include <iostream>
#include <random>
#include <vector>
#include "../../headers/layers/linear.h"


Linear::Linear(int inputTensorShape, int outputTensorShape) {
    inputShape = inputTensorShape;
    outputShape = outputTensorShape;

    weightsMatrix = this->initializeWeights();
    biases = this->initializeBiases();
}

std::vector<std::vector<double>> Linear::initializeWeights() const {
    std::vector<std::vector<double>> layerWeights;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int n = 0; n < this->outputShape; n++) {
        std::vector<double> rowVector;
        rowVector.reserve(this->inputShape);

        for (int i = 0; i < this->inputShape; i++) {
            rowVector.push_back(0.1 * dis(gen));
        }

        layerWeights.push_back(rowVector);
    }

    return layerWeights;
}

std::vector<double> Linear::initializeBiases() const {
    std::vector<double> layerBiases;
    layerBiases.reserve(this->outputShape);

    for (int i = 0; i < this->outputShape; i++) {
        layerBiases.push_back(0.0);
    }

    return layerBiases;

}

void Linear::forward(std::vector<std::vector<double>> &inputMatrix) {
    this->output = this->dotProduct(inputMatrix);

}

std::vector<std::vector<double>> Linear::transposeMatrix(std::vector<std::vector<double>> matrixToTranspose) {
    std::vector<std::vector<double>> transposedMatrix;

    for (int i = 0; i < matrixToTranspose[0].size(); i++) {
        std::vector<double> rowVector;
        rowVector.reserve(matrixToTranspose[0].size());

        for (auto &j: matrixToTranspose) {
            rowVector.push_back(j[i]);
        }

        transposedMatrix.push_back(rowVector);
    }

    return transposedMatrix;
}

std::vector<std::vector<double>> Linear::dotProduct(std::vector<std::vector<double>> inputMatrix) {
    std::vector<std::vector<double>> outputMatrix;
    std::vector<std::vector<double>> transposedInputMatrix = transposeMatrix(inputMatrix);

    for (int i = 0; i < this->weightsMatrix.size(); i++) {
        std::vector<double> rowVector;
        rowVector.reserve(inputMatrix[0].size());

        for (int j = 0; j < transposedInputMatrix[0].size(); j++) {
            double result = 0;

            for (int k = 0; k < this->weightsMatrix[0].size(); k++) {
                result += this->weightsMatrix[i][k] * transposedInputMatrix[k][j];
            }

            rowVector.push_back(result + this->biases[i]);
        }

        outputMatrix.push_back(rowVector);
    }

    return transposeMatrix(outputMatrix);
}

Linear::~Linear() = default;

