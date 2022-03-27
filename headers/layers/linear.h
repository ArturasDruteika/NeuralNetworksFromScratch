#pragma once

#include <vector>


class Linear
{
private:
    static std::vector<std::vector<double>> transposeMatrix(std::vector<std::vector<double>> inputMatrix);

    [[nodiscard]] std::vector<std::vector<double>> initializeWeights() const;

    [[nodiscard]] std::vector<double> initializeBiases() const;

    std::vector<std::vector<double>> addBias(std::vector<std::vector<double>> &inputMatrix);

    void updateTrainableParameters(const std::vector<std::vector<double>> &weightsError,
                                   const std::vector<std::vector<double>> &outputError,
                                   double learningRate);


public:
    int inputShape;
    int outputShape;

    Linear(int inputShape, int outputShape);

    std::vector<std::vector<double>> weightsMatrix;
    std::vector<double> biases;
    std::vector<std::vector<double>> outputMatrix;

    void forward(std::vector<std::vector<double>> &inputTensor);

    void backward(std::vector<std::vector<double>> &outputError,
                  const std::vector<std::vector<double>> &inputMatrix,
                  double learningRate);

    [[nodiscard]] static std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>> &firstMatrix,
                                                                     const std::vector<std::vector<double>> &secondMatrix);

    ~Linear();
};