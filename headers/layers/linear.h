#pragma once

#include <vector>


class Linear
{
private:
    static std::vector<std::vector<double>> transposeMatrix(std::vector<std::vector<double>> inputMatrix);

    [[nodiscard]] std::vector<std::vector<double>> initializeWeights() const;

    [[nodiscard]] std::vector<double> initializeBiases() const;

//    [[nodiscard]] static std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>> &firstMatrix,
//                                                                     const std::vector<std::vector<double>> &secondMatrix);

    std::vector<std::vector<double>> addBias(std::vector<std::vector<double>> &inputMatrix);


public:
    int inputShape;
    int outputShape;

    Linear(int inputShape, int outputShape);

    std::vector<std::vector<double>> weightsMatrix;
    std::vector<double> biases;
    std::vector<std::vector<double>> output;

    void forward(std::vector<std::vector<double>> &inputTensor);

    void backward(std::vector<std::vector<double>> &outputError, double learningRate) const;

    [[nodiscard]] static std::vector<std::vector<double>> dotProduct(const std::vector<std::vector<double>> &firstMatrix,
                                                                     const std::vector<std::vector<double>> &secondMatrix);

    ~Linear();
};