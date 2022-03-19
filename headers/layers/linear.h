#pragma once

#include <vector>


class Linear {
private:
    std::vector<std::vector<double>> dotProduct(std::vector<std::vector<double>> inputMatrix);

    static std::vector<std::vector<double>> transposeMatrix(std::vector<std::vector<double>> inputMatrix);

    [[nodiscard]] std::vector<std::vector<double>> initializeWeights() const;

    [[nodiscard]] std::vector<double> initializeBiases() const;


public:
    int inputShape;
    int outputShape;

    Linear(int inputShape, int outputShape);

    std::vector<std::vector<double>> weightsMatrix;
    std::vector<double> biases;
    std::vector<std::vector<double>> output;

    void forward(std::vector<std::vector<double>> &inputTensor);

    ~Linear();
};