#include <vector>
#include <iostream>

#include "../headers/activations/relu.h"
#include "../headers/activations/sigmoid.h"
#include "../headers/layers/linear.h"


void printMatrixContent(std::vector<std::vector<double>> inputMatrix) {
    for (int i = 0; i < inputMatrix.size(); i++) {
        for (int j = 0; j < inputMatrix[0].size(); j++) {
            std::cout << inputMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    std::vector<int> labels = {1, 2, 1, 2};
    std::vector<std::vector<double>> inputTensor = {{1, 1, 1, 1},
                                                    {2, 2, 2, 2},
                                                    {1, 1, 1, 2},
                                                    {2, 2, 2, 1}};

    int inputShape = inputTensor[1].size();
    int outputNeurons = 5;

    Linear linearLayer1(inputShape, outputNeurons);
    Linear linearLayer2(outputNeurons, 1);

    linearLayer1.forward(inputTensor);
    linearLayer1.output = relu(linearLayer1.output);
    linearLayer2.forward(linearLayer1.output);
    linearLayer2.output = relu(linearLayer2.output);

    std::vector<std::vector<double>> output = sigmoid(linearLayer2.output);

    printMatrixContent(output);

    return 0;
}
