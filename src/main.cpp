#include <iostream>
#include <vector>

#include "../headers/activations/relu.h"
#include "../headers/layers/linear.h"


int main() {
    std::vector<int> labels = {1, 2, 3};
    std::vector<std::vector<double>> inputTensor = {{1, 1, 1, 1},
                                                    {-2, -2, -2, -2},
                                                    {3, 3, 3, 3}};

    int inputShape = inputTensor[1].size();
    int outputNeurons = 5;
    int nClasses = 2;

    Linear linearLayer1(inputShape, outputNeurons);
    Linear linearLayer2(outputNeurons, nClasses);

    linearLayer1.forward(inputTensor);
    linearLayer1.output = relu(linearLayer1.output);

    for (const auto& row : linearLayer1.output) {
        for (const auto element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    linearLayer2.forward(linearLayer1.output);
    linearLayer2.output = relu(linearLayer2.output);

    for (const auto& row : linearLayer2.output) {
        for (const auto element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}



//    for (int i = 0; i < linearLayer.weightsMatrix.size(); i++) {
//        for (int k = 0; k < linearLayer.weightsMatrix[0].size(); k++){
//            std::cout << linearLayer.weightsMatrix[i][k] << " ";
//        }
//        std::cout << std::endl;
//
//    }
//
//    std::cout << std::endl;
//
//    std::vector<std::vector<double>> output = linearLayer.dotProduct(inputTensor);