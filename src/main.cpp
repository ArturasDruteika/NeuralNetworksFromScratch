#include <vector>
#include <iostream>

#include "../headers/activations/relu.h"
#include "../headers/activations/sigmoid.h"
#include "../headers/layers/linear.h"


void printMatrixContent(std::vector<std::vector<double>> inputMatrix)
{
    for (int i = 0; i < inputMatrix.size(); i++)
    {
        for (int j = 0; j < inputMatrix[0].size(); j++)
        {
            std::cout << inputMatrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}


int main()
{
    std::vector<int> labels = {0, 1, 0, 1};
    std::vector<std::vector<double>> inputTensor = {{1, 3, 5, 8},
                                                    {-9, -6, -1, -4},
                                                    {5, 6, 8, 2},
                                                    {-1, -2, -7, -2}};

    int inputShape = inputTensor[1].size();
    int outputNeurons = 5;

    Linear linearLayer1(inputShape, outputNeurons);
    Linear linearLayer2(outputNeurons, 1);

    linearLayer1.forward(inputTensor);
    linearLayer1.output = relu(linearLayer1.output);
    linearLayer2.forward(linearLayer1.output);
    linearLayer2.output = sigmoid(linearLayer2.output);

    printMatrixContent(linearLayer2.output);

    return 0;
}
