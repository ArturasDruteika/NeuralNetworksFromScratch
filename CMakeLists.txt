cmake_minimum_required(VERSION 3.21)
project(NeuralNetworksFromScratch)

set(CMAKE_CXX_STANDARD 23)

add_executable(neural_networks

        src/main.cpp
        headers/layers/conv2d.h
        headers/layers/linear.h

        headers/activations/relu.h
        headers/activations/softmax.h
        headers/activations/sigmoid.h

        src/layers/conv2d.cpp
        src/layers/linear.cpp

        src/activations/relu.cpp
        src/activations/softmax.cpp
        src/activations/sigmoid.cpp)
