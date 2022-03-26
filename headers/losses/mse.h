#pragma once

#include <vector>

double mseLoss(const std::vector<double> &labels, const std::vector<std::vector<double>> &predictionMatrix);
std::vector<std::vector<double>> msePrime(const std::vector<double> &labels, const std::vector<std::vector<double>> &predictionMatrix);