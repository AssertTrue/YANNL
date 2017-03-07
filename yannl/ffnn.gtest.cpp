#include <gtest/gtest.h>

#include "ffnn.hpp"

using namespace yannl;

float Tolerance = 0.00001f;

TEST(ffnn, evaluate)
{
    FFNN ffnn(1, 1, 0, 0);

    EXPECT_EQ(ffnn.numberOfInputs(), 1);
    EXPECT_EQ(ffnn.numberOfOutputs(), 1);
    EXPECT_EQ(ffnn.numberOfHiddenLayers(), 0);
    EXPECT_EQ(ffnn.numberOfNodesPerHiddenLayer(), 0);
    EXPECT_EQ(ffnn.numberOfWeights(), 2);

    ffnn.getWeights() = { 0.8f, 0.5f };

    std::vector<float> output = evaluate({ 1.2f }, ffnn);

    EXPECT_NEAR(output[0], 0.61301f, Tolerance);
}