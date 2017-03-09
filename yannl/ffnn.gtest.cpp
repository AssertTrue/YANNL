#include <gtest/gtest.h>

#include "ffnn.hpp"

using namespace yannl;

float Tolerance = 0.00001f;

TEST(ffnn, evaluate_no_hidden_layers)
{
    FFNN ffnn(1, 1, 0, 0);

    EXPECT_EQ(ffnn.numberOfInputs(), 1);
    EXPECT_EQ(ffnn.numberOfOutputs(), 1);
    EXPECT_EQ(ffnn.numberOfHiddenLayers(), 0);
    EXPECT_EQ(ffnn.numberOfNodesPerHiddenLayer(), 0);
    EXPECT_EQ(ffnn.numberOfWeights(), 2);

    ffnn.getWeights()[0] = 0.8f;
    ffnn.getWeights()[1] = 0.5f;

    std::vector<float> output = evaluate({ 1.2f }, ffnn);

    EXPECT_NEAR(output[0], 0.61301f, Tolerance);
}

TEST(ffnn, evaluate_invalid_network)
{
    EXPECT_DEATH(FFNN(0, 0, 0, 0), "");
    EXPECT_DEATH(FFNN(1, 0, 0, 0), "");
    EXPECT_DEATH(FFNN(0, 1, 0, 0), "");
    EXPECT_DEATH(FFNN(-1, 1, 0, 0), "");
    EXPECT_DEATH(FFNN(1, -1, 0, 0), "");
    EXPECT_DEATH(FFNN(1, 1, -1, 0), "");
    EXPECT_DEATH(FFNN(1, 1, 0, -1), "");
    EXPECT_DEATH(FFNN(1, 1, 1, 0), ""); // Layer with no nodes

    // Too few inputs
    FFNN network(1, 1, 0, 0);

    EXPECT_DEATH(evaluate({}, network), "");
    EXPECT_DEATH(evaluate({ 1, 2 }, network), "");
}

TEST(ffnn, evaluate_with_hidden_layers)
{
    FFNN ffnn(2, 4, 2, 3);

    EXPECT_EQ(ffnn.numberOfInputs(), 2);
    EXPECT_EQ(ffnn.numberOfOutputs(), 4);
    EXPECT_EQ(ffnn.numberOfHiddenLayers(), 2);
    EXPECT_EQ(ffnn.numberOfNodesPerHiddenLayer(), 3);
    EXPECT_EQ(ffnn.numberOfWeights(), 37);

    float weights[] = { -0.996777454f, -0.863441195f, 0.470315692f, -0.377533666f, 0.239034347f, -0.700948593f, 0.727923212f, 0.032970345f, 0.897148472f, 0.155163534f, -0.348505041f, -0.690309079f, -0.342748347f, 0.489008183f, 0.401770705f, 0.774834195f, 0.550771839f, 0.213473774f, -0.540993331f, -0.170803675f, 0.277790251f, -0.556754311f, -0.76309386f, 0.378873368f, 0.49559025f, -0.643425222f, 0.507739436f, -0.258631964f, -0.26149227f, -0.651029465f, 0.545696603f, -0.691562398f, 0.700588407f, 0.755926461f, 0.80508147f, -0.642217184f, 0.676940276f };
    memcpy(ffnn.getWeights(), weights, 37 * sizeof(float));
    std::vector<float> output = evaluate({ 1.2f, 1.3f }, ffnn);

    EXPECT_NEAR(output[0], 0.26447f, Tolerance);
    EXPECT_NEAR(output[1], 0.53950f, Tolerance);
    EXPECT_NEAR(output[2], 0.28271f, Tolerance);
    EXPECT_NEAR(output[3], 0.46847f, Tolerance);
}