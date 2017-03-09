#pragma once

#include <cassert>
#include <vector>

namespace yannl
{
    class FFNN
    {
    public:

        FFNN(int aNumberOfInputs, int aNumberOfOutputs, int aNumberOfHiddenLayers, int aNumberOfNodesPerHiddenLayer)
            : mNumberOfInputs(aNumberOfInputs)
            , mNumberOfOutputs(aNumberOfOutputs)
            , mNumberOfHiddenLayers(aNumberOfHiddenLayers)
            , mNumberOfNodesPerHiddenLayer(aNumberOfNodesPerHiddenLayer)
        {
            assert(aNumberOfInputs > 0);
            assert(aNumberOfOutputs > 0);
            assert(aNumberOfHiddenLayers >= 0);
            assert(aNumberOfNodesPerHiddenLayer >= 0);
            assert(aNumberOfHiddenLayers == 0 || aNumberOfNodesPerHiddenLayer > 0);

            int weightCount = mNumberOfHiddenLayers > 0 ? mNumberOfOutputs * (mNumberOfNodesPerHiddenLayer + 1) + mNumberOfNodesPerHiddenLayer * (mNumberOfInputs + 1) + (mNumberOfHiddenLayers - 1) * mNumberOfNodesPerHiddenLayer * (mNumberOfNodesPerHiddenLayer + 1) : mNumberOfOutputs * (mNumberOfInputs + 1);
            mWeights.resize(weightCount, 0);
        }

        int numberOfInputs() const { return mNumberOfInputs; }
        int numberOfOutputs() const { return mNumberOfOutputs; }
        int numberOfHiddenLayers() const { return mNumberOfHiddenLayers; }
        int numberOfNodesPerHiddenLayer() const { return mNumberOfNodesPerHiddenLayer; }
        int numberOfWeights() { return mWeights.size(); }

        std::vector<float> & getWeights() { return mWeights; }

    private:

        std::vector<float> mWeights;
        int mNumberOfInputs;
        int mNumberOfOutputs;
        int mNumberOfHiddenLayers;
        int mNumberOfNodesPerHiddenLayer;
    };

    std::vector<float> evaluate(const std::vector<float> & aInputs, FFNN & aFFNN)
    {
        assert(aInputs.size() == aFFNN.numberOfInputs());

        const std::vector<float> & weights = aFFNN.getWeights();
        float bias = -1;

        int weightIndex = 0;
        std::vector<float> inputs = aInputs;
        std::vector<float> outputs;

        for (int layer = 0; layer < aFFNN.numberOfHiddenLayers() + 1; ++layer)
        {
            outputs.clear();
            outputs.resize(layer == aFFNN.numberOfHiddenLayers() ? aFFNN.numberOfOutputs() : aFFNN.numberOfNodesPerHiddenLayer(), 0);
            for (int node = 0; node < (int)outputs.size(); ++node)
            {
                float & sum = outputs[node];

                for (int inputIndex = 0; inputIndex < (int)inputs.size(); ++inputIndex)
                {
                    sum += weights[weightIndex++] * inputs[inputIndex];
                }
                sum += weights[weightIndex++] * bias;
                sum = (1 / (1 + exp(-sum)));
            }

            inputs = outputs;
        }

        return outputs;
    }
}