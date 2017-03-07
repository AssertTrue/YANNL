#pragma once

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
            mWeights.resize(numberOfWeights(), 0);
        }

        int numberOfInputs() const { return mNumberOfInputs; }
        int numberOfOutputs() const { return mNumberOfOutputs; }
        int numberOfHiddenLayers() const { return mNumberOfHiddenLayers; }
        int numberOfNodesPerHiddenLayer() const { return mNumberOfNodesPerHiddenLayer; }
        int numberOfWeights() { return (mNumberOfOutputs * (mNumberOfInputs + 1)); }

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
        const std::vector<float> & weights = aFFNN.getWeights();
        float bias = -1;

        float output = 0;
        for (float input : aInputs)
        {
            output += input * weights[0];
        }
        output += bias * weights[1];

        return { (1 / (1 + exp(-output))) };
    }
}