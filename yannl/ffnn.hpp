#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <cmath>

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

            mWeightCount = mNumberOfHiddenLayers > 0 ? mNumberOfOutputs * (mNumberOfNodesPerHiddenLayer + 1) + mNumberOfNodesPerHiddenLayer * (mNumberOfInputs + 1) + (mNumberOfHiddenLayers - 1) * mNumberOfNodesPerHiddenLayer * (mNumberOfNodesPerHiddenLayer + 1) : mNumberOfOutputs * (mNumberOfInputs + 1);
            mWeights = new float[mWeightCount];
        }

        ~FFNN()
        {
            delete[] mWeights;
        }

        int numberOfInputs() const { return mNumberOfInputs; }
        int numberOfOutputs() const { return mNumberOfOutputs; }
        int numberOfHiddenLayers() const { return mNumberOfHiddenLayers; }
        int numberOfNodesPerHiddenLayer() const { return mNumberOfNodesPerHiddenLayer; }
        int numberOfWeights() { return mWeightCount; }

        float * getWeights() { return mWeights; }

    private:

        float * mWeights;
        int mWeightCount;
        int mNumberOfInputs;
        int mNumberOfOutputs;
        int mNumberOfHiddenLayers;
        int mNumberOfNodesPerHiddenLayer;
    };

    std::vector<float> evaluate(const std::vector<float> & aInputs, FFNN & aFFNN)
    {
        assert(aInputs.size() == aFFNN.numberOfInputs());

        float * weights = aFFNN.getWeights();
        float bias = -1;

        int weightIndex = 0;
        int outputBufferSize = std::max(aFFNN.numberOfOutputs(), aFFNN.numberOfNodesPerHiddenLayer());
        int inputBufferSize = std::max(aFFNN.numberOfInputs(), outputBufferSize);
        float * inputbuffer = new float[inputBufferSize];
        float * outputbuffer = new float[outputBufferSize];
        memcpy(inputbuffer, &aInputs[0], aInputs.size() * sizeof(float));
        memset(outputbuffer, 0, outputBufferSize * sizeof(float));
        int inputCount = aInputs.size();

        for (int layer = 0; layer < aFFNN.numberOfHiddenLayers() + 1; ++layer)
        {
            int outputCount = layer == aFFNN.numberOfHiddenLayers()  ? aFFNN.numberOfOutputs() : aFFNN.numberOfNodesPerHiddenLayer();
            memset(outputbuffer, 0, outputCount * sizeof(float));

            for (int node = 0; node < outputCount; ++node)
            {
                float & sum = outputbuffer[node];

                for (int inputIndex = 0; inputIndex < inputCount; ++inputIndex)
                {
                    sum += weights[weightIndex++] * inputbuffer[inputIndex];
                }
                sum += weights[weightIndex++] * bias;
                sum = (1 / (1 + exp(-sum)));
            }

            inputCount = outputCount;
            memcpy(inputbuffer, outputbuffer, outputCount * sizeof(float));
        }

        std::vector<float> result(inputbuffer, inputbuffer + inputCount);

        delete[] inputbuffer;
        delete[] outputbuffer;

        return result;
    }
}
