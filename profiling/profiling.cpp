#include <yannl/ffnn.hpp>

void main()
{
    for (int i = 0; i < 10; ++i)
    {
        yannl::FFNN nn(500, 500, 100, 500);

        std::vector<float> input(500, 0);

        std::vector<float> result = yannl::evaluate(input, nn);
    }
}