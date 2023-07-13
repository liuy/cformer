#include "cformer.h"

tensor& linear::forward(tensor &x)
{
    tensor &y = x.matmul(weight);
    if (!no_bias) {
        broadcast(bias, x.data.dims(0));
        y += bias;
    }
    switch(act) {
    case ReLU:
        return y.relu();
    case Sigmoid:
        return y.sigmoid();
    case Tanh:
        return y.tanh();
    case None:
        return y;
    default:
        panic("Unknown activation function %d", act);
    }
}
