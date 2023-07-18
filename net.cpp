#include "cformer.h"

tensor& linear::forward(tensor &x)
{
    tensor &y = x.matmul(weight);
    if (!no_bias)
        y += bias.bdim0(x.data.dims(0));
    switch(act) {
    case ReLU:
        return y.relu();
    case Sigmoid:
        return y.sigmoid();
    case Tanh:
        return y.tanh();
    case Softmax:
        return y.exp()/y.exp().bsum(1);
    case None:
        return y;
    default:
        panic("Unknown activation function %d", act);
    }
}

tensor& seq_net::forward(tensor &x)
{
    tensor *y = &x;
    for (auto layer : layers)
        y = &layer->forward(*y);
    return *y;
}

static tensor& categorical_cross_entropy(tensor &y_true, tensor &y_pred, bool from_logits=false)
{
    if (from_logits)
        return  (y_true * (y_pred.exp().bsum(1).log() - y_pred)).sum(1);
    return -(y_true*y_pred.log()).sum(1);
}

void seq_net::train(tensor &x, tensor &y, float lr, int bacth_size, int epoch)
{
    int n = x.data.dims(0);
    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < n; j += bacth_size) {
            tensor x_batch(x.data.rows(j, j + bacth_size - 1));
            tensor y_true(y.data.rows(j, j + bacth_size - 1));
            tensor &y_pred = forward(x_batch);
            tensor &loss = categorical_cross_entropy(y_true, y_pred);

            // if (j == 0)
            //     loss.print_graph();
            loss.backward();
            //af_print(y_pred.data);
            if (j % 20000 == 0)
                printf("epoch %d, %d, %f\n", i, j, af::sum<float>(loss.data) / bacth_size);
            //af_print(loss.data);
            for (auto t : params) {
                t->data -= lr * t->grad;
                // af_print(t->data);
                // af_print(t->grad);
                t->grad = 0;
            }
            loss.destroy_graph();
        }
    }
}
