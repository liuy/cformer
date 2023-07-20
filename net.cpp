#include "cformer.h"

tensor& linear::forward(tensor &x)
{
    tensor &y = x.matmul(weight);
    if (!no_bias)
        y += bias.bdim0(x);
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

tensor& seqnet::forward(tensor &x)
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

static void update_loss_metrics(float loss, float accu, int epoch, bool end)
{
    static std::vector<float> epoch_loss;
    static std::vector<float> epoch_accu;
    epoch_loss.push_back(loss);
    epoch_accu.push_back(accu);

    if (!end)
        return;

    float avg_loss = std::accumulate(epoch_loss.begin(), epoch_loss.end(), 0.0) / epoch_loss.size();
    float avg_accu = std::accumulate(epoch_accu.begin(), epoch_accu.end(), 0.0) / epoch_accu.size();
    printf("| %-5d | %-9.1f | %-10.5f | %-10.5f |\n", epoch, af::timer::stop(), avg_loss, avg_accu);
    epoch_loss.clear();
    epoch_accu.clear();
}

void seqnet::train(data &set, float lr, int bacth_size, int epoch)
{
    size_t n = set.num_examples();
    printf("| Epoch | Time Used | Train Loss | Train Accu |\n");
    for (int i = 0; i < epoch; i++) {
        af::timer::start();
        for (int j = 0; j < n; j += bacth_size) {
            tensor x_batch(set.train_x.data.rows(j, j + bacth_size - 1));
            tensor y_true(set.train_y.data.rows(j, j + bacth_size - 1));
            tensor &y_pred = forward(x_batch);
            tensor &loss = categorical_cross_entropy(y_true, y_pred);

            loss.backward();
            for (auto t : params) {
                t->data -= lr * t->grad;
                t->grad = 0;
            }
            float batch_loss = af::sum<float>(loss.data) / bacth_size;
            float batch_accu = categorical_accuracy(y_true, y_pred);
            update_loss_metrics(batch_loss, batch_accu, i, j + bacth_size >= n);
            loss.destroy_graph();
        }
    }
}
