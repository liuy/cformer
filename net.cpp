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

tensor& categorical_cross_entropy(tensor &y_true, tensor &y_pred)
{
    return -(y_true*y_pred.log()).sum(1);
}

float categorical_accuracy(tensor &y_true, tensor &y_pred)
{
    return af::sum<float>(argmax(y_true.data) == argmax(y_pred.data)) / y_true.data.dims(0);
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

/**
 * Stochastic Gradient Descent (SGD) with momentum and weight decay
 *
 * Weight decay is L2 regularization to simplify the complexity of the model
 * by penalizing large weights. It is implemented by adding a term(w^2) to the loss
 * function. So
 *      new_grad = grad + weight_decay * w.
 *
 * Momentum is a method that helps accelerate SGD in the relevant direction and
 * dampens oscillations. It is implemented by adding a fraction of the gradients
 * of the past time step, stored as t->velocity, to the current gradient. So
 *      t->velocity  = momentum * t->velocity  + (1-momentum) * t->grad
 * https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
 */
void SGD::step(void)
{
    for (auto &t : params) {
        if (weight_decay > 0.0)
            t->grad += weight_decay * t->data;

        if (momentum > 0.0) {
            t->velocity  = momentum * t->velocity  + (1-momentum) * t->grad;
            t->data -= lr * t->velocity ;
        } else
            t->data -= lr * t->grad;

        t->grad = 0;
    }
}

void seqnet::train(data &set, trainer &tr)
{
    size_t n = set.num_examples();
    printf("| Epoch | Time Used | Train Loss | Train Accu |\n");
    for (int i = 0; i < tr.epochs; i++) {
        af::timer::start();
        for (int j = 0; j < n; j += tr.batch_size) {
            tensor x_batch(set.train_x.data.rows(j, j + tr.batch_size - 1));
            tensor y_true(set.train_y.data.rows(j, j + tr.batch_size- 1));
            tensor &y_pred = forward(x_batch);
            tensor &loss = tr.loss_fn(y_true, y_pred);

            loss.backward();
            tr.optimizer.step();
            float batch_loss = af::sum<float>(loss.data) / tr.batch_size;
            float batch_accu = tr.metrics_fn(y_true, y_pred);
            update_loss_metrics(batch_loss, batch_accu, i, j + tr.batch_size >= n);
            loss.destroy_graph();
        }
    }
}
