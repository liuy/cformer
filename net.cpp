#include "cformer.h"

tensor& softmax(tensor &x)
{
    tensor &exp = (x - x.bmax(1)).exp();
    return exp/exp.bsum(1);
}

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
        return softmax(y);
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
    printf("| %-5d | %-9.1f | %-10.8f | %-10.8f |\n", epoch, af::timer::stop(), avg_loss, avg_accu);
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
 *      t->velocity = momentum * t->velocity + (1-momentum) * t->grad (Andrew NG)
 *      t->velocity = momentum * t->velocity + t->grad (PyTorch)
 *      v_lookahead = momentum * v + grad
 * https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
 */
void SGD::step(void)
{
    for (auto &t : params) {
        if (weight_decay > 0.0)
            t->grad += weight_decay * t->data;

        if (momentum > 0.0) {
            if (unlikely(t->velocity.isempty()))
                t->velocity = af::constant(0, t->grad.dims());

            // Andrew NG's version works a little bit better than PyTorch's but we choose pytorch's
            // version for simplicity (less computation)
            t->velocity  = momentum * t->velocity  + t->grad;
            t->data -= nesterov ? lr * (momentum * t->velocity + t->grad) : lr * t->velocity;
        } else
            t->data -= lr * t->grad;

        t->zero_grad();
    }
}

// For more details, see https://arxiv.org/abs/1412.6980
void Adam::step(void)
{
    for (auto &t : params) {
        if (weight_decay > 0.0)
            t->grad += weight_decay * t->data;

        if (unlikely(t->mean.isempty()))
            t->mean = af::constant(0, t->grad.dims());
        if (unlikely(t->variance.isempty()))
            t->variance = af::constant(0, t->grad.dims());

        t->mean = beta1 * t->mean + (1 - beta1) * t->grad;
        t->variance = beta2 * t->variance + (1 - beta2) * t->grad * t->grad;
        // array mean_hat = t->mean / (1 - beta1);
        // array variance_hat = t->variance / (1 - beta2);
        // t->data -= lr * mean_hat / (af::sqrt(variance_hat) + epsilon);
        // comment out bias correction for simplicity
        t->data -= lr * t->mean / (af::sqrt(t->variance) + epsilon);

        t->zero_grad();
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
    tr.optimizer.finish();
}
