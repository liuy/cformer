#include "cformer.h"

tensor& Linear::forward(tensor &x, bool training)
{
    tensor &y = x.matmul(weight);
    if (!no_bias)
        y += bias.expandas(x);
    switch(act) {
    case ReLU:
        return y.relu();
    case Sigmoid:
        return y.sigmoid();
    case Tanh:
        return y.tanh();
    case Softmax:
        return y.softmax();
    case LogSoftmax:
        return y.logsm();
    case None:
        return y;
    default:
        panic("Unknown activation function %d", act);
    }
}

/**
 * Batch Normalization (BN) is a technique to improve the training speed and performance
 * of a neural network. It is a essentially a normalization of the output of a previous
 * activation layer by subtracting the batch mean and dividing by the batch standard deviation.
 * It is implemented by adding two trainable parameters, gamma(weight) and beta(bias), to the previous
 * activation layer. So the output of the BN layer is given by the following equation:
 *     y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
 * see more details at https://arxiv.org/pdf/1502.03167.pdf
 */
tensor& BatchNorm1d::forward(tensor &x, bool training)
{
    if (training) {
        x.forward();
        // Note: mean() and var() call on batch dimension
        array mean = af::mean(x.data, 0);
        array vari = af::var(x.data, AF_VARIANCE_POPULATION, 0);
        moving_mean.data = momentum * moving_mean.data + (1 - momentum) * mean;
        moving_vari.data = momentum * moving_vari.data + (1 - momentum) * vari;

        tensor &y = x.batchnorm(0);
        return y * weight.expandas(x) + bias.expandas(x);
    }
    tensor &y = (x - moving_mean.expandas(x)) / (moving_vari.expandas(x) + epsilon).pow(0.5);
    return y * weight.expandas(x) + bias.expandas(x);
}

/**
 * Dropout is a regularization technique to prevent overfitting by randomly dropping out some
 * neurons in the network. It is implemented by multiplying the output of a previous activation
 * layer by a mask of 0s and 1s. Divide the output by (1 - p) to keep the sum of the output
 * unchanged.
 *
 * Note: Dropout is only applied during training, not during inference.
 * see more details at https://www.cs.toronto.edu/~hinton/absps/JMLRDropout.pdf
 */
tensor& Dropout::forward(tensor& x, bool training)
{
    if (training) {
        x.forward();
        auto mask = af::randu(x.data.dims()) > p;
        tensor &y = x * mask / (1.0 - p);
        return y;
    }
    return x;
}

tensor& seqnet::forward(tensor &x, bool training)
{
    tensor *y = &x;
    for (auto layer : layers)
        y = &layer->forward(*y, training);
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

tensor& log_softmax_cross_entropy(tensor &y_true, tensor &y_pred)
{
    return -(y_true * y_pred).sum(1);
}

static inline void update_loss_metrics(float loss, float accu, size_t epoch, bool end)
{
    static std::vector<float> epoch_loss;
    static std::vector<float> epoch_accu;
    epoch_loss.push_back(loss);
    epoch_accu.push_back(accu);

    if (!end)
        return;

    float avg_loss = std::accumulate(epoch_loss.begin(), epoch_loss.end(), 0.0) / epoch_loss.size();
    float avg_accu = std::accumulate(epoch_accu.begin(), epoch_accu.end(), 0.0) / epoch_accu.size();
    printf("| %-5zu | %-9.1f | %-10.8f | %-10.8f |\n", epoch, af::timer::stop(), avg_loss, avg_accu);
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
    for (auto &p : params) {
        if (weight_decay > 0.0)
            p.param->grad += weight_decay * p.param->data;

        if (momentum > 0.0) {
            // Andrew NG's version works a little bit better than PyTorch's but we choose pytorch's
            // version for simplicity (less computation)
            p.velocity  = momentum * p.velocity + p.param->grad;
            p.param->data -= nesterov ? lr * (momentum * p.velocity + p.param->grad) : lr * p.velocity;

        } else
            p.param->data -= lr * p.param->grad;

        p.param->zero_grad();
    }
}

// For more details, see https://arxiv.org/abs/1412.6980
void Adam::step(void)
{
    for (auto &p : params) {
        static int t = 0;
        t++;

        if (weight_decay > 0.0)
            p.param->grad += weight_decay * p.param->data;

        p.mean = beta1 * p.mean + (1 - beta1) * p.param->grad;
        p.variance = beta2 * p.variance + (1 - beta2) * p.param->grad * p.param->grad;
        array mean_hat = p.mean / (1 - pow(beta1, t));
        array variance_hat = p.variance / (1 - pow(beta2, t));
        p.param->data -= lr * mean_hat / (af::sqrt(variance_hat) + epsilon);

        p.param->zero_grad();
    }
}

void seqnet::train(data &set, trainer &tr)
{
    size_t n = set.num_examples();
    set.init_train_idx(tr.batch_size);
    printf("| Epoch | Time Used | Train Loss | Train Accu |\n");
    for (size_t i = 0; i < tr.epochs; i++) {
        af::timer::start();
        for (std::vector<size_t>::iterator it = set.train_idx.begin(); it != set.train_idx.end(); it++) {
            tensor x_batch, y_true;
            set.get_mini_batch(x_batch, y_true, *it, tr.batch_size);
            tensor &y_pred = forward(x_batch, true);
            tensor &loss = tr.loss_fn(y_true, y_pred);

            loss.backward();
            tr.optimizer.step();
            float batch_loss = af::mean<float>(loss.data);
            float batch_accu = tr.metrics_fn(y_true, y_pred);
            update_loss_metrics(batch_loss, batch_accu, i, it == set.train_idx.end() - 1);
            loss.destroy_graph();
        }
        if (set.shuffle)
            set.shuffle_train_idx();
    }
}

void seqnet::summary(void)
{
    int i = 0;
    size_t total_params = 0;
    printf("\n%s:\n", name);
    printf("+-------+---------+-------+--------+------+------------+------------+\n");
    printf("| Layer | Name    | Input | Output | Bias | Activation | Parameters |\n");
    printf("+-------+---------+-------+--------+------+------------+------------+\n");
    auto param_num = [](layer *l) {
            if (l->no_bias)
                return l->weight.data.elements();
            return l->weight.data.elements() + l->bias.data.elements();
        };
    for (auto layer : layers) {
        size_t np = param_num(layer);
        total_params += np;
        printf("| %-5d | %-7s | %-5lld | %-6lld | %-4s | %-10s | %-'10ld |\n", i++, layer->name,
            layer->weight.data.isempty() ? 0 : layer->weight.data.dims(0),
            layer->weight.data.isempty() ? 0 : layer->weight.data.dims(1), layer->no_bias ? "None" : "Yes",
            activ_name[layer->act], np);
    }
    printf("+-------+---------+-------+--------+------+------------+------------+\n");
    printf("Total params: %ld\n", total_params);
    printf("Running on:\n");
    af::info();
    printf("\n");
}
