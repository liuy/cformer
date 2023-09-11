#include "cformer.h"
#include "pbar.h"

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

// TODO: implement memory efficient and flash attention.
static tensor& scaled_dot_product(tensor &q, tensor &k, tensor &v)
{
    tensor &qk = q.matmul(k.T());
    tensor &qk_scaled = qk / std::sqrt(q.data.dims(1));
    tensor &qk_scaled_sm = qk_scaled.softmax();
    return qk_scaled_sm.matmul(v);
}

/**
 * Multihead Attention is a type of attention mechanism that is used in the Transformer
 *
 * Input: x of shape (seq_len, embed_dim, batch_size)
 * Output: shape (seq_len, embed_dim, batch_size)
 */
tensor& multihead_attention::forward(tensor &x)
{
    tensor &qkv = x.matmul(weight_qkv);
    if (!no_bias)
        qkv += bias_qkv.expandas(x);

    dim_t seq_len = x.data.dims(0);
    dim_t batch_size = x.data.dims(2);
    dim_t head_dim = embed_dim / num_heads;
    // [seq_len, embed_dim * 3, batch_size] - > [seq_len, head_dim * 3, num_heads, batch_size]
    tensor &qkv_r = qkv.reshape({seq_len, head_dim * 3, num_heads, batch_size});
    tensor &q = qkv_r.slice(1, 0, head_dim - 1);
    tensor &k = qkv_r.slice(1, head_dim, head_dim * 2 - 1);
    tensor &v = qkv_r.slice(1, head_dim * 2, head_dim * 3 - 1);

    tensor &vals = scaled_dot_product(q, k, v);
    // [seq_len, head_dim, num_heads, batch_size] -> [seq_len, embed_dim, batch_size]
    tensor &vals_r = vals.reshape({seq_len, embed_dim, batch_size});
    tensor &out = vals_r.matmul(weight_o);
    if (!no_bias)
        out += bias_o.expandas(x);
    return out;
}

/**
 * Maps token indices to one-hot vectors, then projects to embedding space
 *
 * Input: x of shape (seq_len, batch_size)
 * Output: shape (seq_len, batch_size, out)
 */
tensor& Embedding::forward(tensor& x, bool training)
{
    x.forward();
    af::dim4 dims = x.data.dims();
    dims[2] = weight.data.dims(1); // (seq_len, batch_size, out)
    x.init(onehot(x.data, weight.data.dims(0))); // (seq_len * batch_size, in)
    return x.matmul(weight).reshape(dims);
}

static inline rnn_cell* rnn_cell_create(int in, int out, bool nb, const af::dtype t, rnn_t r)
{
    if (r == LSTM)
        return new lstm_cell(in, out, nb, t);
    else if (r == Simple)
        return new elman_cell(in, out, nb, t);
    else if (r == GRU)
        panic("GRU not implemented yet");
    else
        panic("Unknown RNN type %d", r);
}

RNN::RNN(int in, int out, int num_layers, rnn_t r, bool nb, const af::dtype t)
{
    name = rnn_name[r]; no_bias = nb;
    cells.reserve(num_layers);

    cells.emplace_back(rnn_cell_create(in, out, nb, t, r));
    for (int i = 0; i < num_layers - 1; i++)
        cells.emplace_back(rnn_cell_create(out, out, nb, t, r));
}

/**
 * Transforms the features in the embedding space to the hidden space
 *
 * Input: x of shape (seq_len, batch_size, in)
 * Output: shape (seq_len * batch_size, out)
*/
tensor& RNN::forward(tensor &x, bool training)
{
    x.forward();
    dim_t seq_len = x.data.dims(0);
    dim_t batch_size = x.data.dims(1);
    dim_t out_size = cells[0]->out_size;
    tensor *y = nullptr;

    for (int i = 0; i < seq_len; i++) {
        tensor* seq = &x.rslice(0, i);
        for (auto cell : cells)
            seq = cell->forward(*seq);
        if (!y)
            y = &seq->reshape({1, batch_size, out_size});
        else
            y = &seq->xstack(*y, 0);
    }
    af::dim4 dims = { seq_len * batch_size, out_size };
    return y->reshape(dims);
}

std::vector<tensor *> RNN::parameters(void)
{
    std::vector<tensor*> ret;
    for (auto c : cells) {
        auto p = c->parameters();
        ret.insert(ret.end(), p.begin(), p.end());
    }
    return ret;
}

layer_stat RNN::stat(void)
{
    layer_stat ret = { 0, 0, 0 };
    for (auto& c : cells) {
        auto s = c->stat();
        ret.num_params += s.num_params;
    }
    ret.in = cells[0]->in_size;
    ret.out = cells[0]->out_size;
    return ret;
}

// rnn uniform value in [-sqrt(1/out), sqrt(1/out)] as suggested by PyTorch
static array rnn_uniform(int in, int out, const af::dtype t)
{
    float r = 1.0 / std::sqrt(out);
    return af::randu(in, out, t) * 2 * r - r;
}

lstm_cell::lstm_cell(int in, int out, bool nb, const af::dtype t) : type(t)
{
    in_size = in; out_size = out; no_bias = nb;
    array wh[4];
    for (int i = 0; i < 4; i++)
        wh[i] = orthogonal(out, out, t);
    array wh4 = af::join(1, wh[0], wh[1], wh[2], wh[3]);

    // I've tried to init ih with orthogonal but it worked worse than xavier_uniform
    // tensorflow suggests xavier_uniform for ih, orthogonal for hh
    weight_ih.init(xavier_uniform(in, 4 * out, t));
    weight_hh.init(wh4);

    // I've tried to set bias_inputgate as ones, but it worked worse than zeros
    if (!no_bias)
        bias.init(zeros(1, 4 * out, t));
}

/**
 * Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that
 * is used to learn long-term dependencies. It is implemented by adding a memory cell and three
 * gates to a vanilla RNN. The memory cell is used to store the long-term memory. The three gates
 * are used to control the flow of information into and out of the memory cell. The three gates
 * are input gate, forget gate and output gate.
 * intput gate: controls the flow of information from the input to the memory cell.
 * forget gate: controls the flow of information from the memory cell to itself.
 * output gate: controls the flow of information from the memory cell to the output.
 *
 * LSTM cell has two states: hidden state and cell state.
 * cell_state: the memory cell of the previous RNN cell, updated as follows:
 *   cell_state = cell_state * forget_gate + input_gate * g
 * hidden_state: the output of the previous RNN cell, updated as:
 *   hidden_state = tanh(cell_state) * output_gate
 * see more details at https://www.bioinf.jku.at/publications/older/2604.pdf
 *
 * Note: {i,f,g,o} are just implemented as slices of one gates tensor.
 *
 * Input:  x of shape (batch_size, in)
 * output: hidden_state of shape (batch_size, out)
 */
tensor* lstm_cell::forward(tensor &x)
{
    if (unlikely(hidden_state.data.isempty())) {
        x.forward();
        int batch_size = x.data.dims(0);
        hidden_state.init(zeros(batch_size, out_size, type));
        cell_state.init(zeros(batch_size, out_size, type));
    }

    tensor &gates = x.matmul(weight_ih) + hidden_state.detach().matmul(weight_hh);
    if (!no_bias)
        gates += bias.expandas(x);

    tensor &input = gates.slice(1, 0, out_size - 1).sigmoid();
    tensor &forget = gates.slice(1, out_size, out_size*2 -1).sigmoid();
    tensor &g = gates.slice(1, 2*out_size, 3*out_size - 1).tanh();
    tensor &output = gates.slice(1, 3*out_size, 4*out_size - 1).sigmoid();

    tensor &new_cell_state = cell_state.detach() * forget + input * g;
    tensor &new_hidden_state = new_cell_state.tanh() * output;
    new_hidden_state.forward();
    hidden_state.data = new_hidden_state.data;
    cell_state.data = new_cell_state.data;
    return &new_hidden_state;
}

elman_cell::elman_cell(int in, int out, bool nb, const af::dtype t) : type(t)
{
    in_size = in; out_size = out; no_bias = nb;
    weight_ih.init(rnn_uniform(in, out, t));
    weight_hh.init(rnn_uniform(out, out, t));
    if (!no_bias) {
        bias.init(rnn_uniform(1, out, t));
    }
}

// Input:  x of shape (batch_size, in)
// output: hidden_state of shape (batch_size, out)
tensor* elman_cell::forward(tensor &x)
{
    if (unlikely(hidden_state.data.isempty())) {
        x.forward();
        int batch_size = x.data.dims(0);
        hidden_state.init(zeros(batch_size, out_size, type));
    }

    tensor &y = x.matmul(weight_ih) + hidden_state.detach().matmul(weight_hh);
    if (!no_bias)
        y += bias.expandas(x);
    tensor &new_hidden_state = y.tanh();
    new_hidden_state.forward();
    hidden_state.data = new_hidden_state.data;
    return &new_hidden_state;
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

        tensor &y = x.normalize1d(0);
        return y * weight.expandas(x) + bias.expandas(x);
    }
    tensor &y = (x - moving_mean.expandas(x)) / (moving_vari.expandas(x) + epsilon).pow(0.5);
    return y * weight.expandas(x) + bias.expandas(x);
}

/**
 * Layer Normalization (LN) is different from Batch Normalization (BN) in that it normalizes
 * features across the features dimension instead of the batch dimension.
 */
tensor& LayerNorm1d::forward(tensor &x, bool training)
{
    tensor &y = x.normalize1d(1);
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

tensor& logits_cross_entroy(tensor &y_true, tensor &y_pred)
{
    return -(y_true * y_pred.logsm()).sum(1);
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

static inline void update_loss_metrics(float loss, float accu, af::timer &e, size_t epoch, bool end)
{
    static std::vector<float> epoch_loss;
    static std::vector<float> epoch_accu;
    epoch_loss.push_back(loss);
    epoch_accu.push_back(accu);

    if (!end)
        return;

    float avg_loss = std::accumulate(epoch_loss.begin(), epoch_loss.end(), 0.0) / epoch_loss.size();
    float avg_accu = std::accumulate(epoch_accu.begin(), epoch_accu.end(), 0.0) / epoch_accu.size();
    printf("| %-5zu | %-9.1f | %-10.8f | %-10.8f |\n", epoch, af::timer::stop(e), avg_loss, avg_accu);
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
        array mean_hat = p.mean / (1 - std::pow(beta1, t));
        array variance_hat = p.variance / (1 - std::pow(beta2, t));
        p.param->data -= lr * mean_hat / (af::sqrt(variance_hat) + epsilon);

        p.param->zero_grad();
    }
}

static void check_layer_dimension(struct std::vector<layer *> layers)
{
    for (size_t i = 0; i < layers.size() - 1; i++) {
        int j = 1;
        if (layers[i]->stat().out == 0)
            continue;
        if (layers[i+j]->stat().in == 0)
                j++; // skip dropout layer for now
        if (layers[i]->stat().out != layers[i+j]->stat().in)
            panic("%s layer[%ld] output dimension %lld does not match %s layer[%ld] input dimension %lld",
                layers[i]->name, i, layers[i]->stat().out, layers[i+j]->name, i+j, layers[i+j]->stat().in);
    }
}

seqnet::seqnet(std::initializer_list<layer *> layers)
{
    for (auto layer : layers)
        add(layer);
    check_layer_dimension(layers);
}

void seqnet::train(data &set, trainer &tr)
{
    progress_bar bar;
    if (tr.seq_len) // time series data need to be reshaped as (*, batch_size)
        set.reshape(tr.batch_size);
    size_t batch_size = tr.seq_len ? tr.seq_len : tr.batch_size;
    set.init_train_idx(batch_size);
    size_t n = set.train_idx.size();
    bar.max = n;

    printf("| Epoch | Time Used | Train Loss | Train Accu |\n");
    for (size_t i = 0; i < tr.epochs; i++) {
        af::timer e = af::timer::start();
        bar.prefix_text = "| " + std::to_string(i) + " ";
        for (std::vector<size_t>::iterator it = set.train_idx.begin(); it != set.train_idx.end(); it++) {
            tensor x_batch, y_true;

            af::timer b = af::timer::start();
            set.get_mini_batch(x_batch, y_true, *it, batch_size);
            tensor &y_pred = forward(x_batch, true);
             if (tr.seq_len)
                 y_true.data = onehot(y_true.data, set.tokenizer.vocab.size());
            tensor &loss = tr.loss_fn(y_true, y_pred);

            loss.backward();
            tr.optimizer.step();

            float batch_loss = af::mean<float>(loss.data);
            float batch_accu = tr.metrics_fn(y_true, y_pred);
            loss.destroy_graph();

            float bt = af::timer::stop(b);
            std::stringstream str;
            if ( bt > 1 )
                str << std::fixed << std::setprecision(2) << " " << bt << " s/it";
            else
                str << std::fixed << std::setprecision(2) << " " << 1/bt << " it/s";
            bar.postfix_text = str.str();
            bar.tick();

            update_loss_metrics(batch_loss, batch_accu, e, i, it == set.train_idx.end() - 1);
        }
        if (set.shuffle)
            set.shuffle_train_idx();
        /**
         * unlike tensorflow, we reset hidden states after each epoch instead of each batch, meaning that
         * our RNN is stateful during the training process. This improves the accuracy of the model
         */
        if (tr.seq_len)
            reset_hidden_states();
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
    for (auto layer : layers) {
        layer_stat st = layer->stat();
        total_params += st.num_params;
        printf("| %-5d | %-7s | %-5lld | %-6lld | %-4s | %-10s | %-'10lld |\n", i++, layer->name,
            st.in, st.out, layer->no_bias ? "None" : "Yes", activ_name[layer->act], st.num_params);
    }
    printf("+-------+---------+-------+--------+------+------------+------------+\n");
    printf("Total params: %ld\n", total_params);
    printf("Running on:\n");
    af::info();
    printf("\n");
}
