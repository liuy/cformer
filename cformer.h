#ifndef CFORMER_H
#define CFORMER_H

#include <arrayfire.h>
#include <unordered_set>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>
#include <bits/stdc++.h>
#include <cmath>

using af::array;

struct tensor;

typedef array (*forward_fn_t)(tensor *, tensor *, tensor *);
typedef void (*backward_fn_t)(tensor *, tensor *, tensor *);

struct oper {
    const char *name = nullptr;          // name of the operator
    forward_fn_t forward_fn = nullptr;   // forward function that computes the data
    backward_fn_t backward_fn = nullptr; // backward function that computes the gradient
    oper(const char *n, forward_fn_t ffn, backward_fn_t bfn)
        : name(n), forward_fn(ffn), backward_fn(bfn) {}
};

struct param {
    int dim;           // parameter of lhs
    int int1;          // first int parameter
    int int2;          // second int parameter
    float float1;      // first float parameter
    af::dim4 dim4;     // parameter of reshape oper
};

struct tensor {
    array data = array();  // evaluated data of the tensor
    array grad = array();  // gradient of the tensor
    float scalar;      // parameter of scalar
    tensor *lhs = nullptr; // left-hand-side of the expression
    tensor *rhs = nullptr; // right-hand-side of the expression
    oper *op = nullptr;    // operator of the expression
    struct param param;    // parameter of the oper
    bool no_delete = true; // whether to delete the tensor by .destroy_graph()
    bool need_grad = false; // whether to compute the gradient of the tensor
    bool data_computed = true; // whether the data of the tensor is computed

#define copy_delete(t) data = t.data; grad = t.grad; scalar = t.scalar; lhs = t.lhs; rhs = t.rhs; param = t.param; \
        op = t.op; need_grad = t.need_grad; data_computed = t.data_computed; if (!t.no_delete) delete &t;

    tensor() = default;
    tensor(const float f) : scalar(f) {} // for float leaf tensor
    tensor(const array &a, bool ng = false) : data(a), need_grad(ng)
        {if (need_grad) grad = af::constant(0, a.dims());} // for leaf tensor
    tensor(tensor &t) {copy_delete(t);} // Note: no_delete = true, for root tensor mostly. USE WITH CAUTION!
    tensor(tensor &&t) = default; // for tensor t = tensor(a)
    tensor(tensor *a, tensor *b, oper *o) // for non-leaf tensor by operators
        : lhs(a), rhs(b), op(o), no_delete(false), need_grad(true), data_computed(false) {}
    tensor(tensor *a, const array &b, oper *o) // create non-leaf node from an array
        : lhs(a), op(o), no_delete(false), need_grad(true), data_computed(false) { rhs = new tensor(b); rhs->no_delete = false; }
    tensor(tensor *a, const float f, oper *o) // create non-leaf node from a float
        : lhs(a), op(o), no_delete(false), need_grad(true), data_computed(false)
        { rhs = new tensor(f); rhs->no_delete = false; }
    void operator=(tensor &t) {copy_delete(t);} // Note: no_delete = true, for root tensor mostly. USE WITH CAUTION!
    tensor& operator=(tensor &&t) = default; // for t = tensor(a)
    //~tensor() { printf("~tensor() %p\n", this); }
#undef copy_delete
    // we overload the operators to construct a computational graph. For e.g,
    // tensor y = a * b + a / c will be constructed as follows:
    //        y(+)        y  = t1 + t2
    //       /   \
    //    t1(*)  t2(/)    t1 = a * b
    //     /\     /\      t2 = a / c
    //    a  b   a  c
    // Then we call forward() on y to evaluate the expressions and backward() to
    // compute the gradients. With .forward() and .backward() we implement the
    // so-called "autograd"(Automatic Differentiation) in ML frameworks like PyTorch
    // and TensorFlow. Temporary tensors t1, t2 are created on the heap and call
    // .destroy_graph() to delete them after use.
    // For e.g, if we call y.backward() on the above graph, we will get by Chain Rule:
    // use da to denote dL/da, and so on...
    // y.grad(dy) = 1, t1.grad(dt1) = 1, t2.grad(dt2) = 1,
    // a.grad(da) = b + 1/c, b.grad(db) = a, c.grad(dc) = -a/c^2
    //
    // we recommend root and leaf tensors are allocated on the stack and tensors returned
    // by operators are allocated on the heap.

    void forward(void);
    void backward(void);
    void backward(const array &g);
    void destroy_graph(void);
    void print_graph(void);
    inline void zero_grad(void) {grad = 0;}
    inline void init(const array &a){data = a; if (need_grad) grad = af::constant(0, a.dims());}
    inline bool is_leaf(void) {return lhs == nullptr && rhs == nullptr;}
    tensor& matmul(tensor &t);
    tensor& log(void);
    tensor& exp(void);
    tensor& relu(void);
    tensor& sigmoid(void);
    tensor& tanh(void);
    tensor& softmax(void);
    /**
     * Gaussian Error Linear Unit is used for LM and transformer models like GPT-2 and BERT,
     * this functions prevents the problem of vanishing gradients also unlike ELU, it has a
     * continuous derivative at 0, which can sometimes make training faster.
     *
     * Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
     */
    tensor& gelu(void);
    /**
     * Sigmoid Linear Units, it serves as a smooth approximation to the ReLU function.
     * It is used mostly in computer vision tasks.
     *
     * Ref: https://arxiv.org/abs/1702.03118
     */
    tensor& silu(void);
    /**
     * @brief Sum all values along dimension dim and broadcast to the original shape.
     * @param dim The dimension along which the add operation occurs.
     */
    tensor& bsum(int);
    tensor& sum(int);
    tensor& expandas(tensor &t);
    tensor& bmax(int);
    /// @brief Oper fusion: LogSumExp along the dimension 1
    tensor& lse(void);
    tensor& logsm(void);
    tensor& submean(int);
    tensor& bstd(int);
    /**
     * @brief Normalize(subtract mean and divide by standard deviation) tensor along the dimension dim.
     * @param dim The dimension along which the normalization is performed.
     * @param eps Epsilon to avoid dividing by zero.
     */
    tensor& normalize1d(int dim, float eps = 1e-5);
    tensor& pow(float);
    /// @brief Slice the tensor along the dimension dim. For e.g, dim=1, T[span, begin:end]
    /// @return the slice of tensor
    tensor& slice(int dim, int begin, int end);
    /**
     * detach tensor to a temporary tensor.
     *
     * y = y + x; will cause Directed Cyclic Graph problem.
     * y = y.detach() + x; solve the problem.
     */
    tensor& detach(void);
    tensor& reshape(const af::dim4 &d);
    /**
     * For an array a with two dimensions, T() gives the matrix transpose.
     * For an array with more than two dimensions, the first two dimensions are transposed.
     */
    tensor& T(void);
    /// @brief Concatenate tensors along the dimension dim.
    /// For e.g, t1 = [1, 2], t2 = [2, 2], t2.stack(t1, 0) = [3, 2]
    /// @param dim along which dimension to concatenate
    /// @param tensor the tensor to concatenate
    /// @return the reference of the concatenated tensor
    tensor& stack(tensor &t, int dim = 0);
    /// @brief slice and reduce the tensor along the dimension dim.
    /// For e.g, t1 = [2, 2, 3], t1.rslice(0, 0) = [2, 3]
    /// @param dim along which dimension to rslice
    /// @param n the nth slice
    /// @return the reference of the reduced tensor
    tensor& rslice(int dim, int n);
    /// @brief expand and concatenate the tensor along the dimension dim.
    /// For e.g, t1 = [2, 3], t2[1, 2, 3], t1.xstack(t2, 0) = [2, 2, 3]
    /// @param t the tensor to expand and concatenate
    /// @param dim along which dimension to expand and concatenate
    /// @return the reference of the expanded and concatenated tensor
    tensor& xstack(tensor &t, int dim = 0);
    tensor& operator+(tensor &t);
    tensor& operator-(tensor &t);
    tensor& operator*(tensor &t);
    tensor& operator/(tensor &t);
    tensor& operator+(const array &a);
    tensor& operator-(const array &a);
    tensor& operator*(const array &a);
    tensor& operator/(const array &a);
    tensor& operator+(float f);
    tensor& operator-(float f);
    tensor& operator*(float f);
    tensor& operator/(float f);
    void operator+=(tensor &t);
    tensor& operator-(void);
};

// kaiming_uniform is randu value [-limit, limit] and mostly for ReLU activation
static inline array kaiming_uniform(int in, int out, const af::dtype t = f32)
{
    double limit = std::sqrt(6.0 / in);
    return af::randu(in, out, t) * 2 * limit - limit;
}

// xavier_uniform is randu value [-limit, limit] and mostly for tanh and sigmoid
static inline array xavier_uniform(int in, int out, const af::dtype t = f32)
{
    double limit = std::sqrt(6.0 / (in + out));
    return af::randu(in, out, t) * 2 * limit - limit;
}

// kaiming_normal is randn value with mean 0 and std sqrt(2.0 / in)
static inline array kaiming_normal(int in, int out, const af::dtype ty = f32)
{
   return af::randn({in, out, 1, 1}, ty) * std::sqrt(2.0 / in);
}

// xavier_normal is randn value with mean 0 and std sqrt(2.0 / (in + out))
static inline array xavier_normal(int in, int out, const af::dtype ty = f32)
{
    return af::randn({in, out, 1, 1}, ty) * std::sqrt(2.0 / (in + out));
}

/**
 * Orthogonal initializer. (M @ M.T = I)
 *
 * If the matrix has fewer rows than columns then the output will have orthogonal rows.
 * Otherwise, the output will have orthogonal columns.
 */
static inline array orthogonal(int in, int out, const af::dtype ty = f32)
{
    array a;
    // hack to make sure q is square with max(in, out) dimension
    if (in >= out)
        a = af::randn(in, out, ty);
    else
        a = af::randn(out, in, ty);

    array q, r, tau;
    // qr returns q as shape [dim0, dim0], is this a bug?
    af::qr(q, r, tau, a);
    return q(af::seq(0, in - 1), af::seq(0, out - 1));
}

static inline array zeros(int in, int out, const af::dtype ty = f32)
{
    return af::constant(0, {in, out}, ty);
}

static inline array ones(int in, int out, const af::dtype ty = f32)
{
    return af::constant(1, {in, out}, ty);
}

typedef void (*data_reader_t)(struct data &);

struct transform {
    virtual array operator()(const array &a, struct data &d) = 0;
    virtual ~transform() = default;
};

// random rotate image by [-degree, degree] degrees
struct random_rotate : transform {
    float degree;
    float ratio;
    random_rotate(float d, float r=1.0f) : degree(d), ratio(r) {}
    array operator()(const array &a, struct data &d) override;
};

/**
 * Randomly transforms the morphology of objects in images and produces a see-through-water-like effect.
 * @alpha – Magnitude of displacements.
 * @sigma – Smoothness of displacements.
 */
struct elastic_transform : transform {
    float alpha;
    float sigma;
    float ratio;
    elastic_transform(float a = 5.0, float s = 4.0, float r=0.5) : alpha(a), sigma(s), ratio(r) {}
    array operator()(const array &a, struct data &d) override;
};

struct tokenizer {
    std::unordered_map<std::string, uint32_t> token2idx;
    std::unordered_map<uint32_t, std::string> idx2token;
    std::vector<std::string> vocab;
    tokenizer() = default;
    // encode a text to a vector of word indices
    std::vector<uint32_t> encode_word(const std::string &s);
    // encode a text to a vector of char indices
    std::vector<uint32_t> encode_char(const std::string &s);
    // decode a vector of word indices to a text
    std::string decode(const std::vector<uint32_t> &v);
private:
    std::vector<uint32_t> encode(const std::vector<std::string>& tokens);

};

struct data {
    std::vector<size_t> train_idx;
    tensor train_x, train_y, test_x, test_y;
    size_t nrow, ncol; // for images
    bool shuffle;
    data_reader_t data_reader;
    struct tokenizer tokenizer;
    data(data_reader_t dr, bool shf = true) : data_reader(dr), shuffle(shf) {}
    inline size_t num_examples(void) { return train_x.data.dims(0); }
    void load(std::initializer_list<transform*> list = {});
    void init_train_idx(size_t batch_size);
    void get_mini_batch(tensor &x, tensor &y, size_t idx, size_t batch_size);
    inline void shuffle_train_idx(void) {
        std::shuffle(train_idx.begin(), train_idx.end(), std::default_random_engine());}
    void reshape(size_t batch_size);
};

typedef array (*initializer_t)(int, int, const af::dtype);

static const char *activ_name[] = {"None", "ReLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax", "GELU"};
enum activ_t {None, ReLU, Sigmoid, Tanh, Softmax, LogSoftmax, GELU};

struct layer_stat {
    dim_t in;
    dim_t out;
    dim_t num_params;
};

struct layer {
    const char *name;
    bool no_bias = false;
    activ_t act = None;
    virtual ~layer() = default;
    virtual tensor& forward(tensor &x, bool = false) = 0;
    virtual std::vector<tensor *> parameters(void) = 0;
    virtual layer_stat stat(void) = 0;
    virtual void reset_hidden_states(void) {}
    inline tensor& operator()(tensor &x) { return forward(x); } // make layer as functor for convention
};

struct Linear : layer {
    initializer_t init;
    tensor weight = tensor(array(), true);
    tensor bias = tensor(array(), true);
    Linear(int in, int out, activ_t a = None, bool nb = false, const af::dtype t = f32)
        : init(a == ReLU ? kaiming_uniform : xavier_uniform)
    /** Notes on bias initialization:
     * Generally, there are 4 recommended ways to initialize the bias:
     * 1. use weight initializer (default for this layer)
     * 2. constant 0
     * 3. constant 0.01
     * 4. just any small random value
     * You can actually set layer.bias directly if you want to override the default.
     */
    {name = "Linear"; act = a; no_bias = nb; weight.init(init(in, out, t));
    if (!no_bias) bias.init(af::transpose(init(out, 1, t)));}
    tensor& forward(tensor &x, bool training = false) override;
    std::vector<tensor *> parameters(void) override
    { if (no_bias) return {&weight}; else return {&weight, &bias}; }
    layer_stat stat(void) override
    { return {weight.data.dims(0), weight.data.dims(1),
      no_bias ? weight.data.elements() : weight.data.elements() + bias.data.elements()}; }
};

struct BatchNorm1d : layer {
    float momentum;
    float epsilon;
    tensor moving_mean;
    tensor moving_vari;
    tensor weight = tensor(array(), true);
    tensor bias = tensor(array(), true);
    BatchNorm1d(int dim, float m = 0.9, float e = 1e-5, const af::dtype t = f32)
        : momentum(m), epsilon(e)
        {name = "BN1d"; weight.init(ones(1, dim, t)); bias.init(zeros(1, dim, t));
         moving_mean.init(zeros(1, dim, t)); moving_vari.init(ones(1, dim, t));}
    tensor& forward(tensor &x, bool training = false) override;
    std::vector<tensor *> parameters(void) override { return {&weight, &bias}; }
    layer_stat stat(void) override
    { return {weight.data.dims(1), weight.data.dims(1), weight.data.elements() + bias.data.elements()}; }
};

struct LayerNorm1d : layer {
    float epsilon;
    tensor weight = tensor(array(), true);
    tensor bias = tensor(array(), true);
    LayerNorm1d(int dim, float e = 1e-5, const af::dtype t = f32)
        : epsilon(e)
        {name = "LN1d"; weight.init(ones(1, dim, t)); bias.init(zeros(1, dim, t));}
    tensor& forward(tensor &x, bool training = false) override;
    std::vector<tensor *> parameters(void) override { return {&weight, &bias}; }
    layer_stat stat(void) override
    { return {weight.data.dims(1), weight.data.dims(1), weight.data.elements() + bias.data.elements()}; }
};

struct Dropout : layer {
    float p;
    Dropout(float prob = 0.2) : p(prob) {name = "Dropout"; no_bias = true;}
    tensor& forward(tensor &x, bool training = false) override;
    std::vector<tensor *> parameters(void) override { return {}; }
    layer_stat stat(void) override { return {0, 0, 0}; }
};

struct Embedding : layer {
    tensor weight = tensor(array(), true);
    Embedding(int in, int out, const af::dtype t = f32)
    {name = "Embed"; no_bias = true; weight.init(xavier_normal(in, out, t));}
    tensor& forward(tensor &x, bool training = false) override;
    std::vector<tensor *> parameters(void) override { return {&weight}; }
    layer_stat stat(void) override { return {weight.data.dims(0), weight.data.dims(1), weight.data.elements()}; }
};

struct rnn_cell {
    bool no_bias;
    int in_size, out_size;
    tensor weight_ih = tensor(array(), true), weight_hh = tensor(array(), true);
    tensor bias = tensor(array(), true);
    virtual tensor* forward(tensor &x) = 0;
    virtual void reset_hidden_states(void) = 0;
    std::vector <tensor *> parameters(void) {
        if (no_bias) return {&weight_ih, &weight_hh};
        else return {&weight_ih, &weight_hh, &bias};
    }
    layer_stat stat(void) {
        return {in_size, out_size, no_bias ? weight_ih.data.elements() + weight_hh.data.elements() :
                weight_ih.data.elements() + weight_hh.data.elements() +
                bias.data.elements()};
    }
};

struct lstm_cell : rnn_cell {
    af::dtype type;
    tensor hidden_state, cell_state;
    lstm_cell(int in, int out, bool nb = false, const af::dtype t = f32);
    tensor* forward(tensor &x) override;
    void reset_hidden_states(void) override {hidden_state.data = array(); cell_state.data = array();}
};

struct elman_cell : rnn_cell {
    af::dtype type;
    tensor hidden_state;
    elman_cell(int in, int out, bool nb = false, const af::dtype t = f32);
    tensor* forward(tensor &x) override;
    void reset_hidden_states(void) override {hidden_state.data = array();}
};

static const char *rnn_name[] = {"RNN", "LSTM", "GRU"};
enum rnn_t {Simple, LSTM, GRU};

struct RNN : layer {
    std::vector<rnn_cell *> cells;
    RNN(int in, int out, int num_layers = 1, rnn_t r = Simple, bool nb = false, const af::dtype t = f32);
    void reset_hidden_states(void) override {for (auto c : cells) c->reset_hidden_states();}
    tensor& forward(tensor &x, bool training) override;
    std::vector<tensor *> parameters(void) override;
    layer_stat stat(void) override;
    ~RNN(void) {for (auto c : cells) delete c;}
};

struct multihead_attention {
    int num_heads;
    int embed_dim;
    float dropout;
    bool no_bias;
    tensor weight_qkv = tensor(array(), true);
    tensor bias_qkv = tensor(array(), true);
    tensor weight_o = tensor(array(), true);
    tensor bias_o = tensor(array(), true);
    multihead_attention(int dim, int nheads = 8, bool nb = false, float dp = 0.0, const af::dtype t = f32) {
        assert(dim % nheads == 0);
        num_heads = nheads; embed_dim = dim; dropout = dp; no_bias = nb;
        weight_qkv.init(xavier_normal(dim, dim * 3, t));
        weight_o.init(xavier_normal(dim, dim, t));
        if (!no_bias) {
            bias_qkv.init(zeros(1, dim * 3, t));
            bias_o.init(zeros(1, dim, t));
        }
    }
    tensor& forward(tensor &x);
    std::vector<tensor *> parameters(void) {
        return {&weight_qkv, &weight_o, &bias_o};
    }
    layer_stat stat(void) {
        return {embed_dim, embed_dim, weight_qkv.data.elements() + weight_o.data.elements()
                + bias_o.data.elements()};
    }
};

struct optimizer {
    virtual void step(void) = 0;
};

struct sgd_param{
    tensor *param;
    array velocity;
};

struct SGD : optimizer {
    std::vector<sgd_param> params;
    float lr;
    float momentum;
    float weight_decay;
    bool nesterov;
    SGD(std::vector<tensor*> &p, float l = 5e-4, float m = 0.8, bool n=false, float wd = 0.0)
        : lr(l), momentum(m), nesterov(n), weight_decay(wd)
        {for (auto t : p) params.push_back({t, m ? af::constant(0, t->grad.dims()) : array()});}
    void step(void) override;
};

struct adam_param {
    tensor *param;
    array mean;
    array variance;
};

struct Adam : optimizer {
    std::vector<adam_param> params;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    Adam(std::vector<tensor*> &p, float l = 1e-3, float wd =0.0, float b1 = 0.9, float b2 = 0.999, float e = 1e-8)
        : lr(l), weight_decay(wd), beta1(b1), beta2(b2), epsilon(e)
        { for (auto t : p) params.push_back({t, af::constant(0, t->grad.dims()), af::constant(0, t->grad.dims())}); }
    void step(void) override;
};

typedef tensor& (*loss_fn_t)(tensor &y_true, tensor &y_pred);
typedef float (*metrics_fn_t)(tensor &y_true, tensor &y_pred);
struct trainer {
    size_t epochs;
    size_t batch_size;
    size_t seq_len;
    struct optimizer &optimizer;
    loss_fn_t loss_fn;
    metrics_fn_t metrics_fn;
};

struct seqnet {
    const char *name = "Sequential Network";
    std::vector<tensor*> params;
    std::vector<layer*> layers;
    seqnet(std::initializer_list<layer*> list);
    ~seqnet() { for (auto i : layers) delete i; }
    inline void add(layer *l)
    {layers.push_back(l); for (auto p : l->parameters()) params.push_back(p);}
    void train(data &set, trainer &tr);
    tensor& forward(tensor &x, bool training = false);
    inline tensor operator()(tensor &x) {
        tensor &t = forward(x); t.forward(); tensor r; r.data = t.data; t.destroy_graph(); return r; }
    void summary(void);
    void reset_hidden_states(void) {for (auto l : layers) l->reset_hidden_states();}
};

#ifdef CF_DEBUG
template <typename... Args>
static inline void _af_debug(Args... args)
{
    for (auto arg : {args...})
        af_print(arg);
}
#define af_debug(...) do { \
    fprintf(stdout, "%s:%d:%s():\n", __FILE__, __LINE__, __func__); \
    _af_debug(__VA_ARGS__); \
    } while (0)
#define cf_debug(fmt, ...) do { \
    fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    } while (0)
#define cf_assert(cond, fmt, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
        exit(EXIT_FAILURE); \
    } \
    } while (0)
#else
    #define af_debug(...) do {} while (0)
    #define cf_debug(...) do {} while (0)
    #define cf_assert(...) do {} while (0)
#endif

#define panic(fmt, ...) do { \
    fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE); \
    } while (0)

#define array_shape(a) do { \
    printf("%s shape: [", #a); \
    for (int i = 0; i < a.numdims(); i++) \
        printf("%lld,", a.dims(i)); \
    printf("]\n"); \
    } while (0)

static const char *array_typename[] = {"f32", "c32", "f64", "c64", "b8", "s32", "u32", "u8", "s64", "u64", "s16", "u16", "f16"};

// ********************** helper functions **********************
static inline std::string read_file(const std::string& file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
        panic("Cannot open file: %s", file_path.c_str());

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

static inline int32_t read_i32(std::ifstream &file)
{
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

static inline void read_data(std::ifstream &file, void *data, size_t size)
{
    file.read(reinterpret_cast<char*>(data), size);
}

static inline void write_data(std::ofstream &file, void *data, size_t size)
{
    file.write(reinterpret_cast<char*>(data), size);
}

static inline array zeros_like(array &a)
{
    return af::constant(0, a.dims(), a.type());
}

static inline array ones_like(array &a)
{
    return af::constant(1, a.dims(), a.type());
}

static inline array argmax(const array &a, int dim = 1)
{
    array max_vals, max_idxs;
    af::max(max_vals, max_idxs, a, dim);
    return max_idxs;
}

/**
 * a[n] -> a[n, num_classes], where a[n, i] = 1 if a[n] == i else 0.
 * a[n, m] -> a[n, m, num_classes], where a[n, m, i] = 1 if a[n, m] == i else 0. for e.g
 * a[2, 3]:
 *  [[1, 2, 0]
 *   [2, 1, 1]]
 *
 * a[2, 3, 3] = onehot(a[2, 3], 3):
 * [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]
 *  [[0, 0, 1], [0, 1, 0], [0, 1, 0]]]
 *
 * if keep_dims is false, then
 * a[n, m] -> a[n * m, num_classes]
 */
static inline array onehot(const array &a, int num_classes = 10, bool keep_dims = false)
{
    array iden = af::identity(num_classes, num_classes, a.type());

    if (keep_dims) {
        af::dim4 dims = a.dims();
        dims[a.dims().ndims()] = num_classes;
        return af::moddims(iden(a, af::span), dims);
    }
    return iden(a, af::span);
}

static inline array bmax(const array &a)
{
    return af::tile(af::max(a, 1), 1, a.dims(1));
}

static inline array bsum(const array &a, int dim)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[dim] = a.dims(dim);

    return af::tile(af::sum(a, dim), dims);
}

/* non-stable for performace */
static inline array softmax(const array &a)
{
    array e = af::exp(a);
    return e / bsum(e, 1);
}

static inline array stable_softmax(const array &a)
{
    array e = af::exp(a - bmax(a));
    return e / bsum(e, 1);
}

// endian conversion helpers
#define bswap_16(x) __builtin_bswap16(x)
#define bswap_32(x) __builtin_bswap32(x)
#define bswap_64(x) __builtin_bswap64(x)

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

static inline int random(int from, int to)
{
    static bool start = true;
    if (unlikely(start)) {
        std::srand(std::time(nullptr));
        start = false;
    }
    return std::rand()%(to-from + 1) + from;
}

#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define tensor_has_inf_nan(a) (af::anyTrue<bool>(af::isNaN(a)) || af::anyTrue<bool>(af::isInf(a)))

// static inline int random(int from, int to)
// {
//     static std::default_random_engine e(std::time(nullptr));
//     static std::uniform_int_distribution<> dis(from, to);
//     return dis(e);
// }

/**
 * \param logits: Data returned by model without activation.
 * \param top_k:  Sample from only top k elements with highest probability.
 * \param top_p:  Sample from only top p elements with cumulative probability.
 * \return Sampled idx from filtered logits by top_k or top_p. If both top_k and top_p
 *         are set, then sample from min(top_k, top_p) elements.
 */
static inline uint32_t logits_sample_next(array &logits, int top_k = 5, float top_p = 0.0)
{
    cf_assert(logits.dims(0) == 1, "only support row vector.");

    // fast path for top most likely element
    if (top_k == 1 && top_p == 0)
        return argmax(logits).as(u32).scalar<uint32_t>();

    array val, idx;
    af::sort(val, idx, logits, 1 /* dim */, false);
    array p = softmax(val);
    array cp = af::accum(p, 1 /* dim */);
    uint32_t predict;

    int p_nr = 0;
    if (top_p > 0) {
        p_nr = af::count<int>(cp < top_p);
        p_nr = p_nr > 0 ? p_nr : 1; // at least one element
    }

    int filter_nr;
    if (top_k > 0 && p_nr > 0)
        filter_nr = std::min(top_k, p_nr);
    else if (top_k > 0 && p_nr == 0)
        filter_nr = top_k;
    else if (p_nr > 0 && top_k == 0)
        filter_nr = p_nr;
    else
        panic("Both top_k and top_p are 0.");

    if (filter_nr == 1)
        return idx(0).scalar<uint32_t>();

    // Sample from multinomial distribution of filtered logits.
    array filtered_p = p(af::seq(0, filter_nr - 1));
    array norm = filtered_p / bsum(filtered_p, 1);
    array cnorm = af::accum(norm, 1 /* dim */);
    float r = random(0, RAND_MAX) / (float)RAND_MAX;

    for (int i = 0; i < filter_nr; i++) {
        if (r < cnorm(i).scalar<float>()) {
            predict = idx(i).scalar<uint32_t>();
            break;
        }
    }

    return predict;
}

// *********************** data/ functions ***********************
void mnist_reader(struct data &);
void write_mnist_images(const array &x, const std::string& path);

// ************************ nn functions ************************

tensor& categorical_cross_entropy(tensor &y_true, tensor &y_pred);
tensor& log_softmax_cross_entropy(tensor &y_true, tensor &y_pred);
tensor& logits_cross_entroy(tensor &y_true, tensor &y_pred);
float categorical_accuracy(tensor &y_true, tensor &y_pred);

#endif /* CFORMER_H */
