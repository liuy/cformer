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

using af::array;

struct tensor;

typedef array (*forward_fn_t)(tensor *, tensor *);
typedef void (*backward_fn_t)(tensor *, tensor *, array &);

struct oper {
    const char *name = nullptr;          // name of the operator
    forward_fn_t forward_fn = nullptr;   // forward function that computes the data
    backward_fn_t backward_fn = nullptr; // backward function that computes the gradient
    oper(const char *n, forward_fn_t ffn, backward_fn_t bfn)
        : name(n), forward_fn(ffn), backward_fn(bfn) {}
};

struct tensor {
    array data = array();  // evaluated data of the tensor
    array grad = array();  // gradient of the tensor
    array velocity = array(); // velocity for SGD with momentum
    array mean = array();  // first moment for Adam
    array variance = array(); // second moment for Adam
    tensor *lhs = nullptr; // left-hand-side of the expression
    int dim = 0;           // parameter of lhs
    tensor *rhs = nullptr; // right-hand-side of the expression
    oper *op = nullptr;    // operator of the expression
    bool no_delete = true; // whether to delete the tensor by .destroy_graph()
    bool need_grad = false; // whether to compute the gradient of the tensor
    bool data_computed = false; // whether the data of non-leaf node is computed

#define copy_delete(t) data = t.data; grad = t.grad; velocity  = t.velocity ; lhs = t.lhs; rhs = t.rhs; \
        dim = t.dim; op = t.op; need_grad = t.need_grad; data_computed = t.data_computed; if (!t.no_delete) delete &t;

    tensor() = default;
    tensor(const array &a, bool ng = false) : data(a), need_grad(ng)
        {if (need_grad) grad = af::constant(0, a.dims());} // for leaf tensor
    tensor(const tensor &t) {copy_delete(t);} // for root tensor mostly. USE WITH CAUTION!
    tensor(tensor *a, tensor *b, oper *o) // for non-leaf tensor by operators
        : lhs(a), rhs(b), op(o), no_delete(false), need_grad(true) {}
    void operator=(tensor &t) {copy_delete(t);} // for root tensor mostly. USE WITH CAUTION!
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
    inline void assign_data(const array &a){data = a; if (need_grad) grad = af::constant(0, a.dims());}
    inline bool is_leaf(void) {return lhs == nullptr && rhs == nullptr;}
    tensor& matmul(tensor &t);
    tensor& log(void);
    tensor& exp(void);
    tensor& relu(void);
    tensor& sigmoid(void);
    tensor& tanh(void);
    tensor& softmax(void);
    /// @brief Sum all values along dimension dim and broadcast to the original shape.
    /// @param dim The dimension along which the add operation occurs.
    tensor& bsum(int);
    tensor& sum(int);
    tensor& bdim0(tensor &t);
    tensor& bmax(int);
    /// @brief Oper fusion: LogSumExp along the dimension 1
    tensor& lse(void);
    tensor& logsm(void);
    tensor& submean(void);
    tensor& bstd(void);
    tensor& operator+(tensor &t);
    tensor& operator-(tensor &t);
    tensor& operator*(tensor &t);
    tensor& operator/(tensor &t);
    void operator+=(tensor &t);
    tensor& operator-(void);
};

// kaiming_uniform is randu value [-limit, limit] and mostly for ReLU activation
static inline array kaiming_uniform(int in, int out, const af::dtype t = f32)
{
    double limit = sqrt(6.0 / in);
    return af::randu(in, out, t) * 2 * limit - limit;
}

// xavier_uniform is randu value [-limit, limit] and mostly for tanh and sigmoid
static inline array xavier_uniform(int in, int out, const af::dtype t = f32)
{
    double limit = sqrt(6.0 / (in + out));
    return af::randu(in, out, t) * 2 * limit - limit;
}

// kaiming_normal is randn value with mean 0 and std sqrt(2.0 / in) and mostly for ReLU
static inline array kaiming_normal(int in, int out, const af::dtype ty = f32)
{
   return af::randn({in, out, 1, 1}, ty) * sqrt(2.0 / in);
}

// xavier_normal is randn value with mean 0 and std sqrt(2.0 / (in + out)) and mostly for tanh and sigmoid
static inline array xavier_normal(int in, int out, const af::dtype ty = f32)
{
    return af::randn({in, out, 1, 1}, ty) * sqrt(2.0 / (in + out));
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

struct data {
    std::vector<size_t> train_idx;
    tensor train_x, train_y, test_x, test_y;
    size_t nrow, ncol; // for images
    bool shuffle;
    data_reader_t data_reader;
    data(data_reader_t dr, bool shf = true) : data_reader(dr), shuffle(shf) {}
    inline size_t num_examples(void) { return train_x.data.dims(0); }
    void load(std::initializer_list<transform*> list = {});
    void init_train_idx(size_t batch_size);
    void get_mini_batch(tensor &x, tensor &y, size_t idx, size_t batch_size);
    inline void shuffle_train_idx(void) {
        std::shuffle(train_idx.begin(), train_idx.end(), std::default_random_engine());}
};

typedef array (*initializer_t)(int, int, const af::dtype);

static const char *activ_name[] = {"None", "ReLU", "Sigmoid", "Tanh", "Softmax", "LogSoftmax"};
enum activ_t {None, ReLU, Sigmoid, Tanh, Softmax, LogSoftmax};

struct layer {
    const char *name;
    bool no_bias;
    tensor weight = tensor(array(), true);
    tensor bias = tensor (array(), true);
    activ_t act;
    virtual ~layer() = default;
    virtual tensor& forward(tensor &x, bool = false) = 0;
    inline tensor& operator()(tensor &x) { return forward(x); } // make layer as functor for convention
};

struct linear : layer {
    initializer_t init;
    linear(int in, int out, activ_t a = None, bool nb = false, const af::dtype t = f32)
        : init(a == ReLU ? kaiming_uniform : xavier_uniform)
    /** Notes on bias initialization:
     * Generally, there are 4 recommended ways to initialize the bias:
     * 1. use weight initializer (default for this layer)
     * 2. constant 0
     * 3. constant 0.01
     * 4. just any small random value
     * You can actually set layer.bias directly if you want to override the default.
     */
    {name = "Linear"; act = a; no_bias = nb; weight.assign_data(init(in, out, t));
    if (!no_bias) bias.assign_data(af::transpose(init(out, 1, t)));}
    tensor& forward(tensor &x, bool training = false) override;
};

struct optimizer {
    std::vector<tensor*> params;
    optimizer(std::vector<tensor*> p)
        : params(p) {}
    virtual void step(void) = 0;
    virtual void finish(void) = 0;
};

struct SGD : optimizer {
    float lr;
    float momentum;
    float weight_decay;
    bool nesterov;
    SGD(std::vector<tensor*> &p, float l = 5e-4, float m = 0.8, bool n=false, float wd = 0.0)
        : optimizer(p), lr(l), momentum(m), nesterov(n), weight_decay(wd) {}
    void step(void) override;
    void finish(void) override {for (auto &p : params) {p->velocity = array();}}
};

struct Adam : optimizer {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    Adam(std::vector<tensor*> &p, float l = 1e-4, float wd =0.0, float b1 = 0.9, float b2 = 0.999, float e = 1e-8)
        : optimizer(p), lr(l), weight_decay(wd), beta1(b1), beta2(b2), epsilon(e) {}
    void step(void) override;
    void finish(void) override {for (auto &p : params) {p->mean = array(); p->variance = array();}}
};

typedef tensor& (*loss_fn_t)(tensor &y_true, tensor &y_pred);
typedef float (*metrics_fn_t)(tensor &y_true, tensor &y_pred);
struct trainer {
    size_t epochs;
    size_t batch_size;
    struct optimizer &optimizer;
    loss_fn_t loss_fn;
    metrics_fn_t metrics_fn;
};

struct seqnet {
    const char *name = "Sequential Network";
    std::vector<tensor*> params;
    std::vector<layer*> layers;
    seqnet(std::initializer_list<layer*> list) { for (auto i : list) add(i); }
    ~seqnet() { for (auto i : layers) delete i; }
    inline void add(layer *l)
    { layers.push_back(l); params.push_back(&l->weight); if (!l->no_bias) params.push_back(&l->bias); }
    void train(data &set, trainer &tr);
    tensor& forward(tensor &x, bool training = false);
    inline tensor operator()(tensor &x) {
        tensor &t = forward(x); t.forward(); tensor r; r.data = t.data; t.destroy_graph(); return r; }
    void summary(void);
};

// ********************** helper functions **********************
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


static inline array argmax(const array &a, int dim = 1)
{
    array max_vals, max_idxs;
    af::max(max_vals, max_idxs, a, dim);
    return max_idxs;
}

static inline array onehot(const array &a, int num_classes = 10)
{
    assert(a.numdims() == 1);
    array iden = af::identity(num_classes, num_classes, a.type());
    return af::lookup(iden, a);
}

// endian conversion helpers
#define bswap_16(x) __builtin_bswap16(x)
#define bswap_32(x) __builtin_bswap32(x)
#define bswap_64(x) __builtin_bswap64(x)

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
#else
    #define af_debug(...) do {} while (0)
    #define cf_debug(...) do {} while (0)
#endif

#define panic(fmt, ...) do { \
    fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE); \
    } while (0)

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

// *********************** data/ functions ***********************
void mnist_reader(struct data &);
void write_mnist_images(const array &x, const std::string& path);

// ************************ nn functions ************************

tensor& categorical_cross_entropy(tensor &y_true, tensor &y_pred);
tensor& log_softmax_cross_entropy(tensor &y_true, tensor &y_pred);
float categorical_accuracy(tensor &y_true, tensor &y_pred);

#endif /* CFORMER_H */
