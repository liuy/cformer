#ifndef CFORMER_H
#define CFORMER_H

#include <arrayfire.h>
#include <unordered_set>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>

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
    tensor *lhs = nullptr; // left-hand-side of the expression
    int dim = 0;           // parameter of lhs
    tensor *rhs = nullptr; // right-hand-side of the expression
    oper *op;              // operator of the expression
    bool no_delete = true; // whether to delete the tensor by .destroy_graph()

#define copy_delete(t) data = t.data; grad = t.grad; lhs = t.lhs; rhs = t.rhs; \
        dim = t.dim; op = t.op; if (!t.no_delete) delete &t;

    tensor() = default;
    tensor(const array &a) : data(a), grad(af::constant(0, a.dims())) {} // for leaf tensor
    tensor(const tensor &t) {copy_delete(t);} // for root tensor mostly. USE WITH CAUTION!
    tensor(tensor *a, tensor *b, oper *o) // for non-leaf tensor by operators
        : lhs(a), rhs(b), op(o), no_delete(false) {}
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
    void destroy_graph(void);
    void print_graph(void);
    inline void zero_grad(void) {grad = af::constant(0, data.dims());}
    inline void assign_data(const array &a) {data = a; zero_grad();}
    inline bool is_leaf(void) {return lhs == nullptr && rhs == nullptr;}
    tensor& matmul(tensor &t);
    tensor& log(void);
    tensor& exp(void);
    tensor& relu(void);
    tensor& sigmoid(void);
    tensor& tanh(void);
    /// @brief Sum all values along dimension dim and broadcast to the original shape.
    /// @param dim The dimension along which the add operation occurs.
    tensor& bsum(int);
    tensor& sum(int);
    tensor& bdim0(tensor &t);
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
static inline array xavier_normal(int in, int out, uint64_t seed = 0, const af::dtype ty = f32)
{
    return af::randn({in, out, 1, 1}, ty) * sqrt(2.0 / (in + out));
}

typedef array (*initializer_t)(int, int, const af::dtype);

struct layer {
    bool no_bias;
    tensor weight, bias;
    virtual ~layer() = default;
    virtual tensor& forward(tensor &x) = 0;
    inline tensor& operator()(tensor &x) { return forward(x); } // make layer as functor for convention
};

enum activ_t {None, ReLU, Sigmoid, Tanh, Softmax};

struct linear : layer {
    activ_t act;
    initializer_t init;

    linear(int in, int out, activ_t a = None, bool nb = false, const af::dtype t = f32)
        : act(a), init(a == ReLU ? kaiming_uniform : xavier_uniform)
    /** Notes on bias initialization:
     * Generally, there are 4 recommended ways to initialize the bias:
     * 1. use weight initializer (default for this layer)
     * 2. constant 0
     * 3. constant 0.01
     * 4. just any small random value
     * You can actually set layer.bias directly if you want to override the default.
     */
    {no_bias = nb; weight.assign_data(init(in, out, t));
    if (!no_bias) bias.assign_data(af::transpose(init(out, 1, t)));}
    //af_print(weight.data); af_print(bias.data);}
    tensor& forward(tensor &x) override;
};

struct seq_net {
    std::vector<tensor*> params;
    std::vector<layer*> layers;
    seq_net(std::initializer_list<layer*> list) { for (auto i : list) add(i); }
    ~seq_net() { for (auto i : layers) delete i; }
    inline void add(layer *l)
    { layers.push_back(l); params.push_back(&l->weight); if (!l->no_bias) params.push_back(&l->bias); }
    void train(tensor &input, tensor &target, float lr = 0.001, int batch_size = 64, int epoch = 10);
    tensor& forward(tensor &x);
    inline tensor& operator()(tensor &x) { tensor &r = forward(x); r.forward(); return r; }
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

#endif /* CFORMER_H */
