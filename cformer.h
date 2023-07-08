#ifndef CFORMER_H
#define CFORMER_H

#include <arrayfire.h>
#include <unordered_set>
#include <iostream>
#include <string>

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
    oper *op; // operator of the expression

    tensor(const array &a): data(a), grad(af::constant(0, a.dims())) {}
    tensor(const tensor &t) = delete; // No tensor y = a; use tensor &y = a instead
    tensor(tensor *a, tensor *b, oper *o) : lhs(a), rhs(b), op(o) {}
    //~tensor() { printf("~tensor() %p\n", this); }

    // we overload the operators to construct a computational graph. For e.g,
    //  y = a * b + a / c will be constructed as follows:
    //        y(+)        y  = t1 + t2
    //       /   \
    //    t1(*)  t2(/)    t1 = a * b
    //     /\     /\      t2 = a / c
    //    a  b   a  c
    // Then we call forward() on y to evaluate the expressions and backward() to
    // compute the gradients. With .forward() and .backward() we implement the
    // so-called "autograd"(Automatic Differentiation) in ML frameworks like PyTorch
    // and TensorFlow. Temporary tensors t1, t2, y are created on the heap and call
    // .destroy_graph() to delete them after use.
    // For e.g, if we call y.backward() on the above graph, we will get by Chain Rule:
    // use da to denote dL/da, and so on...
    // y.grad(dy) = 1, t1.grad(dt1) = 1, t2.grad(dt2) = 1,
    // a.grad(da) = b + 1/c, b.grad(db) = a, c.grad(dc) = -a/c^2
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
    /// @brief Sum all values along dimension dim and broadcast to the original shape.
    /// @param dim The dimension along which the add operation occurs.
    tensor& bsum(int);
    tensor& operator+(tensor &t);
    tensor& operator-(tensor &t);
    tensor& operator*(tensor &t);
    tensor& operator/(tensor &t);
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
#else
    #define af_debug(...) do {} while (0)
    #define cf_debug(...) do {} while (0)
#endif

#define panic(fmt, ...) do { \
    fprintf(stderr, "%s:%d:%s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__); \
    exit(EXIT_FAILURE); \
    } while (0)

#endif /* CFORMER_H */