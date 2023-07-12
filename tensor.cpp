#include "cformer.h"

// Leaf tensors might be shared by non-leaf tensors in the graph.
// So we need to accumulate the gradients for leaf tensors.
static inline void update_grad(tensor *t, const array &grad)
{
    t->grad = t->is_leaf() ? t->grad + grad : grad;
}

static array fwd_add(tensor *a, tensor *b)
{
    return a->data + b->data;
}

static void bwd_add(tensor *a, tensor *b, array &grad)
{
    update_grad(a, grad);
    update_grad(b, grad);
}

static array fwd_sub(tensor *a, tensor *b)
{
    return a->data - b->data;
}

static void bwd_sub(tensor *a, tensor *b, array &grad)
{
    update_grad(a, grad);
    update_grad(b, -grad);
    af_debug(grad, a->grad, b->grad);
}

static array fwd_mul(tensor *a, tensor *b)
{
    return a->data * b->data;
}

static void bwd_mul(tensor *a, tensor *b, array &grad)
{
    update_grad(a, b->data * grad);
    update_grad(b, a->data * grad);
    af_debug(grad, a->grad, b->grad);
}

static array fwd_div(tensor *a, tensor *b)
{
    return a->data / b->data;
}

static void bwd_div(tensor *a, tensor *b, array &grad)
{
    update_grad(a, grad / b->data);
    update_grad(b, -grad * a->data / (b->data * b->data));
    af_debug(grad, a->grad, b->grad);
}

static array fwd_matmul(tensor *a, tensor *b)
{
    return af::matmul(a->data, b->data);
}

// y = a @ b => dy = y.grad, da = y.grad @ b.T, db = a.T @ y.grad
static void bwd_matmul(tensor *a, tensor *b, array &grad)
{
    update_grad(a, af::matmulNT(grad, b->data));
    update_grad(b, af::matmulTN(a->data, grad));
}

static array fwd_log(tensor *a, tensor *dummy)
{
    return af::log(a->data);
}

static void bwd_log(tensor *a, tensor *b, array &grad)
{
    update_grad(a, grad / a->data);
}

static array fwd_exp(tensor *a, tensor *dummy)
{
    return af::exp(a->data);
}

static void bwd_exp(tensor *a, tensor *b, array &grad)
{
    update_grad(a, grad * af::exp(a->data));
    af_debug(grad, a->grad);
}

static array fwd_relu(tensor *a, tensor *dummy)
{
    array zero = af::constant(0, a->data.dims(), a->data.type());
    return af::max(a->data, zero);
}

// y = relu(x) => dx = dy * (x > 0)
static void bwd_relu(tensor *a, tensor *dummy, array &grad)
{
    array zero = af::constant(0, a->data.dims(), a->data.type());
    update_grad(a, af::select(a->data > zero, grad, zero));
    af_debug(grad, a->grad);
}

// y = broadcast(sum(x)), sum(x) over dim and then bradcast it to same shape as x.
// sum() redues the dimension of x along dim to 1 and brad() matmul it back by a broadcasting matrix.
// broadcast(a) = B0 @ a if dim = 0, B0 = ones(a.dims[0], 1)
// broadcast(a) = a @ B1 if dim = 1, B1 = ones(1, a.dims[1])
static array fwd_bsum(tensor *a, tensor *dummy)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[a->dim] = a->data.dims(a->dim);
    return af::tile(af::sum(a->data, a->dim), dims);
}

// y = brad(sum(x)), dy = y.grad.
// dx = y.grad @ ones(d, d) if dim = 1, d = x.dims[1]
// dx = ones(d, d) @ y.grad if dim = 0, d = x.dims[0]
static void bwd_bsum(tensor *a, tensor *dummy, array &grad)
{
    array t;
    int d = a->data.dims(a->dim);
    if (a->dim == 0)
        t = af::matmul(af::constant(1, d, d), grad);
    else if (a->dim == 1)
        t = af::matmul(grad, af::constant(1, d, d));
    else
        panic("dim must be 0 or 1, but got %d", a->dim);
    update_grad(a, t);
    af_debug(grad, a->grad);
}

static array fwd_sigmoid(tensor *a, tensor *dummy)
{
    return af::sigmoid(a->data);
}

static void bwd_sigmoid(tensor *a, tensor *dummy, array &grad)
{
    array sig = af::sigmoid(a->data);
    update_grad(a, grad * sig * (1 - sig));
    af_debug(grad, a->grad);
}

static array fwd_tanh(tensor *a, tensor *dummy)
{
    return af::tanh(a->data);
}

static void bwd_tanh(tensor *a, tensor *dummy, array &grad)
{
    array tanh = af::tanh(a->data);
    update_grad(a, grad * (1 - tanh * tanh));
    af_debug(grad, a->grad);
}

#define OPERATOR(name) static oper oper_##name = {#name, fwd_##name, bwd_##name}
OPERATOR(add);
OPERATOR(sub);
OPERATOR(mul);
OPERATOR(div);
OPERATOR(matmul);
OPERATOR(log);
OPERATOR(exp);
OPERATOR(relu);
OPERATOR(sigmoid);
OPERATOR(tanh);
OPERATOR(bsum);

#define METHOD(name, arg, new_arg, op, ...) tensor& tensor::name(arg) \
    { __VA_ARGS__ ; tensor *r = new tensor(this, new_arg, &oper_##op); return *r;}
METHOD(matmul, tensor &t, &t, matmul)
METHOD(operator+, tensor &t, &t, add)
METHOD(operator-, tensor &t, &t, sub)
METHOD(operator*, tensor &t, &t, mul)
METHOD(operator/, tensor &t, &t, div)
METHOD(log, void, nullptr, log)
METHOD(exp, void, nullptr, exp)
METHOD(relu, void, nullptr, relu)
METHOD(sigmoid, void, nullptr, sigmoid)
METHOD(tanh, void, nullptr, tanh)
METHOD(bsum, int dim, nullptr, bsum, this->dim = dim)

void tensor::forward(void)
{
    if (is_leaf())
        return;
    if (lhs && !lhs->is_leaf())
        lhs->forward();
    if (rhs && !rhs->is_leaf())
        rhs->forward();
    cf_debug("%s", op->name);
    data = op->forward_fn(lhs, rhs);
    //data_computed = true;
    grad = af::constant(0, data.dims());
}

static void do_backward(tensor *t)
{
    if (t->is_leaf())
        return;
    if (t->op && t->op->backward_fn)
        t->op->backward_fn(t->lhs, t->rhs, t->grad);
    if (t->lhs)
        do_backward(t->lhs);
    if (t->rhs)
        do_backward(t->rhs);
}

// The gradient of root tensor is initialized as ones with the same shape as the
// data tensor. Then we recursively compute the gradients of each tensor against
// the root tensor in the computational graph by Chain Rule in DFS order.
void tensor::backward(void)
{
    if (!is_leaf())
        forward();
    grad = af::constant(1, data.dims());
    do_backward(this);
}

static void get_nonleafs(tensor *t, std::unordered_set<tensor *> &nonleafs)
{
    if (!t->is_leaf())
        nonleafs.insert(t);
    if (t->lhs)
        get_nonleafs(t->lhs, nonleafs);
    if (t->rhs)
        get_nonleafs(t->rhs, nonleafs);
}

void tensor::destroy_graph(void)
{
    // non-leaf tensors might be shared by other nodes in the graph, so we need
    // a set to avoid deleting them multiple times.
    std::unordered_set<tensor *> nonleafs;
    get_nonleafs(this, nonleafs);
    for (auto t : nonleafs)
        delete t;
}

void do_print(const std::string& prefix, tensor* node, bool left, bool root=false)
{
    std::cout << prefix;

    if (root)
        std::cout << "Root ";
    else
        std::cout << (left ? "|---" : "+---");

    std::cout << (node->is_leaf() ? "Leaf" : node->op->name) << std::endl;

    if (node->lhs)
        do_print(prefix + (left ? "|    " : "     "), node->lhs, true);
    if (node->rhs)
        do_print(prefix + (left ? "|    " : "     "), node->rhs, false);
}

void tensor::print_graph(void)
{
    do_print("", this, false, true);
}

