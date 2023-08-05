#include "cformer.h"

// Leaf tensors might be shared by non-leaf tensors in the graph.
// So we need to accumulate the gradients for leaf tensors.
static inline void update_grad(tensor *t, const array &grad)
{
    if (t->is_leaf() && !t->need_grad)
        return;
    // This is the hottes path in the training, we cann't keep it.
    // uncomment it when you hit "inf or NaN" problems to debug.
    // if (unlikely(tensor_has_inf_nan(grad))) {
    //     t->print_graph();
    //     af_print(t->data);
    //     af_print(grad);
    //     panic("got inf or nan problem");
    // }
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
}

static array fwd_mul(tensor *a, tensor *b)
{
    return a->data * b->data;
}

static void bwd_mul(tensor *a, tensor *b, array &grad)
{
    update_grad(a, b->data * grad);
    update_grad(b, a->data * grad);
}

static array fwd_div(tensor *a, tensor *b)
{
    return a->data / b->data;
}

static void bwd_div(tensor *a, tensor *b, array &grad)
{
    update_grad(a, grad / b->data);
    update_grad(b, -grad * a->data / (b->data * b->data));
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

/**
 * Softmax of very confident network could either produce 0(underflow) or extreamly small
 * number for the y_pred.data, then forward and backward of log(y_pred) will produce inf or NaN
 * gradient problem.
 *
 * pytorch and tensorflow add EPSILON in *_cross_entropy functions to avoid log(0).
 * we clip value(v < EPSILON) with EPSILON here.
 */
static array fwd_log(tensor *a, tensor *dummy)
{
#define EPSILON 1e-8
    af::replace(a->data, a->data >= EPSILON, EPSILON);
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
    af::dim4 dims = {1, 1, 1, 1};
    dims[a->dim] = a->data.dims(a->dim);
    update_grad(a, af::tile(af::sum(grad, a->dim), dims));
}

static array fwd_sigmoid(tensor *a, tensor *dummy)
{
    return af::sigmoid(a->data);
}

static void bwd_sigmoid(tensor *a, tensor *dummy, array &grad)
{
    array sig = af::sigmoid(a->data);
    update_grad(a, grad * sig * (1 - sig));
}

static array fwd_tanh(tensor *a, tensor *dummy)
{
    return af::tanh(a->data);
}

static void bwd_tanh(tensor *a, tensor *dummy, array &grad)
{
    array tanh = af::tanh(a->data);
    update_grad(a, grad * (1 - tanh * tanh));
}

static array fwd_sum(tensor *a, tensor *dummy)
{
    return af::sum(a->data, a->dim);
}

static void bwd_sum(tensor *a, tensor *dummy, array &grad)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[a->dim] = a->data.dims(a->dim);
    array t = af::tile(grad, dims);
    update_grad(a, t);
}

static array fwd_neg(tensor *a, tensor *dummy)
{
    return -a->data;
}

static void bwd_neg(tensor *a, tensor *dummy, array &grad)
{
    update_grad(a, -grad);
}

// For x@w + b to work, b is broadcasted to the same shape as x@w (batch_size, out).
static array fwd_bdim0(tensor *a, tensor *b)
{
    int d = b->data.dims(0);
    cf_debug("bdim0: %d", d);
    assert(a->data.dims(0) == 1 && d != 0);
    return af::tile(a->data, d);
}

// y = bdim0(x) => dx = sum(dy, dim=0)
static void bwd_bdim0(tensor *a, tensor *b, array &grad)
{
    update_grad(a, af::sum(grad, 0));
}

static inline array bmax(const array &a)
{
    return af::tile(af::max(a, 1), 1, a.dims(1));
}

static inline array bsum(const array &a)
{
    return af::tile(af::sum(a, 1), 1, a.dims(1));
}

// Suppport dim=1 right now, TODO: need refine onehot to support more dims
static array fwd_bmax(tensor *a, tensor *dummy)
{
    assert(a->dim == 1);
    return bmax(a->data);
}

// y = bmax(x) => dx = bsum(dy) * onehot(max_idx)
static void bwd_bmax(tensor *a, tensor *dummpy, array &grad)
{
    array dummy, idx;
    af::max(dummy, idx, a->data, a->dim);
    update_grad(a, bsum(grad) * onehot(idx, grad.dims(a->dim)));
}

// LogSumExp(x) trick to avoid overflow/underflow,
// see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
static array fwd_lse(tensor *a, tensor *dummy)
{
    return af::log(bsum(af::exp(a->data - bmax(a->data)))) + bmax(a->data);
}

// y = lse(x) => dx = bsum(dy) * exp(x) / bsum(exp(x))
static void bwd_lse(tensor *a, tensor *dummy, array &grad)
{
    array exp = af::exp(a->data - bmax(a->data));
    update_grad(a, bsum(grad) * exp / bsum(exp));
}

static array fwd_logsm(tensor *a, tensor *dummy)
{
    return a->data - bmax(a->data) - af::log(bsum(af::exp(a->data - bmax(a->data))));
}

static void bwd_logsm(tensor *a, tensor *dummy, array &grad)
{
    array exp = af::exp(a->data - bmax(a->data));
    update_grad(a, grad - bsum(grad) * exp / bsum(exp));
}

static array fwd_softmax(tensor *a, tensor *dummy)
{
    array exp = af::exp(a->data - bmax(a->data));
    return exp / bsum(exp);
}

// y = softmax(x) => dx = softmax(x) * (dy - bsum(dy * softmax(x)))
static void bwd_softmax(tensor *a, tensor *dummy, array &grad)
{
    array exp = af::exp(a->data - bmax(a->data));
    array sm = exp / bsum(exp);
    update_grad(a, sm * (grad - bsum(grad * sm)));
}

//array g = grad - bsum(grad) / batch_size; // d(x - mean) = dy - bsum(dy) / batch_size
// Support dim=1 only right now.
static inline array bmean(const array &a)
{
    return af::tile(af::mean(a, 1), 1, a.dims(1));
}

// Support dim=1 only right now.
static inline array bvar(const array &a)
{
    return af::tile(af::var(a, AF_VARIANCE_POPULATION, 1), 1, a.dims(1));
}

// Support dim=1 only right now.
static inline array bstd(const array &a)
{
    return af::tile(af::stdev(a, AF_VARIANCE_POPULATION, 1), 1, a.dims(1));
}

static array fwd_bstd(tensor *a, tensor *dummy)
{
    return bstd(a->data);
}

static void bwd_bstd(tensor *a, tensor *dummy, array &grad)
{
    size_t batch_size = a->data.dims(1);
    array mean = bmean(a->data);
    array std = bstd(a->data);
    af::replace(std, std >= 1e-5, 1e-5); // avoid divide by zero. batchnorm of pytorch use 1e-5
    array y = (a->data - mean) / std;
    array dx = bsum(grad) * y / batch_size;

    update_grad(a, dx);
}

#define OPERATOR(name) static oper oper_##name = {#name, fwd_##name, bwd_##name}
OPERATOR(add);
OPERATOR(sub);
OPERATOR(mul);
OPERATOR(div);
OPERATOR(neg);
OPERATOR(matmul);
OPERATOR(log);
OPERATOR(exp);
OPERATOR(relu);
OPERATOR(sigmoid);
OPERATOR(tanh);
OPERATOR(bsum);
OPERATOR(sum);
OPERATOR(bdim0);
OPERATOR(bmax);
OPERATOR(lse);
OPERATOR(logsm);
OPERATOR(softmax);
OPERATOR(bstd);

#define METHOD(name, arg, new_arg, op, ...) tensor& tensor::name(arg) \
    { __VA_ARGS__ ; tensor *r = new tensor(this, new_arg, &oper_##op); return *r;}
METHOD(matmul, tensor &t, &t, matmul)
METHOD(operator+, tensor &t, &t, add)
METHOD(operator-, tensor &t, &t, sub)
METHOD(operator*, tensor &t, &t, mul)
METHOD(operator/, tensor &t, &t, div)
METHOD(operator-, void, nullptr, neg)
METHOD(log, void, nullptr, log)
METHOD(exp, void, nullptr, exp)
METHOD(relu, void, nullptr, relu)
METHOD(sigmoid, void, nullptr, sigmoid)
METHOD(tanh, void, nullptr, tanh)
METHOD(bsum, int dim, nullptr, bsum, this->dim = dim)
METHOD(sum, int dim, nullptr, sum, this->dim = dim)
METHOD(bdim0, tensor &t, &t, bdim0)
METHOD(bmax, int dim, nullptr, bmax, this->dim = dim)
METHOD(lse, void, nullptr, lse)
METHOD(logsm, void, nullptr, logsm)
METHOD(softmax, void, nullptr, softmax)
METHOD(bstd, void, nullptr, bstd)

// y += c will create a new tensor y' takes the value of y, then y = y' + c
void tensor::operator+= (tensor &t)
{
    /* FIXME: this is a hack to copy tensor blindly. can we do better? */
    tensor *tmp = new tensor(this->lhs, this->rhs, this->op); // no_delete = false
    tmp->data = this->data;
    tmp->grad = this->grad;
    tmp->dim = this->dim;
    // Note = is a copy_delete operation. can we swap it to avoid extra copy?
    *this = *tmp + t;
}

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

void tensor::backward(const array &g)
{
    if (!is_leaf())
        forward();
    grad = g;
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
        if (!t->no_delete)
            delete t;
}

static void do_print(const std::string& prefix, tensor* node, bool left, bool root=false)
{
    std::cout << prefix;

    if (root)
        std::cout << "Root ";
    else
        std::cout << (left ? "|---" : "+---");
    std::cout << (node->is_leaf() ? "Leaf" : node->op->name) << (node->no_delete ? "" : "*") << std::endl;

    if (node->lhs)
        do_print(prefix + (left ? "|    " : "     "), node->lhs, true);
    if (node->rhs)
        do_print(prefix + (left ? "|    " : "     "), node->rhs, false);
}

void tensor::print_graph(void)
{
    do_print("", this, false, true);
}

