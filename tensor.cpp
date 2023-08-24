#include "cformer.h"

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

static inline array bmean(const array &a, int dim)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[dim] = a.dims(dim);

    return af::tile(af::mean(a, dim), dims);
}

static inline array bvar(const array &a, int dim)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[dim] = a.dims(dim);

    return af::tile(af::var(a, AF_VARIANCE_POPULATION, dim), dims);
}

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

static inline void update_grad(tensor *t, const array &grad, int dim, int begin, int end)
{
    if (t->is_leaf() && !t->need_grad)
        return;
    // t->grad will be sliced in the backward, so we need to make sure it's not empty.
    if (!t->is_leaf())
        t->grad = zeros_like(t->data);
    if (dim == 0) {
        t->grad.rows(begin, end) = t->is_leaf() ? t->grad.rows(begin, end) + grad : grad;
    } else if (dim == 1) {
        t->grad.cols(begin, end) = t->is_leaf() ? t->grad.cols(begin, end) + grad : grad;
    } else
        panic("dimination must be 0 or 1");
}

static array fwd_add(tensor *a, tensor *b, tensor *p)
{
    return a->data + b->data;
}

static void bwd_add(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad);
    update_grad(b, p->grad);
}

static array fwd_addf(tensor *a, tensor *b, tensor *p)
{
    return a->data + b->scalar;
}

static void bwd_addf(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad);
}

static array fwd_sub(tensor *a, tensor *b, tensor *p)
{
    return a->data - b->data;
}

static void bwd_sub(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad);
    update_grad(b,-p->grad);
}

static array fwd_subf(tensor *a, tensor *b, tensor *p)
{
    return a->data - b->scalar;
}

static void bwd_subf(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad);
}

static array fwd_mul(tensor *a, tensor *b, tensor *p)
{
    cf_assert(a->data.dims(0) == b->data.dims(0) && a->data.dims(1) == b->data.dims(1),
        "Dimension Mismatch a[%lld, %lld] != b[%lld, %lld]", a->data.dims(0), a->data.dims(1),
        b->data.dims(0), b->data.dims(1));
    return a->data * b->data;
}

static void bwd_mul(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, b->data * p->grad);
    update_grad(b, a->data * p->grad);
}

static array fwd_mulf(tensor *a, tensor *b, tensor *p)
{
    return a->data * b->scalar;
}

static void bwd_mulf(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, b->scalar * p->grad);
}

static array fwd_div(tensor *a, tensor *b, tensor *p)
{
    return a->data / b->data;
}

static void bwd_div(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad / b->data);
    update_grad(b,-p->grad * p->data / b->data);
}

static array fwd_divf(tensor *a, tensor *b, tensor *p)
{
    return a->data / b->scalar;
}

static void bwd_divf(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad / b->scalar);
}

static array fwd_matmul(tensor *a, tensor *b, tensor *p)
{
    cf_assert(a->data.type() == b->data.type(), "Type Mismatch lhs(%s) != rhs(%s)",
              array_typename[a->data.type()], array_typename[b->data.type()]);
    cf_assert(a->data.dims(1) == b->data.dims(0), "Dimension Mismatch lhs(%lld) != rhs(%lld)",
              a->data.dims(1), b->data.dims(0));
    return af::matmul(a->data, b->data);
}

// y = a @ b => dy = y.grad, da = y.grad @ b.T, db = a.T @ y.grad
static void bwd_matmul(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, af::matmulNT(p->grad, b->data));
    update_grad(b, af::matmulTN(a->data, p->grad));
}

/**
 * Softmax of very confident network could either produce 0(underflow) or extreamly small
 * number for the y_pred.data, then forward and backward of log(y_pred) will produce inf or NaN
 * gradient problem.
 *
 * pytorch and tensorflow add EPSILON in *_cross_entropy functions to avoid log(0).
 * we clip value(v < EPSILON) with EPSILON here.
 */
static array fwd_log(tensor *a, tensor *dummy, tensor *p)
{
#define EPSILON 1e-8
    af::replace(a->data, a->data >= EPSILON, EPSILON);
    return af::log(a->data);
}

static void bwd_log(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad / a->data);
}

static array fwd_exp(tensor *a, tensor *dummy, tensor *p)
{
    return af::exp(a->data);
}

static void bwd_exp(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, p->grad * p->data);
}

static array fwd_relu(tensor *a, tensor *dummy, tensor *p)
{
    array zero = af::constant(0, a->data.dims(), a->data.type());
    return af::max(a->data, zero);
}

// y = relu(x) => dx = dy * (x > 0)
static void bwd_relu(tensor *a, tensor *dummy, tensor *p)
{
    array zero = af::constant(0, a->data.dims(), a->data.type());
    update_grad(a, af::select(a->data > zero, p->grad, zero));
}

// y = broadcast(sum(x)), sum(x) over dim and then bradcast it to same shape as x.
// sum() redues the dimension of x along dim to 1 and brad() matmul it back by a broadcasting matrix.
// broadcast(a) = B0 @ a if dim = 0, B0 = ones(a.dims[0], 1)
// broadcast(a) = a @ B1 if dim = 1, B1 = ones(1, a.dims[1])
static array fwd_bsum(tensor *a, tensor *dummy, tensor *p)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[p->param.dim] = a->data.dims(p->param.dim);
    return af::tile(af::sum(a->data, p->param.dim), dims);
}

// y = brad(sum(x)), dy = y.grad.
// dx = y.grad @ ones(d, d) if dim = 1, d = x.dims[1]
// dx = ones(d, d) @ y.grad if dim = 0, d = x.dims[0]
static void bwd_bsum(tensor *a, tensor *dummy, tensor *p)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[p->param.dim] = a->data.dims(p->param.dim);
    update_grad(a, af::tile(af::sum(p->grad, p->param.dim), dims));
}

static array fwd_sigmoid(tensor *a, tensor *dummy, tensor *p)
{
    return af::sigmoid(a->data);
}

static void bwd_sigmoid(tensor *a, tensor *dummy, tensor *p)
{
    update_grad(a, p->grad * p->data * (1 - p->data));
}

static array fwd_tanh(tensor *a, tensor *dummy, tensor *p)
{
    return af::tanh(a->data);
}

static void bwd_tanh(tensor *a, tensor *dummy, tensor *p)
{
    update_grad(a, p->grad * (1 - p->data * p->data));
}

static array fwd_sum(tensor *a, tensor *dummy, tensor *p)
{
    return af::sum(a->data, p->param.dim);
}

static void bwd_sum(tensor *a, tensor *dummy, tensor *p)
{
    af::dim4 dims = {1, 1, 1, 1};
    dims[p->param.dim] = a->data.dims(p->param.dim);
    array t = af::tile(p->grad, dims);
    update_grad(a, t);
}

static array fwd_neg(tensor *a, tensor *dummy, tensor *p)
{
    return -a->data;
}

static void bwd_neg(tensor *a, tensor *dummy, tensor *p)
{
    update_grad(a,-p->grad);
}

// For x@w + b to work, b is broadcasted to the same shape as x@w (batch_size, out).
static array fwd_expandas(tensor *a, tensor *b, tensor *p)
{
    int d = b->data.dims(0);
    cf_debug("expandas: %d", d);
    cf_assert(a->data.dims(0) == 1 && d != 0, "expandas only support dim 0");
    return af::tile(a->data, d);
}

// y = expandas(x) => dx = sum(dy, dim=0)
static void bwd_expandas(tensor *a, tensor *b, tensor *p)
{
    update_grad(a, af::sum(p->grad, 0));
}

// Suppport dim=1 right now.
static array fwd_bmax(tensor *a, tensor *dummy, tensor *p)
{
    cf_assert(p->param.dim == 1, "bmax only support dim 1");
    return bmax(a->data);
}

// y = bmax(x) => dx = bsum(dy) * onehot(max_idx)
static void bwd_bmax(tensor *a, tensor *dummpy, tensor *p)
{
    array dummy, idx;
    af::max(dummy, idx, a->data, p->param.dim);
    update_grad(a, bsum(p->grad, 1) * onehot(idx, p->grad.dims(p->param.dim)));
}

// LogSumExp(x) trick to avoid overflow/underflow,
// see https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
static array fwd_lse(tensor *a, tensor *dummy, tensor *p)
{
    return af::log(bsum(af::exp(a->data - bmax(a->data)), 1)) + bmax(a->data);
}

// y = lse(x) => dx = bsum(dy) * exp(x) / bsum(exp(x))
static void bwd_lse(tensor *a, tensor *dummy, tensor *p)
{
    array exp = af::exp(a->data - bmax(a->data));
    update_grad(a, bsum(p->grad, 1) * exp / bsum(exp, 1));
}

static array fwd_logsm(tensor *a, tensor *dummy, tensor *p)
{
    return a->data - bmax(a->data) - af::log(bsum(af::exp(a->data - bmax(a->data)), 1));
}

static void bwd_logsm(tensor *a, tensor *dummy, tensor *p)
{
    array exp = af::exp(a->data - bmax(a->data));
    update_grad(a, p->grad - bsum(p->grad, 1) * exp / bsum(exp, 1));
}

static array fwd_softmax(tensor *a, tensor *dummy, tensor *p)
{
    array exp = af::exp(a->data - bmax(a->data));
    return exp / bsum(exp, 1);
}

// y = softmax(x) => dx = softmax(x) * (dy - bsum(dy * softmax(x)))
static void bwd_softmax(tensor *a, tensor *dummy, tensor *p)
{
    // Note: higher level oper might modify output of softmax to avoid 0 (such as log)
    // so seems that using 'y' is more numerically accurate than recomputing from 'a->data'
    update_grad(a, p->data * (p->grad - bsum(p->grad * p->data, 1)));
}

static array fwd_bstd(tensor *a, tensor *dummy, tensor *p)
{
    array var = bvar(a->data, p->param.dim);
    // bwd_bstd will divide by std, so replace 0 with 1e-5, which is from pytorch's batchnorm
    af::replace(var, var >= 1e-5, 1e-5);
    return af::sqrt(var);
}

static void bwd_bstd(tensor *a, tensor *dummy, tensor *p)
{
    int d = p->param.dim;
    size_t n = a->data.dims(d);
    array mean = bmean(a->data, d);
    array bn = (a->data - mean) / p->data;
    array dx = bsum(p->grad, d) * bn / n;

    update_grad(a, dx);
}

static array fwd_submean(tensor *a, tensor *dummy, tensor *p)
{
    return a->data - bmean(a->data, p->param.dim);
}

static void bwd_submean(tensor *a, tensor *dummy, tensor *p)
{
    int d = p->param.dim;
    size_t n = a->data.dims(d);
    array mean = bmean(a->data, d);
    array dx = p->grad - bsum(p->grad, d) / n;

    update_grad(a, dx);
}

static array fwd_batchnorm(tensor *a, tensor *dummy, tensor *p)
{
    array mean = bmean(a->data, p->param.dim);
    array var = bvar(a->data, p->param.dim);
    af::replace(var, var >= 1e-5, 1e-5);
    array std = af::sqrt(var);
    return (a->data - mean) / std;
}

static void bwd_batchnorm(tensor *a, tensor *dummy, tensor *p)
{
    array y = p->data;
    size_t n = a->data.dims(p->param.dim);
    array var = bvar(a->data, p->param.dim);
    af::replace(var, var >= 1e-5, 1e-5);
    array std = af::sqrt(var);
    array dx = (p->grad - bsum(p->grad, p->param.dim) / n - y * bsum(p->grad * y, p->param.dim) / n) / std;
    update_grad(a, dx);
}

static array fwd_pow(tensor *a, tensor *dummy, tensor *p)
{
    return af::pow(a->data, p->param.p);
}

static void bwd_pow(tensor *a, tensor *dummy, tensor *p)
{
    update_grad(a, p->grad * p->param.p * af::pow(a->data, p->param.p - 1));
}

static array fwd_slice(tensor *a, tensor *dummy, tensor *p)
{
    if (p->param.dim == 0)
        return a->data.rows(p->param.int1, p->param.int2);
    else if (p->param.dim == 1)
        return a->data.cols(p->param.int1, p->param.int2);
    else
        panic("slice only support dim 0 or 1");
}

static void bwd_slice(tensor *a, tensor *dummy, tensor *p)
{
    if (p->param.dim == 0)
        update_grad(a, p->grad, p->param.dim, p->param.int1, p->param.int2);
    else if (p->param.dim == 1)
        update_grad(a, p->grad, p->param.dim, p->param.int1, p->param.int2);
    else
        panic("slice only support dim 0 or 1");
}

static array fwd_reshape(tensor *a, tensor *dummy, tensor *p)
{
    return af::moddims(a->data, p->param.dim4);
}

static void bwd_reshape(tensor *a, tensor *dummy, tensor *p)
{
    update_grad(a, af::moddims(p->grad, a->data.dims()));
}

static array fwd_transpose(tensor *a, tensor *dummy, tensor *p)
{
    return af::transpose(a->data);
}

static void bwd_transpose(tensor *a, tensor *dummy, tensor *p)
{
    update_grad(a, af::transpose(p->grad));
}

static array fwd_stack(tensor *a, tensor *b, tensor *p)
{
    return af::join(p->param.dim, b->data, a->data);
}

static void bwd_stack(tensor *a, tensor *b, tensor *p)
{
    if (p->param.dim == 0) {
        update_grad(b, p->grad.rows(0, b->data.dims(0) - 1));
        update_grad(a, p->grad.rows(b->data.dims(0), a->data.dims(0) + b->data.dims(0) - 1));
    } else if (p->param.dim == 1) {
        update_grad(b, p->grad.cols(0, b->data.dims(1) - 1));
        update_grad(a, p->grad.cols(b->data.dims(1), a->data.dims(1) + b->data.dims(1) - 1));
    } else
        panic("stack only support dim 0 or 1");
}

static array fwd_rslice(tensor *a, tensor *b, tensor *p)
{
    af::dim4 dims = a->data.dims();
    if (p->param.dim == 0)
        return af::moddims(a->data.row(p->param.int1), {dims[1], dims[2], dims[3]});
    else if (p->param.dim == 1)
        return af::moddims(a->data.col(p->param.int1), {dims[0], dims[2], dims[3]});
    else
        panic("rslice only support dim 0 or 1");
}

static void bwd_rslice(tensor *a, tensor *b, tensor *p)
{
    af::dim4 dims = a->data.dims();
    if (p->param.dim == 0)
        update_grad(a, af::moddims(p->grad, {1, dims[1], dims[2], dims[3]}), 0, p->param.int1, p->param.int1);
    else if (p->param.dim == 1)
        update_grad(a, af::moddims(p->grad, {dims[0], 1, dims[2], dims[3]}), 1, p->param.int1, p->param.int1);
    else
        panic("rslice only support dim 0 or 1");
}

static array fwd_xstack(tensor *a, tensor *b, tensor *p)
{
    af::dim4 dims = a->data.dims();
    array xa;
    if (p->param.dim == 0)
        xa = af::moddims(a->data, {1, dims[0], dims[1], dims[2]});
    else if (p->param.dim == 1)
        xa = af::moddims(a->data, {dims[0], 1, dims[1], dims[2]});
    else
        panic("xstack only support dim 0 or 1");
    return af::join(p->param.dim, b->data, xa);
}

static void bwd_xstack(tensor *a, tensor *b, tensor *p)
{
    af::dim4 dims = a->data.dims();
    if (p->param.dim == 0) {
        update_grad(b, p->grad.rows(0, b->data.dims(0) - 1));
        update_grad(a, af::moddims(p->grad.row(b->data.dims(0)), dims));
    } else if (p->param.dim == 1) {
        update_grad(b, p->grad.cols(0, b->data.dims(1) - 1));
        update_grad(a, af::moddims(p->grad.col(b->data.dims(1)), dims));
    } else
        panic("xstack only support dim 0 or 1");
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
OPERATOR(expandas);
OPERATOR(bmax);
OPERATOR(lse);
OPERATOR(logsm);
OPERATOR(softmax);
OPERATOR(submean);
OPERATOR(bstd);
OPERATOR(batchnorm);
OPERATOR(pow);
OPERATOR(addf);
OPERATOR(subf);
OPERATOR(mulf);
OPERATOR(divf);
OPERATOR(slice);
OPERATOR(reshape);
OPERATOR(transpose);
OPERATOR(stack);
OPERATOR(rslice);
OPERATOR(xstack);

#define VA_LIST(...) __VA_ARGS__
#define METHOD(name, args, new_arg, op, stmts...) \
    tensor& tensor::name(VA_LIST args) { \
        tensor *r = new tensor(this, new_arg, &oper_##op); \
        stmts; \
        return *r; \
    }
METHOD(matmul, (tensor &t), &t, matmul)
METHOD(operator+, (tensor &t), &t, add)
METHOD(operator-, (tensor &t), &t, sub)
METHOD(operator*, (tensor &t), &t, mul)
METHOD(operator/, (tensor &t), &t, div)
METHOD(operator-, (void), nullptr, neg)
METHOD(operator+, (const array &a), a, add)
METHOD(operator-, (const array &a), a, sub)
METHOD(operator*, (const array &a), a, mul)
METHOD(operator/, (const array &a), a, div)
METHOD(operator+, (float f), f, addf)
METHOD(operator-, (float f), f, subf)
METHOD(operator*, (float f), f, mulf)
METHOD(operator/, (float f), f, divf)
METHOD(log, (void), nullptr, log)
METHOD(exp, (void), nullptr, exp)
METHOD(relu, (void), nullptr, relu)
METHOD(sigmoid, (void), nullptr, sigmoid)
METHOD(tanh, (void), nullptr, tanh)
METHOD(bsum, (int dim), nullptr, bsum, r->param.dim = dim)
METHOD(sum, (int dim), nullptr, sum, r->param.dim = dim)
METHOD(expandas, (tensor &t), &t, expandas)
METHOD(bmax, (int dim), nullptr, bmax, r->param.dim = dim)
METHOD(lse, (void), nullptr, lse)
METHOD(logsm, (void), nullptr, logsm)
METHOD(softmax, (void), nullptr, softmax)
METHOD(bstd, (int dim), nullptr, bstd, r->param.dim = dim)
METHOD(submean, (int dim), nullptr, submean, r->param.dim = dim)
METHOD(batchnorm, (int dim), nullptr, batchnorm, r->param.dim = dim)
METHOD(pow, (float p), nullptr, pow, r->param.p = p)
METHOD(slice, (int dim, int begin, int end), nullptr, slice, \
       r->param.dim = dim; r->param.int1 = begin; r->param.int2 = end)
METHOD(reshape, (const af::dim4 &dims), nullptr, reshape, r->param.dim4 = dims)
METHOD(T, (void), nullptr, transpose)
METHOD(stack, (tensor &t, int dim), &t, stack, r->param.dim = dim)
METHOD(rslice, (int dim, int n), nullptr, rslice, \
       r->param.dim = dim; r->param.int1 = n;)
METHOD(xstack, (tensor &t, int dim), &t, xstack, r->param.dim = dim)

static inline tensor& detach_tensor(tensor &t)
{
/* FIXME: this is a hack to copy tensor blindly. can we do better? */
    tensor *r = new tensor();
    r->data = t.data;
    r->grad = t.grad;
    r->scalar = t.scalar;
    r->lhs = t.lhs;
    r->rhs = t.rhs;
    r->op = t.op;
    r->param = t.param;
    r->need_grad = t.need_grad;
    r->data_computed = t.data_computed;
    r->no_delete = false; // we need to delete this tensor
    return *r;
}

tensor& tensor::detach(void)
{
    return detach_tensor(*this);
}

// y += c will create a new tensor y' takes the value of y, then y = y' + c
void tensor::operator+= (tensor &t)
{
    // Note = is a copy_delete operation. can we swap it to avoid extra copy?
    *this = this->detach() + t;
}

void tensor::forward(void)
{
    if (data_computed)
        return;
    if (lhs && !lhs->data_computed)
        lhs->forward();
    if (rhs && !rhs->data_computed)
        rhs->forward();
    cf_debug("%s", op->name);
    data = op->forward_fn(lhs, rhs, this);
    data_computed = true;
}

static void do_backward(tensor *t)
{
    if (t->is_leaf())
        return;
    if (t->op->backward_fn)
        t->op->backward_fn(t->lhs, t->rhs, t);
    else
        return; // no backward for the rest of the branch
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

static void prepare_graph_nodes(tensor *t, std::unordered_set<tensor *> &nodes)
{
    if (!t->no_delete)
        nodes.insert(t);
    if (t->lhs)
        prepare_graph_nodes(t->lhs, nodes);
    if (t->rhs)
        prepare_graph_nodes(t->rhs, nodes);
}

void tensor::destroy_graph(void)
{
    // non-leaf tensors might be shared by other nodes in the graph, so we need
    // a set to avoid deleting them multiple times.
    std::unordered_set<tensor *> nodes;
    prepare_graph_nodes(this, nodes);
    for (auto t : nodes)
        delete t;
}

static void do_print(const std::string& prefix, tensor* node, bool left, bool root=false)
{
    std::cout << prefix;

    if (root)
        std::cout << "Root ";
    else
        std::cout << (left ? "|---" : "+---");
    std::cout << (node->is_leaf() ? "Leaf" : node->op->name) << (node->no_delete ? "" : "*")
        << (node->need_grad ? "" : "!") << std::endl;

    if (node->lhs)
        do_print(prefix + (left ? "|    " : "     "), node->lhs, true);
    if (node->rhs)
        do_print(prefix + (left ? "|    " : "     "), node->rhs, false);
}

void tensor::print_graph(void)
{
    do_print("", this, false, true);
}
