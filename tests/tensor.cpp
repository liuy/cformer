#include "../cformer.h"
#include <gtest/gtest.h>

#define first(arr) arr.scalar<float>() // get the first element of an array
#define index(arr, i) arr(i).scalar<float>() // get the i-th element of an array

static inline void array_eq(const array &a, std::initializer_list<float> list)
{
    ASSERT_EQ(a.elements(), list.size());
    for (int i = 0; i < list.size(); i++)
        EXPECT_NEAR(index(a, i), list.begin()[i], 1e-6);
}

static inline void array_eq(const array &a, const array &b)
{
    ASSERT_EQ(a.elements(), b.elements());
    for (int i = 0; i < a.elements(); i++)
        EXPECT_NEAR(index(a, i), index(b, i), 1e-6);
}

TEST(Tensor, add_sub_mul_div)
{
    tensor a(af::constant(1, 1, 3));
    tensor b(af::constant(2, 1, 3));
    tensor c(af::constant(3, 1, 3));
    tensor d(af::constant(4, 1, 3));
    tensor &y = a * b;
    tensor &z = (y - d) / (y + c);
    z.backward();
    EXPECT_FLOAT_EQ(first(a.grad), 0.56);
    EXPECT_FLOAT_EQ(first(b.grad), 0.28);
    EXPECT_FLOAT_EQ(first(c.grad), 0.08);
    EXPECT_FLOAT_EQ(first(d.grad), -0.20);
    EXPECT_FLOAT_EQ(first(z.data), -0.40);
    z.destroy_graph();
}

static tensor& softmax(tensor &x, int dim)
{
    return x.exp()/x.exp().bsum(dim);
}

TEST(Tensor, exp_bsum)
{
    tensor a(af::randu(2, 3));
    tensor b(af::randu(2, 3));
    tensor c(af::randu(2, 3));

    tensor &y = softmax(b-a*c, 1);
    y.backward();
    EXPECT_NEAR(af::sum<float>(a.grad), 0, 1e-6);
    EXPECT_NEAR(af::sum<float>(b.grad), 0, 1e-6);
    EXPECT_NEAR(af::sum<float>(c.grad), 0, 1e-6);
    EXPECT_FLOAT_EQ(af::sum<float>(y.data), 2);
    y.destroy_graph();

    a.assign_data(af::randu(1, 3));
    b.assign_data(af::randu(1, 3));
    c.assign_data(af::randu(1, 3));
    tensor &z = softmax(b-a*c, 0);
    z.backward();
    EXPECT_NEAR(af::sum<float>(a.grad), 0, 1e-6);
    EXPECT_NEAR(af::sum<float>(b.grad), 0, 1e-6);
    EXPECT_NEAR(af::sum<float>(c.grad), 0, 1e-6);
    EXPECT_FLOAT_EQ(af::sum<float>(z.data), 3);
    z.destroy_graph();
}

TEST(Tensor, relu_tanh_sigmoid)
{
    af::setSeed(0);

    tensor a(af::constant(2, 1, 3));
    tensor b(af::constant(3, 1, 3));
    tensor c(kaiming_uniform(1, 3));

    tensor &x = (c * a - b).relu();
    x.backward();
    array_eq(x.data, {0.f, 0.f, 1.7084155f});
    array_eq(c.data, {0.4945693f, -2.3135002f, 2.3542078f});
    array_eq(a.grad, {0.f, 0.f, 2.3542078f});
    array_eq(b.grad, {0.f, 0.f, -1.f});
    array_eq(c.grad, {0.f, 0.f, 2.f});
    x.destroy_graph();
    a.zero_grad();
    b.zero_grad();
    c.zero_grad();

    tensor &y = (c * a - b).tanh();
    y.backward();
    array_eq(y.data, {-0.9647869f, -0.9999995f, 0.9364529f});
    array_eq(a.grad, {0.0342173f, -0.0000022f, 0.2896995f});
    array_eq(b.grad, {-0.0691862f, -0.0000010f, -0.1230561f});
    array_eq(c.grad, {0.1383723f, 0.0000019f, 0.2461121f});
    y.destroy_graph();
    a.zero_grad();
    b.zero_grad();
    c.zero_grad();

    tensor &z = (c * a - b).sigmoid();
    z.backward();
    array_eq(z.data, {0.1180672f, 0.0004869f, 0.8466307f});
    array_eq(a.grad, {0.0514982f, -0.0011259f, 0.3056872f});
    array_eq(b.grad, {-0.1041274f, -0.0004866f, -0.1298472f});
    array_eq(c.grad, {0.2082547f, 0.0009733f, 0.2596943f});
    z.destroy_graph();
}

static tensor& forward(tensor &x)
{
    static tensor w(af::constant(2, 1, 3));
    static tensor b(af::constant(3, 1, 3));

    return w * x + b;
}

TEST(Tensor, stacked_var)
{
    tensor x = af::constant(2, 1, 3);
    tensor out1 = forward(x);
    tensor out2 = forward(out1);

    out2.backward();
    array_eq(out2.data, {17.f, 17.f, 17.f});
    array_eq(x.grad, {4.f, 4.f, 4.f});
}

TEST(Tensor, add_assign)
{
    tensor a(af::constant(1, 1, 3));
    tensor b(af::constant(2, 3, 3));
    tensor c(af::constant(3, 1, 3));
    tensor &z = a.matmul(b);
    z += c;
    z.backward();
    EXPECT_FLOAT_EQ(first(z.data), 9);
    EXPECT_FLOAT_EQ(first(a.grad), 6);
    EXPECT_FLOAT_EQ(first(b.grad), 1);
    EXPECT_FLOAT_EQ(first(c.grad), 1);
    z.destroy_graph();

    a.zero_grad();
    b.zero_grad();
    c.zero_grad();

    tensor y = a.matmul(b);
    y += c;
    y.backward();
    EXPECT_FLOAT_EQ(first(y.data), 9);
    EXPECT_FLOAT_EQ(first(a.grad), 6);
    EXPECT_FLOAT_EQ(first(b.grad), 1);
    EXPECT_FLOAT_EQ(first(c.grad), 1);
    y.destroy_graph();
}

TEST(Tensor, sum_neg)
{
    tensor a(af::randu(2, 3));
    tensor s = a.sum(0);

    s.backward();
    EXPECT_FLOAT_EQ(first(s.data), af::sum<float>(a.data.col(0)));
    EXPECT_FLOAT_EQ(index(s.data, 1), af::sum<float>(a.data.col(1)));
    EXPECT_FLOAT_EQ(index(s.data, 2), af::sum<float>(a.data.col(2)));
    array_eq(a.grad, af::constant(1, 2, 3));
    s.destroy_graph();
    a.zero_grad();

    s = -a.sum(1);
    s.backward();
    EXPECT_FLOAT_EQ(first(-s.data), af::sum<float>(a.data.row(0)));
    EXPECT_FLOAT_EQ(index(-s.data, 1), af::sum<float>(a.data.row(1)));
    array_eq(a.grad, af::constant(-1, 2, 3));
    s.destroy_graph();
}

TEST(Tensor, bdim0)
{
    tensor x(af::constant(2, 1, 3));
    tensor y(af::constant(3, 2, 3));
    tensor &z = y * x.bdim0(2);
    z.backward();
    EXPECT_FLOAT_EQ(first(z.data), 6);
    EXPECT_FLOAT_EQ(first(x.grad), 6);
    EXPECT_FLOAT_EQ(first(y.grad), 2);
    z.destroy_graph();
}