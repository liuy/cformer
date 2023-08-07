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
    tensor a(af::constant(1, 1, 3), true);
    tensor b(af::constant(2, 1, 3), true);
    tensor c(af::constant(3, 1, 3), true);
    tensor d(af::constant(4, 1, 3), true);
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

static tensor& test_softmax(tensor &x, int dim)
{
    return x.exp()/x.exp().bsum(dim);
}

TEST(Tensor, exp_bsum)
{
    tensor a(af::randu(2, 3), true);
    tensor b(af::randu(2, 3), true);
    tensor c(af::randu(2, 3), true);

    tensor &y = test_softmax(b-a*c, 1);
    y.backward();
    EXPECT_NEAR(af::sum<float>(a.grad), 0, 1e-6);
    EXPECT_NEAR(af::sum<float>(b.grad), 0, 1e-6);
    EXPECT_NEAR(af::sum<float>(c.grad), 0, 1e-6);
    EXPECT_FLOAT_EQ(af::sum<float>(y.data), 2);
    y.destroy_graph();

    a.assign_data(af::randu(1, 3));
    b.assign_data(af::randu(1, 3));
    c.assign_data(af::randu(1, 3));
    tensor &z = test_softmax(b-a*c, 0);
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

    tensor a(af::constant(2, 1, 3), true);
    tensor b(af::constant(3, 1, 3), true);
    tensor c(kaiming_uniform(1, 3), true);

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
    static tensor w(af::constant(2, 1, 3), true);
    static tensor b(af::constant(3, 1, 3), true);

    return w * x + b;
}

TEST(Tensor, stacked_var)
{
    tensor x(af::constant(2, 1, 3), true);
    tensor out1 = forward(x);
    tensor out2 = forward(out1);

    out2.backward();
    array_eq(out2.data, {17.f, 17.f, 17.f});
    array_eq(x.grad, {4.f, 4.f, 4.f});
}

TEST(Tensor, add_assign)
{
    tensor a(af::constant(1, 1, 3), true);
    tensor b(af::constant(2, 3, 3), true);
    tensor c(af::constant(3, 1, 3), true);
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
    tensor a(af::randu(2, 3), true);
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
    tensor x(af::constant(2, 1, 3), true);
    tensor y(af::constant(3, 2, 3), true);
    tensor &z = y * x.bdim0(y);
    z.backward();
    EXPECT_FLOAT_EQ(first(z.data), 6);
    EXPECT_FLOAT_EQ(first(x.grad), 6);
    EXPECT_FLOAT_EQ(first(y.grad), 2);
    z.destroy_graph();
}

TEST(Tensor, log)
{
    tensor zero(af::constant(0, 1, 3), true);
    tensor log = zero.log();
    log.forward();
    EXPECT_FLOAT_EQ(first(log.data), -18.420681f);

    tensor x(array({1,2}, {100000.0f,0.0f}), true);
    tensor y_true(array({1,2}, {1.0f, 0.0f}));
    tensor &y_hat = x.softmax();
    tensor &z = categorical_cross_entropy(y_true, y_hat);
    z.backward();
    EXPECT_FLOAT_EQ(first(z.data), 0.0f);
    array_eq(y_hat.data, {1.0f, 0.0f});
    array_eq(y_hat.grad, {-1.0f, -0.0f});
    array_eq(x.grad, {0.0f, 0.0f});
    z.destroy_graph();
}

TEST(Tensor, bmax)
{
    af::dim4 d={2,3};
    tensor a(array(d, {0.6009535f,0.0277588f,0.9805506f,0.2126322f,0.0654638f,0.5496738f}), true);
    tensor b(af::constant(2,2,3), true);
    tensor &m = b * a.bmax(1);
    m.backward();
    array_eq(m.data, {1.9611012f, 1.0993476f, 1.9611012f, 1.0993476f, 1.9611012f, 1.0993476f});
    array_eq(a.grad, {0.0f,0.0f,6.0f,0.0f,0.0f,6.0f});
    array_eq(b.grad, {0.9805506f,0.5496738f,0.9805506f,0.5496738f,0.9805506f,0.5496738f});

    m.destroy_graph();
}

static inline tensor& log_softmax(tensor &x)
{
    return x - x.lse();
}

TEST(Tensor, lse_logsm)
{
    tensor x(array({2,3}, {1.0f, 1.0f, 2.0f, 10000.0f, 3.0f, 5.0f}), true);
    tensor &y = log_softmax(x);
    y.backward();
    af_print(y.data, 7);
    af_print(x.grad, 7);
    array_eq(y.data, {-2.4076059f, -9999.0f, -1.4076059f, 0.0f, -0.4076059f, -9995.0f});
    array_eq(x.grad, {0.7299083f, 1.0f, 0.2658145f, -2.0f, -0.9957228f, 1.0f});
    y.destroy_graph();
    x.zero_grad();

    tensor &l = x.logsm();
    l.backward();
    array_eq(l.data, {-2.4076059f, -9999.0f, -1.4076059f, 0.0f, -0.4076059f, -9995.0f});
    array_eq(x.grad, {0.7299083f, 1.0f, 0.2658145f, -2.0f, -0.9957228f, 1.0f});
    l.destroy_graph();
}

TEST(Tensor, softmax)
{
    tensor x(array({2,3}, {1.0f, 1.0f, 2.0f, 10000.0f, 3.0f, 5.0f}), true);
    tensor &y = x.softmax();
    y.backward();
    array_eq(y.data, {0.0900306f, 0.0f, 0.2447285f, 1.0f, 0.6652409f, 0.0f});
    array_eq(x.grad, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    y.destroy_graph();
    x.zero_grad();

    // log_softmax != log(softmax) in the sense that the former is more numerically stable and accurate
    tensor &l = x.softmax().log();
    l.backward();
    af_print(l.data, 8);
    af_print(x.grad, 8);
    array_eq(l.data, {-2.4076059f, -18.420681f, -1.4076059f, 0.0f, -0.4076059f, -18.420681f});
    array_eq(x.grad, {0.7299082f, 1.0f, 0.2658145f, -2.0f, -0.9957228f, 1.0f});
    l.destroy_graph();
}

TEST(Tensor, bstd)
{
    tensor x(array({2,3}, {1.0f, 3.0f, 2.0f, 3.0f, 3.0f, 3.0f}), true);
    tensor &y = x.bstd();
    y.backward();
    array_eq(y.data, {0.816496f, 0.003162f, 0.816496f, 0.003162f, 0.816496f, 0.003162f});
    array_eq(x.grad, {-1.224744f, 0.0f, 0.0f, 0.0f, 1.224744f, 0.0f});

    x.zero_grad();
    y.backward(array({2,3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
    array_eq(y.data, {0.816496f, 0.003162f, 0.816496f, 0.003162f, 0.816496f, 0.003162f});
    array_eq(x.grad, {-2.449489f, 0.0f, 0.0f, 0.0f,  2.449489f, 0.0f});
    y.destroy_graph();
}

TEST(Tensor, submean_bstd)
{
    tensor x(array({2,3}, {1.0f, 3.0f, 2.0f, 3.0f, 3.0f, 3.0f}), true);
    tensor y = x.submean();
    y.backward();
    array_eq(y.data, {-1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    array_eq(x.grad, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    x.zero_grad();

    y.backward(array({2,3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    array_eq(y.data, {-1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    array_eq(x.grad, {-2.0f, -2.0f, 0.0f, 0.0f, 2.0f, 2.0f});
    y.destroy_graph();
    x.zero_grad();

    y = x.submean() / x.bstd();
    y.backward();
    array_eq(y.data, {-1.224744f, 0.0f, 0.0f, 0.0f, 1.224744f, 0.0f});
    array_eq(x.grad, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    x.zero_grad();

    y.backward(array({2,3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    af_print(y.data, 8);
    af_print(x.grad, 8);
    array_eq(y.data, {-1.224744f, 0.0f, 0.0f, 0.0f, 1.224744f, 0.0f});
    array_eq(x.grad, {0.0f, -632.455566f, 0.0f, 0.0f, 0.0f, 632.455444f});
    y.destroy_graph();
}

TEST(Tensor, add_sub_mul_div_array)
{
    tensor x(af::constant(3, 1, 3), true);
    tensor &mul = x * af::constant(2, 1, 3);
    tensor &y = (mul + af::constant(3, 1, 3)) / (mul - af::constant(3, 1, 3));
    y.backward();
    array_eq(y.data, {3.0f, 3.0f, 3.0f});
    array_eq(x.grad, {-1.333333f, -1.333333f, -1.333333f});
    y.destroy_graph();
}

TEST(Tensor, add_sub_mul_div_float)
{
    tensor x(af::constant(3, 1, 3), true);
    tensor &mul = x * 2.0f;
    tensor &y = (mul + 3.0f) / (mul - 3.0f);
    y.backward();
    array_eq(y.data, {3.0f, 3.0f, 3.0f});
    array_eq(x.grad, {-1.333333f, -1.333333f, -1.333333f});
    y.destroy_graph();
}

TEST(Tensor, pow)
{
    tensor x(af::constant(3, 1, 3), true);
    tensor y = x.pow(2);
    y.backward();
    array_eq(y.data, {9.0f, 9.0f, 9.0f});
    array_eq(x.grad, {6.0f, 6.0f, 6.0f});
    y.destroy_graph();
    x.zero_grad();

    y = x.pow(0.5);
    y.backward();
    array_eq(y.data, {1.732051f, 1.732051f, 1.732051f});
    array_eq(x.grad, {0.288675f, 0.288675f, 0.288675f});
    y.destroy_graph();
}