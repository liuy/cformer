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
