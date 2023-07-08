#include "../cformer.h"
#include <gtest/gtest.h>

#define first(arr) arr.scalar<float>() // get the first element of an array

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