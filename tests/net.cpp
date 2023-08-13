#include "../cformer.h"
#include <gtest/gtest.h>

static void test_reader(struct data &d)
{
    d.train_x.data = af::randu(60, 300);
    d.train_y.data = onehot(argmax(af::randu(60, 10)));
}

TEST(Net, Linear_BatchNorm1d_Dropout)
{
    seqnet model {
        new Linear(300, 100, ReLU),
        new BatchNorm1d(100),
        new Dropout(0.1),
        new Linear(100, 10, LogSoftmax),
    };
    //printf("%s\n", ::testing::UnitTest::GetInstance()->original_working_dir());
    data set(test_reader, false);
    set.load();
    SGD op(model.params);
    trainer tr = {
        .epochs = 12,
        .batch_size = 30,
        .optimizer = op,
        .loss_fn = log_softmax_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };
    model.summary();
    model.train(set, tr);
    tensor y_pred = model(set.train_x);
    printf("Train accuracy: %.4f\n", categorical_accuracy(set.train_y, y_pred));
    ASSERT_GT(categorical_accuracy(set.train_y, y_pred), 0.9);
}

