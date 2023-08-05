#include "cformer.h"

int main(int argc, char* argv[])
{
    seqnet model {
        new linear(28*28, 300, ReLU),
        new linear(300, 100, ReLU),
        new linear(100, 10, Softmax),
    };
    data set(mnist_reader);
    set.load({
        new random_rotate(30),
        new elastic_transform(5,4,1),
    });
    af::timer t = af::timer::start();
    SGD op(model.params, 2e-4);
    trainer tr = {
        .epochs = 20,
        .batch_size = 512,
        .optimizer = op,
        .loss_fn = categorical_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };
    model.summary();
    model.train(set, tr);

    tensor y_pred = model(set.test_x);
    float accu = categorical_accuracy(set.test_y, y_pred);
    printf("\nTotal time %.1fs, MNIST Test accuracy: %.4f\n", af::timer::stop(t), accu);
    return 0;
}
