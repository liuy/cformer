#include "cformer.h"

int main(int argc, char* argv[])
{
    seqnet model {
        new linear(28*28, 300, ReLU),
        new linear(300, 100, ReLU),
        new linear(100, 10, Softmax),
    };
    data set(mnist_reader);
    set.load();
    af::timer t = af::timer::start();
    SGD sgd(model.params, 0.0005, 0.8, 0);
    trainer tr = {
        .epochs = 20,
        .batch_size = 100,
        .optimizer = sgd,
        .loss_fn = categorical_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };

    model.train(set, tr);

    tensor &y_pred = model(set.test_x);
    float accu = categorical_accuracy(set.test_y, y_pred);
    printf("\nTotal time %.1fs, MNIST Test accuracy: %.4f\n", af::timer::stop(t), accu);
    return 0;
}
