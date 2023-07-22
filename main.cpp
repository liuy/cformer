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
        //new random_rotate(45),
    });
    af::timer t = af::timer::start();
    SGD op(model.params, 1e-4, 0.8);
    trainer tr = {
        .epochs = 25,
        .batch_size = 100,
        .optimizer = op,
        .loss_fn = categorical_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };

    model.train(set, tr);

    tensor &y_pred = model(set.test_x);
    float accu = categorical_accuracy(set.test_y, y_pred);
    printf("\nTotal time %.1fs, MNIST Test accuracy: %.4f\n", af::timer::stop(t), accu);
    return 0;
}
