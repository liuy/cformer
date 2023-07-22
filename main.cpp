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
    //af::timer t = af::timer::start();
    SGD op(model.params, 5e-4, 0.8);
    trainer tr = {
        .epochs = 28,
        .batch_size = 100,
        .optimizer = op,
        .loss_fn = categorical_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };

    af::timer t = af::timer::start();
    std::map<int, int> hist;
    for (int n = 0; n < 100000; ++n)
        ++hist[random(-10,10)];

    for (auto p : hist) {
        std::cout << std::fixed << std::setprecision(1) << std::setw(2)
                  << p.first << ' ' << std::string(p.second/200, '*') << '\n';
    }
    printf("\n Total time %.8fs\n", af::timer::stop(t));
    // model.train(set, tr);

    // tensor &y_pred = model(set.test_x);
    // float accu = categorical_accuracy(set.test_y, y_pred);
    // printf("\nTotal time %.1fs, MNIST Test accuracy: %.4f\n", af::timer::stop(t), accu);
    return 0;
}
