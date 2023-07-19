#include "cformer.h"

int main(int argc, char* argv[])
{
    seq_net model {
        new linear(28*28, 300, ReLU),
        //new linear(256, 128, ReLU),
        new linear(300, 10, Softmax),
    };
    data set(mnist_reader);
    set.load();
    af::timer t = af::timer::start();

    model.train(set, 0.0005, 100, 20);

    tensor &y_pred = model(set.test_x);
    float accu = af::sum<float>(argmax(set.test_y.data) == argmax(y_pred.data)) / y_pred.data.dims(0);
    printf("\nTime used %.1fs, MNIST Test accuracy: %.4f\n", af::timer::stop(t), accu);
    return 0;
}
