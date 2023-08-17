#include "cformer.h"

static void txt_reader(struct data &d)
{
    std::string txt = read_file("data/shakespeare.txt");
    d.tokenizer = tokenizer(txt);
    auto v = d.tokenizer.encode(txt);
    array a(v.size(), v.data());
    d.train_x.init(a(af::seq(0, a.elements() - 2)));
    d.train_y.init(a(af::seq(1, a.elements() - 1)));
}

int main(int argc, char* argv[])
{
    data set(txt_reader, false);
    set.load();
    seqnet model {
        new Embedding(set.tokenizer.vocab.size(), 100),
        // new LSTM(100, 10, 2),
        // new Linear(10, set.tokenizer.vocab.size(), LogSoftmax),
    };
    af::timer t = af::timer::start();
    SGD op(model.params, 2e-4);
    trainer tr = {
        .epochs = 20,
        .batch_size = 512,
        .seq_len = 16,
        .optimizer = op,
        .loss_fn = log_softmax_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };
    model.summary();
    model.train(set, tr);

    // tensor y_pred = model(set.test_x);
    // float accu = categorical_accuracy(set.test_y, y_pred);
    // printf("\nTotal time %.1fs, MNIST Test accuracy: %.4f\n", af::timer::stop(t), accu);
    return 0;
}
