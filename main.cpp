#include "cformer.h"

static void txt_reader(struct data &d)
{
    std::string txt = read_file("data/test.txt");
    auto v = d.tokenizer.encode(txt);
    array a = array(v.size(), v.data()).as(f32);
    d.train_x.init(a(af::seq(0, a.elements() - 2)));
    d.train_y.init(a(af::seq(1, a.elements() - 1)));
}

int main(int argc, char* argv[])
{
    data set(txt_reader, false);
    set.load();
    seqnet model {
        new Embedding(set.tokenizer.vocab.size(), 1000),
        new RNN(1000, 1000, 1),
        new Linear(1000, set.tokenizer.vocab.size(), LogSoftmax),
    };
    af::timer t = af::timer::start();
    Adam op(model.params, 0.001, 1e-4);
    trainer tr = {
        .epochs = 150,
        .batch_size = 128,
        .seq_len = 16,
        .optimizer = op,
        .loss_fn = log_softmax_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };
    model.summary();
    model.train(set, tr);

    tensor y_pred = model(set.train_x);
    tensor y_true(onehot(set.train_y.data, set.tokenizer.vocab.size()));
    float accu = categorical_accuracy(y_true, y_pred);
    printf("\nTotal time %.1fs, RNN Train accuracy: %.4f\n", af::timer::stop(t), accu);

    ((RNN *)model.layers[1])->reset_hidden_states();
    std::string prompt = "You";
    uint32_t i = set.tokenizer.token2idx[prompt];
    tensor x(array(1, &i).as(f32));

    printf("Text generation:\n");
    std::cout << prompt;
    for (int i = 0; i < 400; i++) {
        tensor y = model(x);
        uint32_t idx = argmax(y.data).as(u32).scalar<uint32_t>();
        std::cout << set.tokenizer.idx2token[idx];
        x.init(array(1, &idx).as(f32));
    }
    printf("\n");
    return 0;
}
