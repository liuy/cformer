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
        new Embedding(set.tokenizer.vocab.size(), 500),
        new RNN(500, 500, 1),
        new Linear(500, set.tokenizer.vocab.size(), LogSoftmax),
    };
    af::timer t = af::timer::start();
    Adam op(model.params);
    trainer tr = {
        .epochs = 100,
        .batch_size = 16,
        .seq_len = 8,
        .optimizer = op,
        .loss_fn = log_softmax_cross_entropy,
        .metrics_fn = categorical_accuracy,
    };
    model.summary();
    model.train(set, tr);

    // std::string prompt = "First";
    // uint32_t i = set.tokenizer.token2idx[prompt];
    // array v = array(1, &i).as(f32);
    // tensor x(v);
    // array_shape(x.data);
    // tensor y = model(x);
    // uint32_t y_idx = argmax(y.data).as(u32).scalar<uint32_t>();
    // std::cout << set.tokenizer.idx2token[y_idx] << std::endl;
    // for (int i = 0; i < 100; i++) {
    // }
    // tensor y_pred = model(set.test_x);
    return 0;
}
