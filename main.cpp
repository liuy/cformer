#include "cformer.h"

static tensor idx2tensor(uint32_t i)
{
    return tensor(array(1, &i).as(f32));
}

static array softmax(const array &a)
{
    array e = af::exp(a);
    return e / bsum(e, 1);
}

static uint32_t tensor2idx(const tensor &t)
{
    return argmax(softmax(t.data)).as(u32).scalar<uint32_t>();
}

static void generate(seqnet &model, tokenizer &tok, const std::string &prompt, int n)
{
    tensor x;
    for (auto &c : prompt) {
        std::string ch(1, c);
        uint32_t i = tok.token2idx[ch];
        tensor t = idx2tensor(i);
        x = model(t);
    }

    printf("\nText generation:\n");
    std::cout << prompt;
    uint32_t idx = tensor2idx(x);
    std::cout << tok.idx2token[idx];
    x = idx2tensor(idx);
    for (int i = 0; i < n; i++) {
        tensor y = model(x);
        idx = tensor2idx(y);
        std::cout << tok.idx2token[idx];
        x = idx2tensor(idx);
    }
    printf("\n");
    model.reset_hidden_states();
}

static void txt_reader(struct data &d)
{
    std::string txt = read_file("data/test.txt");
    auto v = d.tokenizer.encode_char(txt);
    array a = array(v.size(), v.data()).as(f32);
    d.train_x.init(a(af::seq(0, a.elements() - 2)));
    d.train_y.init(a(af::seq(1, a.elements() - 1)));
}

int main(int argc, char* argv[])
{
    data set(txt_reader, false);
    set.load();
    seqnet model {
        new Embedding(set.tokenizer.vocab.size(), 256),
        new RNN(256, 512, 1, LSTM),
        new Linear(512, 256, ReLU),
        new Linear(256, set.tokenizer.vocab.size()),
    };
    af::timer t = af::timer::start();
    Adam op(model.params, 0.001, 1e-4);
    trainer tr = {
        .epochs = 100,
        .batch_size = 128,
        .seq_len = 32,
        .optimizer = op,
        .loss_fn = logits_cross_entroy,
        .metrics_fn = categorical_accuracy,
    };

    model.summary();
    model.train(set, tr);
    printf("\nTotal training time %.1fs\n", af::timer::stop(t));

    generate(model, set.tokenizer, "Dursley woke up", 2000);

    return 0;
}
