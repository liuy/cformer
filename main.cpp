#include "cformer.h"

static tensor idx2tensor(uint32_t i)
{
    return tensor(array(1, &i).as(f32));
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
    uint32_t idx = logits_sample_next(x.data, 1, 0.0);
    std::cout << tok.idx2token[idx];
    x = idx2tensor(idx);
    for (int i = 0; i < n; i++) {
        tensor y = model(x);
        idx = logits_sample_next(y.data, 1, 0.0);
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
        new GPT_Embedding(set.tokenizer.vocab.size(), 192, 32),
        new Dropout(0.1),
        new GPT_Block(192, 4, 2, 0.1),
        new LayerNorm1d(192),
        new Linear(192, set.tokenizer.vocab.size(), None, true, xavier_normal),
    };
    af::timer t = af::timer::start();
    Adam op(model.params, 0.0005);
    trainer tr = {
        .epochs = 75,
        .batch_size = 2,
        .seq_len = 32,
        .optimizer = op,
        .loss_fn = logits_cross_entroy,
        .metrics_fn = categorical_accuracy,
    };

    model.summary();
    model.train(set, tr);
    printf("\nTotal training time %.1fs\n", af::timer::stop(t));

    generate(model, set.tokenizer, "King", 2000);

    return 0;
}
