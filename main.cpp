#include "cformer.h"

static tensor idx2tensor(uint32_t i)
{
    return tensor(array(1, &i).as(f32));
}

static tensor ids2tensor(const std::vector<uint32_t> &ids)
{
    return tensor(array(ids.size(), ids.data()).as(f32));
}

template <typename T>
std::vector<T> vector_tail(const std::vector<T>& vec, int n)
{
    if (n >= vec.size()) {
        return vec;
    }
    std::vector<T> result(vec.end() - n, vec.end());
    return result;
}

static void generate(seqnet &model, tokenizer &tok, const std::string &prompt, int n)
{
    std::vector<uint32_t> ids = tok.encode_char(prompt);
    tensor x = ids2tensor(ids);

    printf("\nText generation:\n");
    std::cout << prompt;
    for (int i = 0; i < n; i++) {
        tensor y = model(x);
        // pass a context window of at most seq_len tokens
        uint32_t idx = logits_sample_next(y.data.row(-1), 1, 0.0);
        std::cout << tok.idx2token[idx];
        ids.push_back(idx);
        ids = vector_tail(ids, 64);
        x = ids2tensor(ids);
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
        new GPT_Embedding(set.tokenizer.vocab.size(), 768, 64),
        new Dropout(0.0),
        new GPT_Block(768, 16, 1, 0.0),
        new LayerNorm1d(768),
        new Linear(768, set.tokenizer.vocab.size(), None, true, xavier_normal),
    };
    af::timer t = af::timer::start();
    Adam op(model.params, 0.0005);
    trainer tr = {
        .epochs = 50,
        .batch_size = 256,
        .seq_len = 64,
        .optimizer = op,
        .loss_fn = logits_cross_entroy,
        .metrics_fn = categorical_accuracy,
    };

    model.summary();
    model.train(set, tr);
    printf("\nTotal training time %.1fs\n", af::timer::stop(t));

    generate(model, set.tokenizer, "Before we proceed any further, hear me speak.", 2000);

    return 0;
}
