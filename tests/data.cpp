#include "../cformer.h"
#include <gtest/gtest.h>

TEST(Data, tokenizer)
{
    std::string s = "This is a test sentence, really.\nHello world!\n";

    tokenizer t;
    std::vector<uint32_t> encoded = t.encode_word(s);
    for (int i = 0; i < encoded.size(); i++)
        std::cout << encoded[i] << "(" +  t.idx2token[encoded[i]] + ")" << " ";
    std::cout << std::endl;
    std::string decoded = t.decode(encoded);
    std::cout << decoded;
    EXPECT_EQ(s, decoded);

    tokenizer t2;
    std::vector<uint32_t> encoded2 = t2.encode_char(s);
    for (int i = 0; i < encoded2.size(); i++)
        std::cout << encoded2[i] << "(" +  t2.idx2token[encoded2[i]] + ")" << " ";
    std::cout << std::endl;
    std::string decoded2 = t2.decode(encoded2);
    std::cout << decoded2;
    EXPECT_EQ(s, decoded2);
}

TEST(Data, logits_sample_next)
{
    std::map<int, int> hist;
    array logits({1, 10}, {0.2990f, -0.2473f, 0.1842f, -1.2897f, -0.3420f, \
                            0.8222f, -0.6371f, -2.1130f, 0.9659f, -1.0080f});

    uint32_t idx = logits_sample_next(logits, 1);
    EXPECT_EQ(idx, 8);

    for (int i = 0; i < 10000; i++) {
        idx = logits_sample_next(logits, 5, 0.7);
        ++hist[idx];
    }

    for (auto p : hist)
        std::cout << p.first << ": " << p.second << '\n';

    EXPECT_EQ(hist.size(), 3);
    EXPECT_GE(hist[0], 2000);
    EXPECT_LE(hist[0], 2400);
    EXPECT_GE(hist[5], 3400);
    EXPECT_LE(hist[5], 3800);
    EXPECT_GE(hist[8], 4000);
    EXPECT_LE(hist[8], 4400);
}