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