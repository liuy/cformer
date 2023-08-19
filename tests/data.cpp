#include "../cformer.h"
#include <gtest/gtest.h>

TEST(Data, tokenizer)
{
    std::string s = "This is a test sentence, really.\nHello world!\n";

    tokenizer t;
    std::vector<uint32_t> encoded = t.encode(s);
    for (int i = 0; i < encoded.size(); i++)
        std::cout << encoded[i] << "(" +  t.idx2token[encoded[i]] + ")" << " ";
    std::cout << std::endl;
    std::string decoded = t.decode(encoded);
    std::cout << decoded;
    EXPECT_EQ(s, decoded);
}