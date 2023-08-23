#pragma once
#include <iostream>

struct progress_bar {
    float progress{ 0.0 };
    float bar_width{ 50 };
    float max{ 100.0 };
    std::string prefix_text{ "" };
    std::string start{ "[" };
    std::string fill{ "■" };
    std::string remainder{ " " };
    std::string end{ "]" };
    std::string postfix_text{ "" };

    void set_progress(float value) {
        progress = value;
        print_progress();
    }

    void tick() {
        progress += 1;
        print_progress();
    }

private:
    void print_progress() {
        std::cout << prefix_text;
        std::cout << start;
        float pos = progress * bar_width / max;
        for (size_t i = 0; i < bar_width; ++i) {
            if (i <= pos)
                std::cout << fill;
            else
                std::cout << remainder;
        }
        std::cout << end;
        std::cout << " " << postfix_text << "\r";
        std::cout.flush();
        if (progress >= max) {
            progress = 0;
            std::cout << "\33[2K\r"; // erase line
        }
    }
};