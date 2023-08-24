#pragma once
#include <iostream>
#include <atomic>

struct progress_bar {
    int progress{ 0 };
    float bar_width{ 50 };
    float max{ 100.0 };
    bool show_percentage{ false };
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
        if (show_percentage)
            std::cout << " " << int(progress / max * 100.0) << "%";
        else
            std::cout << " " << progress << "/" << max;
        std::cout << " " << postfix_text << "\r";
        std::cout.flush();
        if (progress >= max) {
            progress = 0;
            std::cout << "\33[2K\r"; // erase line
        }
    }
};