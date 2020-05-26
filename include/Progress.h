#ifndef PROGRESS_H
#define PROGRESS_H

#include <iostream>

class ProgressBar {
public:
    void update(float progress) {
        static int last = -1;
         
        int barWidth = 70;
        int pos = (int) (barWidth * progress);
        if (pos == last) return;
        last = pos;
        
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << "%\r";
        std::cout.flush();
    }

    void finish() {
        int barWidth = 70;
        std::cout << "[";
        for (int i = 0; i < barWidth; ++i) {
            std::cout << "=";
        }
        std::cout << "] " << "100%\n";
        std::cout.flush();
    }
};

#endif