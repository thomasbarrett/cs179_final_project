#ifndef GEOM_TIMER_H
#define GEOM_TIMER_H

#include <chrono>

namespace geom {

class Timer {
private:
    std::chrono::steady_clock::time_point start_;
public:
    Timer() {
       start_ = std::chrono::steady_clock::now();
    };

    float duration() const {
        auto end = std::chrono::steady_clock::now();
        int ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        return ms_elapsed / 1000.0;
    }
};

};

#endif /* GEOM_TIMER_H */