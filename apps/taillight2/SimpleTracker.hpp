#pragma once
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <vector>

struct TrackerInput {
    int trackId;
    std::vector<float> encodedTail;
};

class TrackedInst {
  public:
    TrackedInst(const TrackerInput &input) : mTrackId(input.trackId) {
        mEncodedImgs =
            std::deque<std::vector<float>>(kLenSeq - 1, std::vector<float>(kEncodedSize, 0.0f));
        mEncodedImgs.push_back(input.encodedTail);

        mbDetected = std::deque<bool>(kLenSeq - 1, false);
        mbDetected.push_back(true);
    }

    void update(std::list<TrackerInput> &inputs) {
        mEncodedImgs.pop_front();
        mbDetected.pop_front();

        // inputs 중에 matched trackId가 있는 경우 early return
        // 그리고 inputs에서 해당 elem 제거해줌.
        for (auto it = inputs.begin(); it != inputs.end(); ++it) {
            if (it->trackId == mTrackId) {
                // 이전 프레임이 false였다면 copy해줌.
                if (mbDetected.back() == false) {
                    mEncodedImgs.back() = it->encodedTail;
                }
                mEncodedImgs.push_back(it->encodedTail);
                mbDetected.push_back(true);

                inputs.erase(it);
                return;
            }
        }

        // inputs 중에 matched trackId가 없는 경우
        // 이전 프레임에서 복사해오던지, dummy 집어 넣음.
        if (mbDetected.back() == true) {
            mEncodedImgs.push_back(mEncodedImgs.back());
        } else {
            mEncodedImgs.emplace_back(kEncodedSize, 0.0f);
        }
        mbDetected.push_back(false);
        return;
    }

    bool shouldRemoved() const {
        int sum = std::accumulate(mbDetected.end() - 3, mbDetected.end(), 0);
        return sum == 0 ? true : false;
    }

    void printDetected() const {
        std::cout << trackId() << "   ";
        for (const auto &elem : mbDetected) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    // getters, setters
    int trackId() const { return mTrackId; }

  private:
    std::deque<std::vector<float>> mEncodedImgs;
    std::deque<bool> mbDetected;
    const int mTrackId;
    static constexpr int kLenSeq = 16;
    static constexpr int kEncodedSize = 64 * 28 * 28;
};

class SimpleTracker {

  public:
    // SimpleTracker() {}

    void update(std::list<TrackerInput> &input);

  private:
    std::list<TrackedInst> mTrackedInsts;
};

void SimpleTracker::update(std::list<TrackerInput> &inputs) {
    // Update tracked instances.
    for (auto &elem : mTrackedInsts) {
        elem.update(inputs);
    }

    // Generate New instances
    for (const auto &elem : inputs) {
        mTrackedInsts.emplace_back(elem);
    }

    // 최근 3프레임에 detect이 없으면 제거
    mTrackedInsts.remove_if([](const TrackedInst &inst) { return inst.shouldRemoved(); });

    // print
    for (const auto &elem : mTrackedInsts) {
        elem.printDetected();
    }
}
