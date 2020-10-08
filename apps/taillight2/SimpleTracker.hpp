#pragma once
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <numeric>
#include <vector>

#include "infer-agents/CNN3DInferAgent.hpp"

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

    bool canInfered() const {
        // 1. Check first_detect
        size_t first_detect;
        for (first_detect = 0; first_detect < mbDetected.size(); ++first_detect) {
            if (mbDetected[first_detect]) {
                break;
            }
        }
        if (first_detect > 8) {
            return false;
        }
        // 2. Check sum_latest8
        const int sum_latest8 = std::accumulate(mbDetected.end() - 8, mbDetected.end(), 0);
        if (sum_latest8 < 6) {
            return false;
        }
        // 3. Check sum
        const int sum = std::accumulate(mbDetected.begin(), mbDetected.end(), 0);
        if (sum < 8) {
            return false;
        }

        return true;
    }

    void printDetected() const {
        std::cout << trackId() << "   ";
        for (const auto &elem : mbDetected) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> getConcatedFeats() const {
        const size_t inputSize = mEncodedImgs[0].size() * mEncodedImgs.size();

        std::vector<float> concat;
        concat.reserve(inputSize);
        for (const auto &elem : mEncodedImgs) {
            concat.insert(concat.end(), elem.begin(), elem.end());
        }

        if (concat.size() != 16 * 64 * 28 * 28) {
            std::cout << "Concat Size Error" << std::endl;
            exit(1);
        }

        return concat;
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
    SimpleTracker() {
        const std::string homeDir = std::getenv("HOME");
        InferenceParams params;

        params.trtFilePath =
            homeDir +
            "/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/taillight_3Dconv.trt";
        mInferAgent = std::make_unique<CNN3DInferAgent>(params);
    }

    void update(std::list<TrackerInput> &input);
    std::tuple<std::vector<int>, std::vector<int>> infer();

  private:
    std::list<TrackedInst> mTrackedInsts;
    std::unique_ptr<CNN3DInferAgent> mInferAgent;
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

std::tuple<std::vector<int>, std::vector<int>> SimpleTracker::infer() {
    std::list<std::vector<float>> inputFeats;
    std::vector<int> inferredTrackIds;
    for (const auto &elem : mTrackedInsts) {
        if (elem.canInfered()) {
            inputFeats.push_back(elem.getConcatedFeats());
            inferredTrackIds.push_back(elem.trackId());
        }
    }
    std::vector<int> inferredStates = mInferAgent->infer(inputFeats);

    if (inferredTrackIds.size() != inferredStates.size()) {
        std::cout << "inferredTrackIds and inferredStates should have same size" << std::endl;
        exit(1);
    }
    return {inferredTrackIds, inferredStates};
}
