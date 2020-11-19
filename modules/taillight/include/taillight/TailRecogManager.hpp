#pragma once
#include "TrackedInst.hpp"
#include "instance.hpp"

class RegressInferAgent;
class UNetInferAgent;
class CNN3DInferAgent;

class TailRecogManager {

  public:
    TailRecogManager();
    ~TailRecogManager();
    std::map<int, cv::Rect>
    updateDet(cv::Mat img, std::vector<Instance> &instVec, ArrayXXb &occMask);

    std::map<int, int> infer();

  private:
    std::list<TrackedInst> mTrackedInsts;
    std::unique_ptr<RegressInferAgent> mRegressAgent;
    std::unique_ptr<UNetInferAgent> mUNetAgent;
    std::unique_ptr<CNN3DInferAgent> mInferAgent;
};
