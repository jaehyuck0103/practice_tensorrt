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
    std::tuple<std::vector<cv::Rect>, std::vector<Instance>>
    updateDet(cv::Mat img, std::vector<Instance> &instVec, ArrayXXb &stackMask);

    void update(std::list<TrackerInput> &input);
    std::tuple<std::vector<int>, std::vector<int>> infer();

  private:
    std::list<TrackedInst> mTrackedInsts;
    std::unique_ptr<RegressInferAgent> mRegressAgent;
    std::unique_ptr<UNetInferAgent> mUNetAgent;
    std::unique_ptr<CNN3DInferAgent> mInferAgent;
};
