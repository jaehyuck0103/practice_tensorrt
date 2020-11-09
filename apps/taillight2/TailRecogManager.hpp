#pragma once
#include "TrackedInst.hpp"
#include "infer-agents/CNN3DInferAgent.hpp"
#include "infer-agents/RegressInferAgent.hpp"
#include "infer-agents/UNetInferAgent.hpp"
#include "instance.hpp"

class TailRecogManager {

  public:
    TailRecogManager();
    std::tuple<std::vector<cv::Rect>, std::vector<Instance>>
    updateDet(cv::Mat img, std::vector<Instance> &instVec, MatrixXXb &stackMask);

    void update(std::list<TrackerInput> &input);
    std::tuple<std::vector<int>, std::vector<int>> infer();

  private:
    std::list<TrackedInst> mTrackedInsts;
    std::unique_ptr<RegressInferAgent> mRegressAgent;
    std::unique_ptr<UNetInferAgent> mUNetAgent;
    std::unique_ptr<CNN3DInferAgent> mInferAgent;
};
