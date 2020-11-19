#include "taillight/TailRecogManager.hpp"
#include "infer-agents/CNN3DInferAgent.hpp"
#include "infer-agents/RegressInferAgent.hpp"
#include "infer-agents/UNetInferAgent.hpp"

TailRecogManager::TailRecogManager() {
    const std::string homeDir = std::getenv("HOME");
    InferenceParams params;

    params.trtFilePath =
        homeDir + "/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/tail_det.trt";
    mRegressAgent = std::make_unique<RegressInferAgent>(params);

    params.trtFilePath =
        homeDir + "/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/taillight_unet.trt";
    mUNetAgent = std::make_unique<UNetInferAgent>(params);

    params.trtFilePath =
        homeDir + "/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/taillight_3Dconv.trt";
    mInferAgent = std::make_unique<CNN3DInferAgent>(params);
}

TailRecogManager::~TailRecogManager() = default;

std::tuple<std::vector<cv::Rect>, std::vector<Instance>> TailRecogManager::updateDet(
    cv::Mat img,
    std::vector<Instance> &instVec,
    Eigen::ArrayXi &stackMask) {
    // image 내에 약간이라도 projection되는 instance만 남김.
    instVec.erase(
        std::remove_if(
            instVec.begin(),
            instVec.end(),
            [&img](auto x) { return !x.isValidProjection(img.rows, img.cols); }),
        instVec.end());

    // Sorting instVec by distance
    std::sort(instVec.begin(), instVec.end(), [](const Instance &lhs, const Instance &rhs) {
        return lhs.dist() < rhs.dist();
    });

    // 가림이 없는 tail view를 가지는 instances 추출.
    // 가까이 있는 instance부터 stackMask에 projection 해나간다.
    std::vector<Instance> validTailInsts;
    int tmpCnt = 0;
    for (const auto &eachInst : instVec) {
        ++tmpCnt;
        if (eachInst.isTailInSight(img.rows, img.cols, stackMask)) {
            validTailInsts.push_back(eachInst);
        }
        auto [u_min, u_max] = eachInst.getProjectionLR(img.cols);
        stackMask.segment(u_min, u_max - u_min + 1) = tmpCnt;
    }

    // tail crop image들을 모아서 tensorrt inference
    std::vector<cv::Rect> croppedRois;
    std::vector<cv::Mat> croppedImgs;
    for (auto &inst : validTailInsts) {
        auto [tailU, tailV, tailW, tailH] = inst.getTailRect(img.rows, img.cols, 0.5);
        cv::Rect roi{tailU, tailV, tailW, tailH};
        croppedRois.push_back(roi);
        cv::Mat croppedImg = img(roi);
        cv::resize(croppedImg, croppedImg, cv::Size{RegCfg::inW, RegCfg::inH});
        cv::cvtColor(croppedImg, croppedImg, cv::COLOR_BGR2RGB);
        croppedImg.convertTo(croppedImg, CV_32FC3);
        croppedImg = croppedImg / 255.0f; // (0~255) -> (0~1)
        croppedImgs.push_back(croppedImg);
    }
    std::vector<std::array<float, 4>> regressCoords = mRegressAgent->infer(croppedImgs);

    // Collect RegressedRois
    std::vector<cv::Rect> regressedRois;
    for (size_t i = 0; i < croppedRois.size(); ++i) {
        cv::Rect regressedRoi{
            static_cast<int>(croppedRois[i].x + regressCoords[i][0] * croppedRois[i].width),
            static_cast<int>(croppedRois[i].y + regressCoords[i][1] * croppedRois[i].height),
            static_cast<int>((regressCoords[i][2] - regressCoords[i][0]) * croppedRois[i].width),
            static_cast<int>((regressCoords[i][3] - regressCoords[i][1]) * croppedRois[i].height)};
        regressedRois.push_back(regressedRoi);
    }

    // Collect Regressed Imgs
    std::vector<cv::Mat> regressedImgs;
    for (size_t i = 0; i < regressedRois.size(); ++i) {
        cv::Mat regressedImg;
        cv::resize(img(regressedRois[i]), regressedImg, cv::Size{UNetCfg::inW, UNetCfg::inH});

        cv::cvtColor(regressedImg, regressedImg, cv::COLOR_BGR2RGB);
        regressedImg.convertTo(regressedImg, CV_32FC3);
        regressedImg = regressedImg / 255.0f; // (0~255) -> (0~1)
        regressedImgs.push_back(regressedImg);
    }
    // unet inferece
    std::vector<std::vector<float>> encodedImgs = mUNetAgent->infer(regressedImgs);

    // Tracker Update
    std::list<TrackerInput> trackerInputs;
    for (size_t i = 0; i < encodedImgs.size(); ++i) {
        trackerInputs.push_back(TrackerInput{validTailInsts[i].trackId(), encodedImgs[i]});
    }

    update(trackerInputs);

    return {regressedRois, validTailInsts};
}

void TailRecogManager::update(std::list<TrackerInput> &inputs) {
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

std::tuple<std::vector<int>, std::vector<int>> TailRecogManager::infer() {
    std::list<std::vector<float>> inputFeats;
    std::vector<int> inferredTrackIds;
    for (const auto &elem : mTrackedInsts) {
        if (elem.canInfered()) {
            inputFeats.push_back(elem.getConcatedFeats());
            inferredTrackIds.push_back(elem.trackId());
        }
    }
    std::vector<int> inferredStates = mInferAgent->infer(inputFeats);

    // track되는 대상이 infer 한계보다 많으면 문제 생길것 같은데?
    if (inferredTrackIds.size() != inferredStates.size()) {
        std::cout << "inferredTrackIds and inferredStates should have same size" << std::endl;
        exit(1);
    }
    return {inferredTrackIds, inferredStates};
}
