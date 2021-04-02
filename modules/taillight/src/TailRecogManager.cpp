#include "infer-agents/CNN3DInferAgent.hpp"
#include "infer-agents/RegressInferAgent.hpp"
#include "infer-agents/UNetInferAgent.hpp"
#include "taillight/TailRecogManager.hpp"
#include <chrono>
namespace chrono = std::chrono;

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

std::map<int, cv::Rect>
TailRecogManager::updateDet(cv::Mat img, std::vector<Instance> &instVec, ArrayXXb &occMask) {
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
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
    // 가까이 있는 instance부터 occMask에 projection 해나간다.
    std::vector<Instance> validTailInsts;
    for (const auto &eachInst : instVec) {
        if (eachInst.isTailInSight(img.rows, img.cols, occMask)) {
            validTailInsts.push_back(eachInst);
        }
        auto [u_min, v_min, boxW, boxH] = eachInst.getBoundingRect(img.rows, img.cols);
        occMask.block(v_min, u_min, boxH, boxW) = true;
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
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_before1 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;

    t1 = chrono::high_resolution_clock::now();
    std::vector<std::array<float, 4>> regressCoords = mRegressAgent->infer(croppedImgs);
    t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_1 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;

    t1 = chrono::high_resolution_clock::now();
    // Collect RegressedRois
    std::vector<cv::Rect> regressedRois;
    for (size_t i = 0; i < regressCoords.size(); ++i) {
        cv::Rect regressedRoi{
            static_cast<int>(croppedRois[i].x + regressCoords[i][0] * croppedRois[i].width),
            static_cast<int>(croppedRois[i].y + regressCoords[i][1] * croppedRois[i].height),
            static_cast<int>((regressCoords[i][2] - regressCoords[i][0]) * croppedRois[i].width),
            static_cast<int>((regressCoords[i][3] - regressCoords[i][1]) * croppedRois[i].height),
        };
        regressedRois.push_back(regressedRoi);
    }

    // Collect Regressed Imgs
    std::vector<cv::Mat> regressedImgs;
    for (const auto &eachRoi : regressedRois) {
        cv::Mat regressedImg;
        cv::resize(img(eachRoi), regressedImg, cv::Size{UNetCfg::inW, UNetCfg::inH});

        cv::cvtColor(regressedImg, regressedImg, cv::COLOR_BGR2RGB);
        regressedImg.convertTo(regressedImg, CV_32FC3);
        regressedImg = regressedImg / 255.0f; // (0~255) -> (0~1)
        regressedImgs.push_back(regressedImg);
    }
    t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_1-2 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;
    // unet inferece
    t1 = chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> encodedImgs = mUNetAgent->infer(regressedImgs);
    t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_2 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;

    t1 = chrono::high_resolution_clock::now();
    // Tracker Update
    std::list<TrackerInput> trackerInputs;
    for (size_t i = 0; i < encodedImgs.size(); ++i) {
        trackerInputs.push_back(TrackerInput{validTailInsts[i].trackId(), encodedImgs[i]});
    }

    // Update tracked instances.
    for (auto &elem : mTrackedInsts) {
        elem.update(trackerInputs);
    }

    // Generate New instances
    for (const auto &elem : trackerInputs) {
        mTrackedInsts.emplace_back(elem);
    }

    // 최근 3프레임에 detect이 없으면 제거
    mTrackedInsts.remove_if([](const TrackedInst &inst) { return inst.shouldRemoved(); });

    // print
    for (const auto &elem : mTrackedInsts) {
        elem.printDetected();
    }

    // debugging용 return
    std::map<int, cv::Rect> trackId_to_regressedRoi;
    for (size_t i = 0; i < regressedRois.size(); ++i) {
        trackId_to_regressedRoi.emplace(validTailInsts[i].trackId(), regressedRois[i]);
    }
    t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_2-3 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;

    return trackId_to_regressedRoi;
}

std::map<int, int> TailRecogManager::infer() {
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> inputFeats;
    std::vector<int> inferredTrackIds;
    for (const auto &elem : mTrackedInsts) {
        if (elem.canInfered()) {
            inputFeats.push_back(elem.getConcatedFeats());
            inferredTrackIds.push_back(elem.trackId());
        }
    }
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_2-3 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;

    t1 = chrono::high_resolution_clock::now();
    std::vector<int> inferredStates = mInferAgent->infer(inputFeats);
    t2 = chrono::high_resolution_clock::now();
    std::cout << "processing_time_3 (micro sec): "
              << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << std::endl;

    if ((inferredTrackIds.size() != inferredStates.size()) &&
        (inferredTrackIds.size() <= CNN3DCfg::inB)) {
        std::cout << "inferredTrackIds and inferredStates should have same size" << std::endl;
        exit(1);
    }

    std::map<int, int> trackId_to_state;
    for (size_t i = 0; i < inferredStates.size(); ++i) {
        trackId_to_state.emplace(inferredTrackIds[i], inferredStates[i]);
    }

    return trackId_to_state;
}
