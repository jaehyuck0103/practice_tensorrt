#include "SimpleTracker.hpp"
#include "infer-agents/RegressInferAgent.hpp"
#include "infer-agents/UNetInferAgent.hpp"
#include "instance.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <opencv2/core/eigen.hpp>

using json = nlohmann::json;

int main(int argc, char **argv) {
    std::vector<std::string> arguments(argv + 1, argv + argc);
    bool bImWrite = false;
    if (arguments.size() > 0 && arguments[0] == "--imwrite") {
        bImWrite = true;
    }

    // Read Json file
    std::ifstream ifs{"scripts/json.json"};
    json j = json::parse(ifs);
    ifs.close();

    // Initialize TRT Agents.
    const std::string homeDir = std::getenv("HOME");
    InferenceParams params;

    params.trtFilePath =
        homeDir + "/extern/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/tail_det.trt";
    RegressInferAgent regressAgent(params);

    params.trtFilePath =
        homeDir +
        "/extern/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/taillight_unet.trt";
    UNetInferAgent unetAgent(params);

    int frameIdx = 0;
    for (const auto &eachFrame : j) {
        std::string imgFilePath = eachFrame["img_file"].get<std::string>();
        imgFilePath = "/mnt/EVO_4TB/VoSS/20200316-174732(20191213-125018_emul)/" + imgFilePath;
        std::cout << imgFilePath << std::endl;
        cv::Mat img = cv::imread(imgFilePath);
        cv::Mat displayedImg = img.clone();

        std::vector<Instance> instVec;
        for (const auto &eachObj : eachFrame["objs"]) {
            // Simple filtering
            // int classId = eachObj[0].get<int>();
            float centerX = eachObj[2].get<float>();
            float centerY = eachObj[3].get<float>();
            // if (classId == 2) { continue; }
            if (centerX < 4 || centerX > 40 || abs(centerY) > 10) {
                continue;
            }

            // Generate Instance
            instVec.emplace_back(eachObj);
        }

        // image 내에 약간이라도 projection되는 instance만 남김.
        instVec.erase(
            std::remove_if(instVec.begin(), instVec.end(),
                           [&img](auto x) { return !x.isValidProjection(img.rows, img.cols); }),
            instVec.end());

        // Sorting instVec by distance
        std::sort(instVec.begin(), instVec.end(), [](const Instance &lhs, const Instance &rhs) {
            return lhs.dist() < rhs.dist();
        });

        // 가림이 없는 tail view를 가지는 instances 추출.
        // 가까이 있는 instance부터 stackMask에 projection 해나간다.
        std::vector<Instance> validTailInsts;
        MatrixXXb stackMask = MatrixXXb::Zero(img.rows, img.cols);
        for (const auto &eachInst : instVec) {
            if (eachInst.isTailInSight(img.rows, img.cols, stackMask)) {
                validTailInsts.push_back(eachInst);
            }
            const MatrixXXb instMask = eachInst.getMask(img.rows, img.cols, false);
            stackMask = stackMask || instMask;
        }

        // mask: eigen -> opencv
        cv::Mat displayedMask{img.rows, img.cols, CV_8UC1, cv::Scalar(0)};
        cv::eigen2cv(stackMask, displayedMask);
        displayedMask *= 255;
        cv::cvtColor(displayedMask, displayedMask, cv::COLOR_GRAY2BGR);

        // Render Boxes
        for (const auto &inst : instVec) {
            inst.renderToImg(displayedImg);
        }

        // tail crop image들을 모아서 tensorrt inference
        std::vector<cv::Rect> croppedRois;
        std::vector<cv::Mat> croppedImgs;
        for (auto &inst : validTailInsts) {
            cv::Rect roi = inst.getTailRect(img.rows, img.cols);
            croppedRois.push_back(roi);
            cv::Mat croppedImg = img(roi);
            cv::resize(croppedImg, croppedImg, cv::Size{224, 224});
            croppedImg.convertTo(croppedImg, CV_BGR2RGB);
            croppedImg.convertTo(croppedImg, CV_32FC3);
            croppedImg = ((croppedImg / 255.0f) - 0.5f) * 4.0f; // normalization
            croppedImgs.push_back(croppedImg);
        }
        std::vector<std::array<float, 4>> regressCoords = regressAgent.infer(croppedImgs);

        // Collect RegressedRois
        std::vector<cv::Rect> regressedRois;
        for (size_t i = 0; i < croppedRois.size(); ++i) {
            cv::Rect regressedRoi{
                static_cast<int>(croppedRois[i].x + regressCoords[i][0] * croppedRois[i].width),
                static_cast<int>(croppedRois[i].y + regressCoords[i][1] * croppedRois[i].height),
                static_cast<int>((regressCoords[i][2] - regressCoords[i][0]) *
                                 croppedRois[i].width),
                static_cast<int>((regressCoords[i][3] - regressCoords[i][1]) *
                                 croppedRois[i].height)};
            regressedRois.push_back(regressedRoi);
            img(regressedRoi).copyTo(displayedMask(regressedRoi)); // visualization to Mask
        }

        // Collect Regressed Imgs
        std::vector<cv::Mat> regressedImgs;
        for (size_t i = 0; i < regressedRois.size(); ++i) {
            cv::Mat regressedImg;
            cv::resize(img(regressedRois[i]), regressedImg, cv::Size{112, 112});
            regressedImg.convertTo(regressedImg, CV_BGR2RGB);
            regressedImg.convertTo(regressedImg, CV_32FC3);
            regressedImg = ((regressedImg / 255.0f) - 0.5f) * 4.0f; // normalization
            regressedImgs.push_back(regressedImg);
        }
        // unet inferece
        std::vector<std::vector<float>> encodedImgs = unetAgent.infer(regressedImgs);

        // Infer

        // Display
        if (bImWrite) {
            cv::imwrite("Debug/" + std::to_string(frameIdx) + "img.png", displayedImg);
            cv::imwrite("Debug/" + std::to_string(frameIdx) + "mask.png", displayedMask);
            frameIdx += 1;
        } else {
            cv::imshow("img_display", displayedImg);
            cv::imshow("mask_display", displayedMask);
            if (cv::waitKey() == 'q')
                break;
        }
    }
}
