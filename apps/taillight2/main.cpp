#include "RegressInferAgent.hpp"
#include "instance.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <opencv2/core/eigen.hpp>

using json = nlohmann::json;

int main() {
    // Read Json file
    std::ifstream ifs{"scripts/json.json"};
    json j = json::parse(ifs);
    ifs.close();

    // Initialize TRT
    SampleParams params;
    params.onnxFilePath =
        "/home/jae/extern/Projects/ETRI_TailLightRecognition/scripts/onnx/Output/tail_det.onnx";
    params.inputTensorName = "Input";
    params.outputTensorName = "Output";

    RegressInferAgent regressAgent(params);
    regressAgent.build();

    for (const auto &eachFrame : j) {
        std::string imgFilePath = eachFrame["img_file"].get<std::string>();
        imgFilePath = "/mnt/EVO_4TB/VoSS/20200316-174732(20191213-125018_emul)/" + imgFilePath;
        std::cout << imgFilePath << std::endl;
        cv::Mat img = cv::imread(imgFilePath);
        cv::Mat displayImg = img.clone();

        std::vector<Instance> instVec;
        for (const auto &eachObj : eachFrame["objs"]) {
            // Simple filtering
            int classId = eachObj[0].get<int>();
            float centerX = eachObj[2].get<float>();
            float centerY = eachObj[3].get<float>();
            if (classId == 2) {
                continue;
            }
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

        // Sorting by distance
        std::sort(instVec.begin(), instVec.end(), [](const Instance &lhs, const Instance &rhs) {
            return lhs.dist() < rhs.dist();
        });

        // 가림이 없는 tail view를 가지는 instances 추출.
        std::vector<Instance> tailValidVec;
        MatrixXXb stackMask = MatrixXXb::Zero(img.rows, img.cols);
        for (const auto &eachInst : instVec) {
            if (eachInst.isTailInSight(img.rows, img.cols, stackMask)) {
                tailValidVec.push_back(eachInst);
            }
            const MatrixXXb instMask = eachInst.getMask(img.rows, img.cols, false);
            stackMask = stackMask || instMask;
        }

        // mask: eigen -> opencv
        cv::Mat displayMask{img.rows, img.cols, CV_8UC1, cv::Scalar(0)};
        cv::eigen2cv(stackMask, displayMask);
        displayMask *= 255;
        cv::cvtColor(displayMask, displayMask, cv::COLOR_GRAY2BGR);

        // Render Boxes
        for (const auto &inst : instVec) {
            inst.renderToImg(displayImg);
        }

        // tail crop image들을 모아서 tensorrt infer
        std::vector<cv::Rect> rois;
        std::vector<cv::Mat> croppedImgs;
        for (auto &inst : tailValidVec) {
            cv::Rect roi = inst.getTailRect(img.rows, img.cols);
            rois.push_back(roi);
            cv::Mat croppedImg = img(roi);
            cv::resize(croppedImg, croppedImg, cv::Size{224, 224});
            croppedImg.convertTo(croppedImg, CV_BGR2RGB);
            croppedImg.convertTo(croppedImg, CV_32FC3);
            croppedImg = ((croppedImg / 255.0f) - 0.5f) * 4.0f; // normalization
            croppedImgs.push_back(croppedImg);
        }
        std::vector<std::array<float, 4>> regressCoords = regressAgent.infer(croppedImgs);

        // visulaize regression result
        for (size_t i = 0; i < rois.size(); ++i) {
            cv::Rect regressedRoi{
                static_cast<int>(rois[i].x + regressCoords[i][0] * rois[i].width),
                static_cast<int>(rois[i].y + regressCoords[i][1] * rois[i].height),
                static_cast<int>((regressCoords[i][2] - regressCoords[i][0]) * rois[i].width),
                static_cast<int>((regressCoords[i][3] - regressCoords[i][1]) * rois[i].height)};
            img(regressedRoi).copyTo(displayMask(regressedRoi));
        }

        cv::imshow("img_display", displayImg);
        cv::imshow("mask_display", displayMask);
        if (cv::waitKey() == 'q')
            break;
    }
}
