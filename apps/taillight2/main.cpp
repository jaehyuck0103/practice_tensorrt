#include "TailRecogManager.hpp"
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

    // Manager
    TailRecogManager tailRecogManager;

    // Result json
    json jsonResult = json::array();

    int frameIdx = 0;
    for (const auto &eachFrame : j) {
        std::string imgFilePath = eachFrame["img_file"].get<std::string>();
        imgFilePath = "/mnt/SATA01/VoSS/20200316-174732(20191213-125018_emul)/" + imgFilePath;
        std::cout << frameIdx << ": " << imgFilePath << std::endl;
        cv::Mat img = cv::imread(imgFilePath);
        cv::Mat displayedImg = img.clone();

        std::vector<Instance> instVec;
        for (const auto &eachObj : eachFrame["objs"]) {
            // 0    classId in ascending order (car, truck(bus), pedestrian, bicycle(motorcycle))
            // 1    trackingId
            // 2~7  xyzlwh (unit: meter)
            // 8    heading angle (unit: radian)
            int classId = eachObj[0].get<int>();
            int trackId = eachObj[1].get<int>();
            std::array<float, 3> xyz{eachObj[2].get<float>(), eachObj[3].get<float>(),
                                     eachObj[4].get<float>()};
            std::array<float, 3> lwh{eachObj[5].get<float>(), eachObj[6].get<float>(),
                                     eachObj[7].get<float>()};
            float yaw = eachObj[8].get<float>();

            // Simple filtering
            // if (classId == 2) { continue; }
            if (xyz[0] < 4 || xyz[0] > 40 || abs(xyz[1]) > 10) {
                continue;
            }

            // Generate Instance
            instVec.emplace_back(classId, trackId, xyz, lwh, yaw);
        }

        // -------------------------
        // Run Manager
        // -------------------------
        MatrixXXb stackMask = MatrixXXb::Zero(img.rows, img.cols);
        auto [regressedRois, validTailInsts] = tailRecogManager.updateDet(img, instVec, stackMask);
        auto [inferredTrackIds, inferredStates] = tailRecogManager.infer();

        // ------------------------
        // Write results to json
        // ------------------------
        json jsonInferStates = json::object();
        for (size_t i = 0; i < inferredTrackIds.size(); ++i) {
            jsonInferStates[std::to_string(inferredTrackIds.at(i))] = inferredStates.at(i);
        }

        json jsonRois = json::object();
        for (size_t i = 0; i < regressedRois.size(); ++i) {
            jsonRois[std::to_string(validTailInsts[i].trackId())] = {
                regressedRois[i].x, regressedRois[i].y, regressedRois[i].width,
                regressedRois[i].height};
        }
        jsonResult.push_back({{"result", jsonInferStates}, {"bbox", jsonRois}});

        // -------------------------
        // Display
        // -------------------------
        // Render Boxes
        for (const auto &inst : instVec) {
            inst.renderToImg(displayedImg);
        }

        // Mask: eigen -> opencv
        cv::Mat displayedMask{img.rows, img.cols, CV_8UC1, cv::Scalar(0)};
        cv::eigen2cv(stackMask, displayedMask);
        displayedMask *= 255;
        cv::cvtColor(displayedMask, displayedMask, cv::COLOR_GRAY2BGR);

        // Visualization to Mask
        for (const auto &roi : regressedRois) {
            img(roi).copyTo(displayedMask(roi));
        }

        // Display
        if (bImWrite) {
            cv::imwrite("Debug/" + std::to_string(frameIdx) + "img.png", displayedImg);
            cv::imwrite("Debug/" + std::to_string(frameIdx) + "mask.png", displayedMask);
        } else {
            cv::imshow("img_display", displayedImg);
            cv::imshow("mask_display", displayedMask);
            if (cv::waitKey() == 'q')
                break;
        }
        frameIdx += 1;
    }

    std::ofstream ofs{"Debug/result.json"};
    ofs << std::setw(4) << jsonResult << std::endl;
    ofs.close();
}
