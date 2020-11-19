#include "taillight/TailRecogManager.hpp"
#include "taillight/instance.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <toml.hpp>
#include <vector>

#include <opencv2/core/eigen.hpp>

using json = nlohmann::json;
namespace chrono = std::chrono;

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

    // Read config file
    auto data = toml::parse("./config.toml");
    const std::array<float, 16> RT_vals = toml::find<std::array<float, 16>>(data, "calib", "RT");
    const std::array<float, 16> RL_vals = toml::find<std::array<float, 16>>(data, "calib", "RL");
    const std::array<float, 9> K_vals = toml::find<std::array<float, 9>>(data, "calib", "K");

    const CalibParams calib_params{RT_vals, RL_vals, K_vals};
    calib_params.printParams();

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
            std::array<float, 3> xyz{
                eachObj[2].get<float>(),
                eachObj[3].get<float>(),
                eachObj[4].get<float>(),
            };
            std::array<float, 3> lwh{
                eachObj[5].get<float>(),
                eachObj[6].get<float>(),
                eachObj[7].get<float>(),
            };
            float yaw = eachObj[8].get<float>();

            // Simple filtering
            // if (classId == 2) { continue; }
            if (xyz[0] < 4 || xyz[0] > 40 || abs(xyz[1]) > 10) {
                continue;
            }

            // Generate Instance
            instVec.emplace_back(classId, trackId, xyz, lwh, yaw, calib_params);
        }

        // -------------------------
        // Run Manager
        // -------------------------
        chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
        ArrayXXb occMask = ArrayXXb::Zero(img.rows, img.cols);

        std::map<int, cv::Rect> trackId_to_regressedRoi =
            tailRecogManager.updateDet(img, instVec, occMask);
        std::map<int, int> trackId_to_state = tailRecogManager.infer();
        chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        std::cout << "processing_time (micro sec): " << duration << std::endl;

        // ------------------------
        // Write results to json
        // ------------------------
        json jsonInferStates = json::object();
        for (const auto &[id, state] : trackId_to_state) {
            jsonInferStates[std::to_string(id)] = state;
        }

        json jsonRois = json::object();
        for (const auto &[id, roi] : trackId_to_regressedRoi) {
            jsonRois[std::to_string(id)] = {
                roi.x,
                roi.y,
                roi.width,
                roi.height,
            };
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
        cv::Mat displayedMask;
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> displayedMaskEigen = occMask;
        cv::eigen2cv(displayedMaskEigen, displayedMask);
        displayedMask *= 255;
        cv::cvtColor(displayedMask, displayedMask, cv::COLOR_GRAY2BGR);

        // Visualize regressedTails to Mask
        for (const auto &[_, roi] : trackId_to_regressedRoi) {
            img(roi).copyTo(displayedMask(roi));
        }

        // Display
        if (bImWrite) {
            cv::imwrite("Debug/" + std::to_string(frameIdx) + "img.png", displayedImg);
            cv::imwrite("Debug/" + std::to_string(frameIdx) + "mask.png", displayedMask);
        } else {
            // 0.7초 이전 state이지만 출력.
            for (const auto &[trackId, state] : trackId_to_state) {
                std::string state_str = STATES.at(state);
                cv::Rect roi = trackId_to_regressedRoi[trackId];
                cv::putText(
                    displayedMask,
                    state_str,
                    {roi.x, roi.y},
                    cv::FONT_HERSHEY_PLAIN,
                    1,
                    {0, 0, 255},
                    2);
                std::cout << "drawing " << trackId << std::endl;
            }
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
