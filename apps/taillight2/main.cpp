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

        // image 내에 projection 안되는 instances 제거
        instVec.erase(
            std::remove_if(instVec.begin(), instVec.end(),
                           [&img](auto x) { return !x.isValidForProjection(img.rows, img.cols); }),
            instVec.end());

        // Sorting by distance
        std::sort(instVec.begin(), instVec.end(), [](const Instance &lhs, const Instance &rhs) {
            return lhs.dist() < rhs.dist();
        });

        // Render Boxes
        for (const auto &inst : instVec) {
            inst.renderToImg(displayImg);
        }

        //
        MatrixXXb stackMask = MatrixXXb::Zero(img.rows, img.cols);
        for (const auto &eachInst : instVec) {
            MatrixXXb instMask = eachInst.getMask(img.rows, img.cols);
            stackMask = stackMask || instMask;
        }

        // eigen -> opencv
        cv::Mat displayMask{img.rows, img.cols, CV_8UC1, cv::Scalar(0)};
        cv::eigen2cv(stackMask, displayMask);
        displayMask *= 255;

        cv::imshow("img_display", displayImg);
        cv::imshow("mask_display", displayMask);
        if (cv::waitKey() == 'q')
            break;
    }
}
