#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <third_party/nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

struct CalibParams {
    static const Eigen::Matrix4f RT;
    static const Eigen::Matrix3f K;
    static const Eigen::Matrix<float, 3, 4> P;
};
// clang-format off
const Eigen::Matrix4f CalibParams::RT = (
        Eigen::Matrix4f() << -0.005317, 0.003402, 0.999980, 1.624150,
                             -0.999920, -0.011526, -0.005277, 0.296660,
                             0.011508, -0.999928, 0.003463, 1.457150,
                             0, 0, 0, 1
        ).finished();
const Eigen::Matrix3f CalibParams::K = (
        Eigen::Matrix3f() << 819.162645, 0.000000, 640.000000,
                             0.000000, 819.162645, 240.000000,
                             0.000000, 0.000000, 1.000000
        ).finished();
const Eigen::Matrix<float, 3, 4> CalibParams::P = K * RT.inverse().topRows(3);
// clang-format on

class Instance {
  public:
    Instance(const json &inputRow) {
        // 0    classId in ascending order (car, truck(bus), pedestrian, bicycle(motorcycle))
        // 1    trackingId
        // 2~7  xyzlwh (unit: meter)
        // 8    heading angle (unit: radian)
        mClassId = inputRow[0].get<int>();
        mTrackId = inputRow[1].get<int>();
        mXyzCenter << inputRow[2].get<float>(), inputRow[3].get<float>(), inputRow[4].get<float>();
        mLwh << inputRow[5].get<float>(), inputRow[6].get<float>(), inputRow[7].get<float>();
        mYaw = inputRow[8].get<float>();

        setCorners();
    }

    bool box_in_image(int imgW, int imgH) const {

        Eigen::Matrix<float, 1, 8> cornersU = mCorners2D.row(0);
        Eigen::Matrix<float, 1, 8> cornersV = mCorners2D.row(1);
        Eigen::Matrix<float, 1, 8> cornersX = mCorners3D.row(0);

        Eigen::Matrix<bool, 1, 8> valid = cornersU.array() > 0 && cornersU.array() < imgW &&
                                          cornersV.array() > 0 && cornersV.array() < imgH &&
                                          cornersX.array() > 2;

        return valid.all();
    }

    void render_to_img(cv::Mat &img) {
        const std::vector<std::pair<int, int>> pairs{
            {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
            {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7},
        };

        for (const auto &p : pairs) {
            cv::Point point1{static_cast<int>(mCorners2D(0, p.first) + 0.5),
                             static_cast<int>(mCorners2D(1, p.first) + 0.5)};
            cv::Point point2{static_cast<int>(mCorners2D(0, p.second) + 0.5),
                             static_cast<int>(mCorners2D(1, p.second) + 0.5)};
            cv::line(img, point1, point2, cv::Scalar{0, 0, 255});
        }
    }

  private:
    int mClassId;
    int mTrackId;
    Eigen::Vector3f mXyzCenter;
    Eigen::Vector3f mLwh;
    float mYaw;

    Eigen::Matrix<float, 3, 8> mCorners3D;
    Eigen::Matrix<float, 2, 8> mCorners2D;

    void setCorners() {
        // clang-format off
        mCorners3D << 1, 1, 1, 1, -1, -1, -1, -1, 
                      1, -1, -1, 1, 1, -1, -1, 1,
                      1, 1, -1, -1, 1, 1, -1, -1;
        // clang-format on
        mCorners3D.array().colwise() *= (0.5 * mLwh.array());

        // Rotate and Translate
        const Eigen::Affine3f transform =
            Eigen::Translation3f(mXyzCenter) * Eigen::AngleAxisf(mYaw, Eigen::Vector3f(0, 0, 1));
        mCorners3D = transform * mCorners3D;

        // Project Corners
        mCorners2D = (CalibParams::P * mCorners3D.colwise().homogeneous()).colwise().hnormalized();
    }
};

int main() {
    // Read Json file
    std::ifstream ifs{"json.json"};
    json j = json::parse(ifs);
    ifs.close();

    for (const auto &eachFrame : j) {
        std::string imgFilePath = eachFrame["img_file"].get<std::string>();
        imgFilePath = "/mnt/EVO_4TB/VoSS/20200316-174732(20191213-125018_emul)/" + imgFilePath;
        cv::Mat img = cv::imread(imgFilePath);

        std::cout << imgFilePath << std::endl;

        std::vector<Instance> instVec;
        for (const auto &eachObj : eachFrame["objs"]) {
            // Simple filtering
            int classId = eachObj[0].get<int>();
            float centerX = eachObj[2].get<float>();
            float centerY = eachObj[3].get<float>();
            if (classId == 2) {
                continue;
            }
            if (centerX < 4 || abs(centerY) > 10) {
                continue;
            }

            // Generate Instance
            instVec.emplace_back(eachObj);
        }

        // Filtering
        instVec.erase(
            std::remove_if(instVec.begin(), instVec.end(),
                           [&img](auto x) { return !x.box_in_image(img.cols, img.rows); }),
            instVec.end());

        // Rendering
        for (auto &inst : instVec) {
            inst.render_to_img(img);
        }

        /*
                for (auto &eachInst : instVec) {
            eachInst.box_in_image(img.cols, img.rows);
            exit(0);
            }
        */

        cv::imshow("img", img);
        if (cv::waitKey() == 'q')
            break;
    }
}
