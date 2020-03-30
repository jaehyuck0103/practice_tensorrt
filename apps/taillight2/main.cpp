#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXXb;

struct CalibParams {
    static const Eigen::Matrix4f RT;
    static const Eigen::Matrix4f RTinv;
    static const Eigen::Matrix3f K;
};
// clang-format off
const Eigen::Matrix4f CalibParams::RT = (
        Eigen::Matrix4f() << -0.005317, 0.003402, 0.999980, 1.624150,
                             -0.999920, -0.011526, -0.005277, 0.296660,
                             0.011508, -0.999928, 0.003463, 1.457150,
                             0, 0, 0, 1
        ).finished();
const Eigen::Matrix4f CalibParams::RTinv = RT.inverse();
const Eigen::Matrix3f CalibParams::K = (
        Eigen::Matrix3f() << 819.162645, 0.000000, 640.000000,
                             0.000000, 819.162645, 240.000000,
                             0.000000, 0.000000, 1.000000
        ).finished();
// clang-format on

// angleDiff (-pi, pi)
float angleDiff(float toAngle, float fromAngle) {
    return remainder((toAngle - fromAngle), 2 * M_PI);
}

class Instance {
  public:
    Instance(const json &inputRow) {

        /* ---------------------*
         * Set by input
         * ---------------------*/
        // 0    classId in ascending order (car, truck(bus), pedestrian, bicycle(motorcycle))
        // 1    trackingId
        // 2~7  xyzlwh (unit: meter)
        // 8    heading angle (unit: radian)
        mClassId = inputRow[0].get<int>();
        mTrackId = inputRow[1].get<int>();
        mXyzCenter << inputRow[2].get<float>(), inputRow[3].get<float>(), inputRow[4].get<float>();
        mLwh << inputRow[5].get<float>(), inputRow[6].get<float>(), inputRow[7].get<float>();
        mYaw = inputRow[8].get<float>();

        /* ---------------------*
         * Set corners (3D, 2D)
         * ---------------------*/
        // clang-format off
        mCorners3D << -1, -1, -1, -1, 1, 1, 1, 1, 
                      -1, -1, 1, 1, -1, -1, 1, 1,
                      -1, 1, -1, 1, -1, 1, -1, 1;
        // clang-format on
        mCorners3D.array().colwise() *= (0.5 * mLwh.array());

        // Rotate and Translate
        const Eigen::Affine3f transform =
            Eigen::Translation3f(mXyzCenter) * Eigen::AngleAxisf(mYaw, Eigen::Vector3f(0, 0, 1));
        mCorners3D = transform * mCorners3D;

        // camera coordinates
        mCornersCam3D =
            (CalibParams::RTinv * mCorners3D.colwise().homogeneous()).colwise().hnormalized();

        // Project Corners to image
        mCorners2D = (CalibParams::K * mCornersCam3D).colwise().hnormalized();

        /* ----------------------------------*
         * Minimun distance to box (roughly)
         * ----------------------------------*/
        mDist = mCorners3D.topRows(2).colwise().norm().minCoeff();

        /* -----------------------------------------------------------------*
         * mYawDiff : 차량 뒷면을 바라보는 angle과 차량 yaw angle 간의 차이
         * -----------------------------------------------------------------*/
        const Eigen::Vector3f rearCenter = mCornersCam3D.leftCols(4).rowwise().mean();
        const float viewAngleToRear = atan2(rearCenter(2), rearCenter(0));
        mYawDiff = abs(angleDiff(viewAngleToRear, mYaw));
    }

    bool isAnyCornersInImage(int imgH, int imgW) const {

        const Eigen::Matrix<float, 1, 8> cornersU = mCorners2D.row(0);
        const Eigen::Matrix<float, 1, 8> cornersV = mCorners2D.row(1);

        const Eigen::Matrix<bool, 1, 8> valid = cornersU.array() > 0 && cornersU.array() < imgW &&
                                                cornersV.array() > 0 && cornersV.array() < imgH;

        return valid.any();
    }

    /*
     * corner가 camera 뒤에 위치하면 projection 하였을 때 문제 발생.
     * 하지만 bus와 같이 긴 object이면, corner가 camera 뒤에 위치하면서도,
     * 다른 corner는 camera에 잡힐 수 있다.
     * 추후 해결 필요.
     */
    bool isAllCornersFrontOfCam() const {
        const Eigen::Matrix<float, 1, 8> cornersCamZ = mCornersCam3D.row(2);
        const Eigen::Matrix<bool, 1, 8> valid = cornersCamZ.array() > 0;

        return valid.all();
    }

    bool isValidForProjection(int imgH, int imgW) const {
        return isAnyCornersInImage(imgH, imgW) && isAllCornersFrontOfCam();
    }

    void renderToImg(cv::Mat &img) const {

        auto renderPairs = [&img, this](const std::vector<std::pair<int, int>> &pairs,
                                        cv::Scalar color) {
            for (const auto &p : pairs) {
                cv::Point point1{static_cast<int>(this->mCorners2D(0, p.first) + 0.5),
                                 static_cast<int>(this->mCorners2D(1, p.first) + 0.5)};
                cv::Point point2{static_cast<int>(this->mCorners2D(0, p.second) + 0.5),
                                 static_cast<int>(this->mCorners2D(1, p.second) + 0.5)};
                cv::line(img, point1, point2, color);
            }
        };

        // Render 3D Box
        const std::vector<std::pair<int, int>> boxPairs{
            {0, 1}, {0, 2}, {0, 4}, {1, 3}, {1, 5}, {2, 3},
            {2, 6}, {3, 7}, {4, 5}, {4, 6}, {5, 7}, {6, 7},
        };

        renderPairs(boxPairs, cv::Scalar{0, 0, 255});

        // Render front box
        const std::vector<std::pair<int, int>> frontPairs{
            {4, 5},
            {4, 6},
            {5, 7},
            {6, 7},
        };

        renderPairs(frontPairs, cv::Scalar{255, 0, 0});
    }

    MatrixXXb getMask(int imgH, int imgW) const {

        cv::Mat mask{imgH, imgW, CV_8UC1, cv::Scalar(0)};

        std::vector<cv::Point> points;
        for (int c = 0; c < mCorners2D.cols(); ++c) {
            points.emplace_back(static_cast<int>(mCorners2D(0, c) + 0.5),
                                static_cast<int>(mCorners2D(1, c) + 0.5));
        }
        std::vector<cv::Point> hull;
        cv::convexHull(points, hull);
        cv::fillConvexPoly(mask, hull, cv::Scalar{1});

        // cv -> eigen
        MatrixXXb maskEigen;
        cv::cv2eigen(mask, maskEigen);
        return maskEigen;
    }

    // getter, setter
    float dist() const { return mDist; }

  private:
    int mClassId;
    int mTrackId;
    Eigen::Vector3f mXyzCenter;
    Eigen::Vector3f mLwh;
    float mYaw;

    Eigen::Matrix<float, 3, 8> mCorners3D;    // vehicle coordinate
    Eigen::Matrix<float, 3, 8> mCornersCam3D; // camera coordinate
    Eigen::Matrix<float, 2, 8> mCorners2D;    // image coordinate

    float mDist;
    float mYawDiff;
};

int main() {
    // Read Json file
    std::ifstream ifs{"json.json"};
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
