#include "instance.hpp"
#include <opencv2/core/eigen.hpp>

Instance::Instance(const json &inputRow) {

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

bool Instance::isAnyCornersInImage(int imgH, int imgW) const {

    const Eigen::Matrix<float, 1, 8> cornersU = mCorners2D.row(0);
    const Eigen::Matrix<float, 1, 8> cornersV = mCorners2D.row(1);

    const Eigen::Matrix<bool, 1, 8> valid = cornersU.array() > 0 && cornersU.array() < imgW &&
                                            cornersV.array() > 0 && cornersV.array() < imgH;

    return valid.any();
}

/*
 * corner가 camera 뒤에 위치하면 projection 하였을 때 문제 발생하므로 filterout 시키려함.
 * 하지만 bus와 같이 긴 instance이면, corner가 camera 뒤에 위치하면서도,
 * 다른 corner는 camera에 잡힐 수 있다.
 * 추후 해결 필요.
 */
bool Instance::isAllCornersFrontOfCam() const {
    const Eigen::Matrix<float, 1, 8> cornersCamZ = mCornersCam3D.row(2);
    const Eigen::Matrix<bool, 1, 8> valid = cornersCamZ.array() > 0;

    return valid.all();
}

bool Instance::isValidForProjection(int imgH, int imgW) const {
    return isAnyCornersInImage(imgH, imgW) && isAllCornersFrontOfCam();
}

void Instance::renderToImg(cv::Mat &img) const {

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

MatrixXXb Instance::getMask(int imgH, int imgW) const {

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
