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

    /* ----*
     * ETC
     * ----*/
    mDisplayStr += std::to_string(mClassId) + " ";
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

bool Instance::isValidProjection(int imgH, int imgW) const {
    return isAnyCornersInImage(imgH, imgW) && isAllCornersFrontOfCam();
}

bool Instance::isCar() const {
    if (mClassId == 0 || mClassId == 1) {
        return true;
    } else {
        return false;
    }
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

    // Render display string
    const cv::Point strPos{static_cast<int>(mCorners2D(0, 0) + 0.5),
                           static_cast<int>(mCorners2D(1, 0) + 0.5)};
    cv::putText(img, mDisplayStr, strPos, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);
}

bool Instance::isTailInSight(int imgH, int imgW, const MatrixXXb &stackMask) const {
    // 1. 자동차 맞는지
    if (!isCar())
        return false;

    // 2. tail과 view의 각도
    // yawDiff : 차량 뒷면을 바라보는 angle과 차량 yaw angle 간의 차이
    const Eigen::Vector3f tailCenter = mCornersCam3D.leftCols(4).rowwise().mean();
    const float viewAngleToTail = atan2(-tailCenter(0), tailCenter(2));
    const float yawDiff = abs(angleDiff(viewAngleToTail, mYaw));
    mDisplayStr += std::to_string(static_cast<int>(yawDiff * 180 / M_PI)) + " ";
    if (yawDiff > 0.25 * M_PI)
        return false;

    // 3. tail corner가 모두 이미지 안에 있는지...
    const Eigen::Matrix<float, 2, 4> tailCorners2D = mCorners2D.leftCols(4);
    const Eigen::Matrix<float, 1, 4> tailCornersU = tailCorners2D.row(0);
    const Eigen::Matrix<float, 1, 4> tailCornersV = tailCorners2D.row(1);

    const Eigen::Matrix<bool, 1, 4> valid =
        tailCornersU.array() > 0 && tailCornersU.array() < imgW && tailCornersV.array() > 0 &&
        tailCornersV.array() < imgH;

    if (!valid.all())
        return false;

    // 4. tail projection의 size가 충분히 큰지.
    const MatrixXXb tailMask = getMask(imgH, imgW, true);
    const int tailMaskSize = tailMask.count();
    mDisplayStr += std::to_string(tailMaskSize) + " ";
    if (tailMaskSize < 50 * 50)
        return false;

    // 5. 전방의 물체에 가리진 않는지.
    const int intersection = (stackMask && tailMask).count();
    mDisplayStr += std::to_string(intersection) + " ";
    if (static_cast<float>(intersection) / static_cast<float>(tailMaskSize) > 0.1)
        return false;

    return true;
};

MatrixXXb Instance::getMask(int imgH, int imgW, bool tailOnly) const {

    cv::Mat mask{imgH, imgW, CV_8UC1, cv::Scalar(0)};

    std::vector<cv::Point> points;
    const int numCols = tailOnly ? 4 : mCorners2D.cols();
    for (int c = 0; c < numCols; ++c) {
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
