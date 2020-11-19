#include "taillight/instance.hpp"
#include <algorithm>
#include <opencv2/core/eigen.hpp>

Instance::Instance(
    int classId,
    int trackId,
    std::array<float, 3> xyz,
    std::array<float, 3> lwh,
    float yaw,
    const CalibParams &calib_params) {

    /* ---------------------*
     * Set by input
     * ---------------------*/
    mClassId = classId;
    mTrackId = trackId;
    mXyzCenter << xyz[0], xyz[1], xyz[2];
    mLwh << lwh[0], lwh[1], lwh[2];
    mYaw = yaw;

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
        (calib_params.T_veh2cam * mCorners3D.colwise().homogeneous()).colwise().hnormalized();

    // Project Corners to image
    mCorners2D = (calib_params.K * mCornersCam3D).colwise().hnormalized();

    /* ----------------------------------*
     * Minimun distance to box (roughly)
     * ----------------------------------*/
    mDist = mCorners3D.topRows(2).colwise().norm().minCoeff();

    /* ----*
     * ETC
     * ----*/
    mDisplayStr += std::to_string(mClassId) + "(" + std::to_string(mTrackId) + ") ";
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

    auto renderPairs = [&img,
                        this](const std::vector<std::pair<int, int>> &pairs, cv::Scalar color) {
        for (const auto &p : pairs) {
            cv::Point point1{
                static_cast<int>(this->mCorners2D(0, p.first) + 0.5),
                static_cast<int>(this->mCorners2D(1, p.first) + 0.5),
            };
            cv::Point point2{
                static_cast<int>(this->mCorners2D(0, p.second) + 0.5),
                static_cast<int>(this->mCorners2D(1, p.second) + 0.5),
            };
            cv::line(img, point1, point2, color);
        }
    };

    // Render 3D Box
    const std::vector<std::pair<int, int>> boxPairs{
        {0, 1},
        {0, 2},
        {0, 4},
        {1, 3},
        {1, 5},
        {2, 3},
        {2, 6},
        {3, 7},
        {4, 5},
        {4, 6},
        {5, 7},
        {6, 7},
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
    const cv::Point strPos{
        static_cast<int>(mCorners2D(0, 0) + 0.5),
        static_cast<int>(mCorners2D(1, 0) + 0.5),
    };

    cv::putText(img, mDisplayStr, strPos, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);
}

bool Instance::isTailInSight(int imgH, int imgW, const ArrayXb &stackMask) const {
    // 1. 자동차 맞는지
    if (!isCar())
        return false;

    // 2. tail과 view의 각도
    // yawByView : 차량 뒷면을 바라보는 angle과 차량 yaw angle 간의 차이
    const Eigen::Vector3f tailCenter = mCornersCam3D.leftCols(4).rowwise().mean();
    const float viewAngleToTail = atan2(-tailCenter(0), tailCenter(2));
    const float yawByView = abs(angleDiff(viewAngleToTail, mYaw));
    mDisplayStr += std::to_string(static_cast<int>(yawByView * 180 / M_PI)) + " ";
    if (yawByView > 0.25 * M_PI)
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
    auto [tailU, tailV, tailW, tailH] = getTailRect(imgH, imgW, 0.0);
    const int tailMaskSize = tailW * tailH;
    mDisplayStr += std::to_string(tailMaskSize) + " ";
    if (tailMaskSize < 50 * 50)
        return false;

    // 5. 전방의 물체에 가리진 않는지.
    const int intersection = stackMask.segment(tailU, tailW).count();
    mDisplayStr += std::to_string(intersection) + " ";
    if (static_cast<float>(intersection) / static_cast<float>(tailW) > 0.1)
        return false;

    return true;
};

std::tuple<int, int, int, int> Instance::getTailRect(int imgH, int imgW, float padRatio) const {
    const Eigen::Matrix<float, 2, 4> tailCorners2D = mCorners2D.leftCols(4);
    const Eigen::Matrix<float, 1, 4> tailCornersU = tailCorners2D.row(0);
    const Eigen::Matrix<float, 1, 4> tailCornersV = tailCorners2D.row(1);

    const float minU = tailCornersU.minCoeff();
    const float maxU = tailCornersU.maxCoeff();
    const float minV = tailCornersV.minCoeff();
    const float maxV = tailCornersV.maxCoeff();
    const float padW = (maxU - minU) * padRatio;
    const float padH = (maxV - minV) * padRatio;

    const int minU_int = std::max(static_cast<int>(minU - padW + 0.5), 0);
    const int maxU_int = std::min(static_cast<int>(maxU + padW + 0.5), imgW - 1);
    const int minV_int = std::max(static_cast<int>(minV - padH + 0.5), 0);
    const int maxV_int = std::min(static_cast<int>(maxV + padH + 0.5), imgH - 1);

    // U, V, W, H
    return {minU_int, minV_int, maxU_int - minU_int, maxV_int - minV_int};
}

std::tuple<int, int> Instance::getProjectionLR(int imgW) const {
    const Eigen::Matrix<float, 1, 8> cornersU = mCorners2D.row(0);

    const float minU = cornersU.minCoeff();
    const float maxU = cornersU.maxCoeff();

    const int minU_int = std::max(static_cast<int>(minU), 0);
    const int maxU_int = std::min(static_cast<int>(maxU), imgW - 1);

    return {minU_int, maxU_int};
}

// ----------------
// Deprecated
// ----------------
MatrixXXb Instance::getMask(int imgH, int imgW, bool tailOnly) const {

    cv::Mat mask{imgH, imgW, CV_8UC1, cv::Scalar(0)};

    std::vector<cv::Point> points;
    const int numCols = tailOnly ? 4 : mCorners2D.cols();
    for (int c = 0; c < numCols; ++c) {
        points.emplace_back(
            static_cast<int>(mCorners2D(0, c) + 0.5),
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
