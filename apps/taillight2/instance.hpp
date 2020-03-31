#pragma once
#include "common.hpp"
#include <nlohmann/json.hpp>

#include <opencv2/opencv.hpp>

using json = nlohmann::json;

class Instance {
  public:
    Instance(const json &inputRow);

    bool isAnyCornersInImage(int imgH, int imgW) const;
    bool isAllCornersFrontOfCam() const;
    bool isValidForProjection(int imgH, int imgW) const;

    void renderToImg(cv::Mat &img) const;
    MatrixXXb getMask(int imgH, int imgW) const;

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
