#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXXb;

struct CalibParams {
    static const Eigen::Matrix4f RT;
    static const Eigen::Matrix4f RTinv;
    static const Eigen::Matrix3f K;
};

// angleDiff (-pi, pi)
float angleDiff(float toAngle, float fromAngle);
