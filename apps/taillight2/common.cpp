#include "common.hpp"

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

float angleDiff(float toAngle, float fromAngle) {
    return remainder((toAngle - fromAngle), 2 * M_PI);
}
