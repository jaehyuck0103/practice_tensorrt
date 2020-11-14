#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <numeric>

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXXb;

// clang-format off
struct CalibParams {
    inline static const Eigen::Matrix4f RT = (
            Eigen::Matrix4f() << -0.005317, 0.003402, 0.999980, 1.624150,
                                 -0.999920, -0.011526, -0.005277, 0.296660,
                                 0.011508, -0.999928, 0.003463, 1.457150,
                                 0, 0, 0, 1
            ).finished();
    inline static const Eigen::Matrix4f RTinv = RT.inverse();
    inline static const Eigen::Matrix4f RL = (
            Eigen::Matrix4f() << 0.999844, 0.001632, -0.017567, 0,
                                 -0.001632, 0.999999, 0, 0,
                                 0.017567, 0.000029, 0.999846, 0,
                                 0, 0, 0, 1
            ).finished();
    inline static const Eigen::Matrix4f RLinv = RL.inverse();
    inline static const Eigen::Matrix3f K = (
            Eigen::Matrix3f() << 819.162645, 0.000000, 640.000000,
                                 0.000000, 819.162645, 240.000000,
                                0.000000, 0.000000, 1.000000
            ).finished();
};
// clang-format on

// angleDiff (-pi, pi)
inline float angleDiff(float toAngle, float fromAngle) {
    return remainder((toAngle - fromAngle), 2 * M_PI);
}

namespace Cfg {
constexpr int batchSize = 8;
}

namespace RegCfg {
constexpr int inB = Cfg::batchSize;
constexpr int inH = 224;
constexpr int inW = 224;
constexpr int inC = 3;
inline const std::vector<int> inDims = {inB, inH, inW, inC};
inline const int inNumEl = std::accumulate(inDims.begin(), inDims.end(), 1, std::multiplies<>());

constexpr int outB = Cfg::batchSize;
constexpr int outC = 4;
inline const std::vector<int> outDims = {outB, outC};
inline const int outNumEl =
    std::accumulate(outDims.begin(), outDims.end(), 1, std::multiplies<>());
} // namespace RegCfg

namespace UNetCfg {
constexpr int inB = Cfg::batchSize;
constexpr int inSeqLen = 1;
constexpr int inH = 112;
constexpr int inW = 112;
constexpr int inC = 3;
inline const std::vector<int> inDims = {inB, inSeqLen, inH, inW, inC};
inline const int inNumEl = std::accumulate(inDims.begin(), inDims.end(), 1, std::multiplies<>());

constexpr int outB = Cfg::batchSize;
constexpr int outSeqLen = 1;
constexpr int outC = 64;
constexpr int outH = 28;
constexpr int outW = 28;
inline const std::vector<int> outDims = {outB, outSeqLen, outC, outH, outW};
inline const int outNumEl =
    std::accumulate(outDims.begin(), outDims.end(), 1, std::multiplies<>());
} // namespace UNetCfg

namespace CNN3DCfg {
constexpr int inB = Cfg::batchSize;
constexpr int inSeqLen = 16;
constexpr int inC = UNetCfg::outC;
constexpr int inH = UNetCfg::outH;
constexpr int inW = UNetCfg::outW;
inline const std::vector<int> inDims = {inB, inSeqLen, inC, inH, inW};
inline const int inNumEl = std::accumulate(inDims.begin(), inDims.end(), 1, std::multiplies<>());

constexpr int outB = Cfg::batchSize;
constexpr int outC = 1;
inline const std::vector<int> outDims = {outB, outC};
inline const int outNumEl =
    std::accumulate(outDims.begin(), outDims.end(), 1, std::multiplies<>());
} // namespace CNN3DCfg

constexpr int ENCODED_TAIL_SIZE = UNetCfg::outC * UNetCfg::outH * UNetCfg::outW;

inline const std::array<std::string, 8> STATES{{"None", "Brake", "Left", "Brake Left", "Right",
                                                "Brake Right", "Emergency", "Brake Emergency"}};
