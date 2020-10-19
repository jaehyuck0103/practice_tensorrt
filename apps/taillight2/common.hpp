#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numeric>

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXXb;

struct CalibParams {
    static const Eigen::Matrix4f RT;
    static const Eigen::Matrix4f RTinv;
    static const Eigen::Matrix3f K;
};

// angleDiff (-pi, pi)
float angleDiff(float toAngle, float fromAngle);

namespace Cfg {
constexpr int batchSize = 8;
}

namespace RegCfg {
constexpr int inB = Cfg::batchSize;
constexpr int inC = 3;
constexpr int inH = 224;
constexpr int inW = 224;
inline const std::vector<int> inDims = {inB, inC, inH, inW};
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
constexpr int inC = 3;
constexpr int inH = 112;
constexpr int inW = 112;
inline const std::vector<int> inDims = {inB, inSeqLen, inC, inH, inW};
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
