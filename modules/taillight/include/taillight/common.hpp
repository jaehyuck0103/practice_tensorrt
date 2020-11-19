#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <iostream>
#include <numeric>

typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> MatrixXXb;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> ArrayXb;

struct CalibParams {
    const Eigen::Matrix4f RT;
    const Eigen::Matrix4f RTinv;
    const Eigen::Matrix4f RL;
    const Eigen::Matrix4f RLinv;
    const Eigen::Matrix4f T_veh2cam;
    const Eigen::Matrix3f K;

    CalibParams(
        const std::array<float, 16> &RT_vals,
        const std::array<float, 16> &RL_vals,
        const std::array<float, 9> &K_vals)
        : RT{Eigen::Matrix<float, 4, 4, Eigen::RowMajor>{RT_vals.data()}},
          RTinv{RT.inverse()},
          RL{Eigen::Matrix<float, 4, 4, Eigen::RowMajor>{RL_vals.data()}},
          RLinv{RL.inverse()},
          T_veh2cam{RLinv * RTinv},
          K{Eigen::Matrix<float, 3, 3, Eigen::RowMajor>{K_vals.data()}} {}

    void printParams() const {
        std::cout << "RT\n" << RT << std::endl;
        std::cout << "RL\n" << RL << std::endl;
        std::cout << "K\n" << K << std::endl;
    }
};

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

inline const std::array<std::string, 8> STATES{
    {"None",
     "Brake",
     "Left",
     "Brake Left",
     "Right",
     "Brake Right",
     "Emergency",
     "Brake Emergency"}};
