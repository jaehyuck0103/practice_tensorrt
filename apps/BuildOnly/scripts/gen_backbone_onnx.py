from pathlib import Path

import numpy as np
import onnxruntime
import torch
from torchvision import models

ONNX_SAVE_ROOT = Path(__file__).parent / "onnx_models"


@torch.no_grad()
def gen_onnx(net, net_name, verbose=False):
    print(f"Genrate onnx_model for {net_name}")
    net.eval()
    H = 480
    W = 1280
    onnx_save_path = ONNX_SAVE_ROOT / f"{net_name}.onnx"

    # Input to the model
    batch_size = 1
    dummy_input_rgb = torch.randn(batch_size, 3, H, W)  # , device="cuda")

    # forward_all
    torch.onnx.export(
        net,
        dummy_input_rgb,
        onnx_save_path,
        verbose=verbose,
        opset_version=11,
        input_names=["Input"],
        output_names=["Output"],
    )

    # Onnx랑 원래 pytorch랑 inference 결과 같은지 확인.
    dummy_input_rgb = torch.ones(batch_size, 3, H, W)  # , device="cuda")
    dummy_input_np = np.ones([batch_size, 3, H, W], dtype=np.float32)

    torch_outs = net(dummy_input_rgb)

    ort_session = onnxruntime.InferenceSession(str(onnx_save_path))
    ort_inputs = {
        ort_session.get_inputs()[0].name: dummy_input_np,
    }
    ort_outs = ort_session.run(None, ort_inputs)

    print(torch_outs.detach().cpu().numpy().reshape(-1)[:10])
    print(ort_outs[0].reshape(-1)[:10])
    print("")


def main():
    ONNX_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    # VGG11_bn (2.11ms)
    # gen_onnx(models.vgg11_bn().features, "VGG11_bn")

    # ResNet18 (0.86ms)
    gen_onnx(models.resnet18(pretrained=True), "ResNet18")

    # Densenet121 (5.2ms)
    # gen_onnx(models.densenet121(), "Densenet121")

    # shufflenet v2 x1.0 (0.87ms)
    gen_onnx(models.shufflenet_v2_x1_0(pretrained=True), "ShuffleNetV2_x1.0")

    """  # hard sigmoid 미지원
    # MobileNet V3 Large
    gen_onnx(models.mobilenet_v3_large(), "MobileNetV3_large")

    # MobileNet V3 Small
    gen_onnx(models.mobilenet_v3_small(), "MobileNetV3_small")
    """

    # ResNeXt-50-32x4d (3.15ms)
    # gen_onnx(models.resnext50_32x4d(), "ResNeXt50_32x4d")

    # Wide ResNet-50-2 (3.37ms)
    # gen_onnx(models.wide_resnet50_2(), "WideResNet_50_2")

    # MNASNet 1.0 (0.99ms)
    gen_onnx(models.mnasnet1_0(pretrained=True), "MNASNet_1_0")

    # MNASNet 0.5 (0.6ms)
    gen_onnx(models.mnasnet0_5(pretrained=True), "MNASNet_0_5")


if __name__ == "__main__":
    main()
