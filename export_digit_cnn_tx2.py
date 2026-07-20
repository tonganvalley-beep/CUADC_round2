"""Create a PyTorch-1.11-compatible TorchScript model on Jetson TX2.

Run this file on the TX2, not on the training computer:
    python3.8 export_digit_cnn_tx2.py \
        --input weights/digit_cnn.pth \
        --output weights/digit_cnn.ts
"""

import argparse
import os

import torch
from torch import nn


class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, groups=16, bias=False),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, groups=32, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="weights/digit_cnn.pth")
    parser.add_argument("--output", default="weights/digit_cnn.ts")
    args = parser.parse_args()

    model = DigitCNN()
    state = torch.load(args.input, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    example = torch.zeros(2, 1, 32, 32)
    with torch.no_grad():
        expected = model(example)
    scripted = torch.jit.trace(model, example)
    actual = scripted(example)
    if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        raise RuntimeError("TorchScript verification failed")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    scripted.save(args.output)

    loaded = torch.jit.load(args.output, map_location="cpu")
    check = loaded(example)
    if tuple(check.shape) != (2, 10):
        raise RuntimeError("unexpected output shape: {}".format(tuple(check.shape)))
    print("TX2 TorchScript saved: {}".format(os.path.abspath(args.output)))
    print("PyTorch version: {}".format(torch.__version__))
    print("Output shape: {}".format(tuple(check.shape)))


if __name__ == "__main__":
    main()
