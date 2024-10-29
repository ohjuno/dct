import argparse

import cv2
import numpy as np
import torch

import dct


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--quantization", action="store_true")
    return parser.parse_args()


def l_infinity_norm(x: torch.Tensor, other: torch.Tensor) -> int:
    return (x - other).abs().max().int()


def main(args):
    dct2d, idct2d = dct.DCT2d(device=args.device), dct.IDCT2d(device=args.device)

    image = cv2.imread("asset/mountain.jpg")
    image = cv2.resize(image, (1024, 1024))
    image_visualize = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    image = torch.from_numpy(image)
    image = image.to(args.device)
    image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
    image = image.to(torch.float32).sub(128)

    if args.fp16:
        image = image.half()
        dct2d.half()
        idct2d.half()

    z = dct2d(image)
    image_restored = idct2d(z).round()

    print(f"L-infinity: {l_infinity_norm(image, image_restored)}")

    image_restored = image_restored.add(128).clamp(0, 255).to(torch.uint8)
    image_restored = image_restored.squeeze(0).permute(1, 2, 0).contiguous()
    image_restored = image_restored.numpy(force=True)
    image_restored = cv2.cvtColor(image_restored, cv2.COLOR_YCrCb2BGR)

    cv2.imshow("", np.hstack([image_visualize, image_restored]))
    cv2.waitKey(0)


if __name__ == "__main__":
    main(parse_args())
