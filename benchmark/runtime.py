import argparse
import collections

import numpy as np
import torch

import dct


torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main(args):
    torch.cuda.current_device()
    torch.cuda.empty_cache()

    model = dct.DCT2d().eval()
    model.to(args.device)

    result_dict = collections.OrderedDict()
    result_dict["runtime"] = []
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    x = torch.zeros(1, 3, 1024, 1024, device=args.device)

    if args.fp16:
        model.half()
        x = x.half()

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)

    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.iterations):
            t0.record()
            model(x)
            t1.record()

            torch.cuda.synchronize()
            result_dict["runtime"].append(t0.elapsed_time(t1))

    print(f"{np.mean(result_dict['runtime']):.4f}±{np.std(result_dict['runtime']):.4f}ms")


if __name__ == "__main__":
    main(parse_args())
