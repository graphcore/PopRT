# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate IPU Utilization.')
    parser.add_argument(
        '-c', '--cycles', type=float, required=True, help="Specify the cycles."
    )
    parser.add_argument(
        '-f', '--flops', type=float, required=True, help="Specify the flops."
    )
    parser.add_argument(
        '-n', '--ipus', type=int, default=1, help="Specify number of IPUs."
    )
    parser.add_argument(
        '-p',
        '--platform',
        type=str,
        default='C600',
        choices=['C600', 'M2000', 'Bow2000'],
        help="Specify the platform.",
    )
    args = parser.parse_args()

    clock_sppeds = {'C600': 1.5e9, 'M2000': 1.33e9, 'Bow2000': 1.85e9}

    if args.platform in clock_sppeds.keys():
        clock_speed = clock_sppeds[args.platform]
    else:
        raise ValueError(f'{args.platform} is INVALID.')

    latency_s = args.cycles / clock_speed
    tflops = args.flops / (latency_s * 1e12 * args.ipus)

    print(f"The effective computing power is {tflops} TFLOPS")
