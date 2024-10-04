import os
from argparse import ArgumentParser
from gssr.utils.vastgaussian_utils import split_scene


parser = ArgumentParser("Split the Scene(COLMAP-SFM)")
parser.add_argument("--source_path", "-s", required=True, type=str, help="the colmap-sfm result path")
parser.add_argument("--output_path", "-o", required=True, type=str, help="")
parser.add_argument("--m", type=int, default=2, help="row")
parser.add_argument("--n", type=int, default=2, help="col")
parser.add_argument("--ratio", type=float, default=0.1, help="tile extend ratio")
parser.add_argument("--threshold", type=float, default=0.5, help="visibility threshold")
args = parser.parse_args()

split_scene(args.source_path, args.output_path, args.m, args.n, 
            transform_matrix=None, ratio=args.ratio, threshold=args.threshold, 
            input_format="", output_format=".txt")
