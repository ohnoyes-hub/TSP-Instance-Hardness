from utils.file_utils import get_result_path
from core.experiment import experiment
import numpy as np
import json
import os
import glob
import time
from icecream import ic # debugging
# job scripts stuff
import argparse # for parsing arguments
import ast # for converting string to list

parser = argparse.ArgumentParser(description='Run the experiment with provided parameters.')

parser.add_argument('sizes', type=str, 
                    help='A list of city sizes, e.g., "[10,12]"')
parser.add_argument('ranges', type=str, 
                    help='A list of value ranges, e.g., "[10,1000]"')
parser.add_argument('mutations', type=int, 
                    help='An integer number of mutations, e.g., 500')
parser.add_argument('continuation', type=str, default="", nargs='?', 
                    help='A list of matrix continuations, e.g., "[(7,10),(50,10)]".')
parser.add_argument('--tsp_type', type=str, choices=['euclidean', 'asymmetric'], required=True,
                    help='Type of TSP to generate: symmetric or asymmetric.')
parser.add_argument('--distribution', type=str, choices=['uniform', 'lognormal'], required=True,
                    help='Distribution to use for generating the TSP instance.')
parser.add_argument('--mutation_strategy', type=str, choices=['swap', 'scramble', 'wouter'], required=True,
                    help='Mutation strategy to use.')


def main():
    args = parser.parse_args()
    sizes = ast.literal_eval(args.sizes)
    ranges = ast.literal_eval(args.ranges)

    # auto-detect continuation
    continuations = []
    for citysize in sizes:
        for rang in ranges:
            result_file = get_result_path(citysize, rang, args.distribution, 
                                        args.tsp_type, args.mutation_strategy, is_final=True)
            if os.path.exists(result_file):
                continue  # Skip completed experiments

            # otherwise, load continuation file if it exists
            continuation_file = os.path.join("Continuation", f"{args.distribution}_{args.tsp_type}", 
                                            f"city{citysize}_range{rang}_{args.mutation_strategy}.json")
            if os.path.exists(continuation_file):
                continuations.append(f"{citysize},{rang}") # partial results exist for this configuration

    # manual continuation: merge args.continuation with detected continuations
    if args.continuation != "":
        manual_continuations = [f"{tup[0]},{tup[1]}" for tup in ast.literal_eval(args.continuation)]
        continuations = list(set(continuations + manual_continuations))

    experiment(
        sizes, ranges, args.mutations,
        continuations=continuations,
        distribution=args.distribution,
        tsp_type=args.tsp_type,
        mutation_strategy=args.mutation_strategy
    )

if __name__ == "__main__":
    main()