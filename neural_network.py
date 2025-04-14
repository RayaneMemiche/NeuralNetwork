#!/usr/bin/python3

import os
import numpy as np
from mlp import MLP
import sys
from mytorch import exec_torch


def help_flag():
    print("""USAGE
\t./neural_network.py [-new | -train | -test]
DESCRIPTION
\t-new\tname_ia, input_size, num_neurons_first_layer, num_hidden_layers, neurons_per_hidden_layer
\t-train\tname_ia, end_name_of_training_data, epochs
\t-test\tname_ia, end_name_of_testing_data""")

def errors(argv) -> int:
    valid_options = ["-new", "-train", "-test"]

    if argv[1] not in valid_options:
        print("\033[91mError\033[0m\nInvalid option provided.", file=sys.stderr)
        return 84

def main():
    if len(sys.argv) == 1:
        print("\033[91mError\033[0m\nWrong number of argv: 1", file=sys.stderr)
        exit(84)
    if sys.argv[1] in ("-h", "--help"):
        help_flag()
        exit(0)
    if errors(sys.argv) == 84:
        exit(84)
    exec_torch(sys.argv)

if __name__ == '__main__':
    main()
