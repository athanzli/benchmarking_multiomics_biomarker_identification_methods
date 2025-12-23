import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MORE.")

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate.')

    parser.add_argument('--n_hidden', type=int, default=10,
                        help='Number of hidden units.')
    parser.add_argument('--n_head', type=int, default=4,
                        help='Number of head.')
    parser.add_argument('--nmodal', type=int, default=3,
                        help='Number of omics.')

    return parser.parse_args()

