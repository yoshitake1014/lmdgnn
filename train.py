from models import lmdgnn

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='as', help='')
parser.add_argument('--methods', type=str, default='lmdgnn', help='')


args = parser.parse_args()


def main():
    dataset = args.datasets
    if dataset == 'as':
        pass

    method = args.methods
    if method == 'lmdgnn':
        model = lmdgnn.LMDGNN(args)
        pass

    pass


if __name__ == "__main__":
    main()
