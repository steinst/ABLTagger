import sys
import argparse

def coarsify(inputline):
    try:
        temp = inputline.strip().split()
        return temp[0] + '\t' + temp[1][0] + '\n'
    except:
        return ''

if __name__ == '__main__':
    # reading input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', help='Name of input file.', default="Full.fine.txt")
    parser.add_argument('--output', '-o', help='Name of output file.', default="Full.coarse.txt")

    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    with open(args.output, "w") as f:
        with open(args.input) as training:
            line = training.readline()
            while line:
                f.write(coarsify(line))

