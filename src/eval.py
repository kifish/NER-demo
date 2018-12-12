# see https://github.com/chakki-works/seqeval for package installation
# Well-tested by using the Perl script conlleval, which can be used for measuring the performance of a system that has processed the CoNLL-2000 shared task data.

import sys
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report


def main(argv):
    y_pred = []
    y_true = []
    with open(argv[1], encoding='utf-8') as f1:
        for line in f1:
            if line.strip():
                y_pred.append(line.split('\t')[1])
    with open(argv[2], encoding='utf-8') as f2:
        for line in f2:
            if line.strip():
                y_true.append(line.split('\t')[1])
    if len(y_true) != len(y_pred):
        print("Length of your prediction should be equal to gold's.")
    else:
        print(classification_report(y_true, y_pred, 4))


if __name__ == '__main__':
    main(sys.argv)
