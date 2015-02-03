from data import *
from rntn import *
import cPickle as pkl
import sys

model = pkl.load(open(sys.argv[1]))
dev = pkl.load(open('formatted/dev.pkl'))
test = pkl.load(open('formatted/test.pkl'))

print "accuracy on dev:"
accuracy(model,dev.trees)
print "accuracy on test:"
accuracy(model,test.trees)
