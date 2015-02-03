'''
Ankit Kumar
ankitk@stanford.edu

control scripts/utils for the RNTN
'''
import cPickle as pkl
from data import *
import random
import time
from stepper import *
from rntn import *


class control(object):

	def __init__(self,mbsize=27, d=25, V=16582, epochs=70, unsupervised=False):
		self.mbsize = mbsize
		self.d = d
		self.V = V
		self.epochs = epochs
		self.unsupervised = unsupervised

		train = pkl.load(open('formatted/train.pkl'))
		dev = pkl.load(open('formatted/dev.pkl'))
		both = train.trees #+ dev.trees
		self.minibatches = [both[i:i+mbsize] for i in range(0,len(both), mbsize)]

	def run(self, lr, reg, learning_algo, regularize_bias, activation,transform_eye=False, reset_adagrad=None,name='', costfunction=softmax_crossentropy_cost,wordActivations=True,gpu=False):
		s = stepper(lr=lr, reg=reg, learning_algo=learning_algo)
		if self.unsupervised:
			model = initRNTN(self.d, self.V, self.V, regularize_bias=regularize_bias, activation=activation, transform_eye=transform_eye, wordActivations=wordActivations,gpu=gpu)
		else:
			model = initRNTN(self.d, self.V, 5, regularize_bias=regularize_bias, activation=activation, transform_eye=transform_eye,gpu=gpu)
		costs_full = []
		# try:
		for ep in range(self.epochs):
			if reset_adagrad is not None:
				if ep > 0 and (ep%reset_adagrad == 0):
					print 'resetting adagrad'
					s.historical_gparams=None
			tick = time.time()
			costs = []
			random.shuffle(self.minibatches) # so we dont see the same order every time (matters bc of adagrad resetting)
			for minibatch in self.minibatches:
				cost = s.step(model, minibatch,costfunction,epoch=ep)
				costs.append(cost)
				costs_full.append(cost)
			tock = time.time()
			if ep == 1:
				s.gradcheck(model,minibatch,costfunction)
			print "epoch %s took %s" % (ep, str(tock-tick))
			print "average cost %s" % str(sum(costs)/float(len(costs)))
			if ep>0 and ep%10 == 0:
				pkl.dump(model, open('checkpoints/%s_epoch_%s.pkl' % (name,str(ep)),'w'))
				
		pkl.dump(model, open('checkpoints/%s_epochs_final.pkl' % name,'w'))
		# except:
		return costs_full,model

	def validate(self, num_epochs=None):
		'''
		try a bunch of settings and check the validation set on them
		'''
		accuracies = {}
		lrs = [1e-2, 1e-1,1e-3]
		regs = [1e-6, 1e-4,1e-5]
		learning_algos = ['sgd']#,'adagrad']
		regularize_biases = [True]#, False]
		initLrs = [0.01, 0.0001]
		activations = ['tanh']

		if num_epochs is None:
			num_epochs = self.epochs

		for lr_base in lrs:
			for reg in regs:
				for learning_algo in learning_algos:
					
					for regularize_bias in regularize_biases:
						for initLr in initLrs:
							for activation in activations:
								if learning_algo == 'sgd':
									lr = lr_base*1e-2 # sgd learning rates need to be significantly lower
								else:
									lr = lr_base
								try:
									print "now doing: %s, %s, %s, %s, %s, %s" % (str(lr),str(reg),str(learning_algo),str(regularize_bias),str(initLr), str(activation))
									s = stepper(lr=lr, reg=reg, learning_algo=learning_algo)
									model = initRNTN(self.d, self.V, 5, regularize_bias=regularize_bias, initLr=initLr, activation=activation)
									costs_full = []
									for ep in range(num_epochs):
										tick = time.time()
										costs = []
										for minibatch in self.minibatches:
											cost = s.step(model, minibatch,epoch=ep)
											costs.append(cost)
											costs_full.append(cost)
										tock = time.time()
										print "epoch %s took %s" % (ep, str(tock-tick))
										print "average cost %s" % str(sum(costs)/float(len(costs)))
									
									# pkl.dump(costs_full, open('checkpoints/costs_%s%s%s%s%s%s' % (str(lr),str(reg),str(learning_algo),str(regularize_bias),str(initLr),str(activation)),'w'))
									# pkl.dump(model, open('checkpoints/model_%s%s%s%s%s%s' % (str(lr),str(reg),str(learning_algo),str(regularize_bias),str(initLr),str(activation)),'w'))
									accuracies[(lr,reg,learning_algo,regularize_bias,initLr)] = accuracy(model,self.dev.trees)
									print accuracies
								except Exception as e:
									print e
									continue

		return accuracies
if __name__ == '__main__':
	c = control(epochs=100, unsupervised=False)
	c1,m1 = c.run(lr=1e-2, reg={'Ws': 1e-4, 'L':1e-4, 'W':1e-3, 'V':1e-3, 'bs':1e-4, 'b':1e-3}, learning_algo='adagrad', regularize_bias=True, activation='tanh',transform_eye=True, reset_adagrad=2, name="reset_2", costfunction=softmax_crossentropy_cost, wordActivations=True, gpu=False)
	# c2,m2 = c.run(lr=1e-2, reg={'Ws': 1e-4, 'L':1e-4, 'W':1e-3, 'V':1e-3, 'bs':1e-4, 'b':1e-3}, learning_algo='adagrad', regularize_bias=True, activation='tanh',transform_eye=True, reset_adagrad=3, name="reset_3", costfunction=softmax_crossentropy_cost, wordActivations=True)
	# c3,m3 = c.run(lr=1e-2, reg={'Ws': 1e-4, 'L':1e-4, 'W':1e-3, 'V':1e-3, 'bs':1e-4, 'b':1e-3}, learning_algo='adagrad', regularize_bias=True, activation='tanh',transform_eye=True, reset_adagrad=3, name="unsup", costfunction=unsupervised_softmax_cost, wordActivations=True)
	# c2,m2 = c.run(1e-4,1e-6, 'sgd', True, 0.01, 'tanh')
	# c3,m3 = c.run(1e-2, 1e-4, 'adagrad', True, 0.0001, 'tanh')



	# acc = c.validate(num_epochs=5)

	# pkl.dump(acc, open('acc.pkl','w'))
								



