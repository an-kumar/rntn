'''
Ankit Kumar
ankitk@stanford.edu

learning steps for the rntn model

uses adagrad as in the paper
'''
from rntn import *
import cPickle as pkl
from data import *


def softmax_crossentropy_cost(tree):
    '''
    a costfunction that computes softmax crossentropy gradients and cost
    '''
    cost = 0.
    for node in tree.nodes:
        cost -= np.log(node.rntnparams['softmax'][node.label])
        node.rntnparams['dtop'] = np.copy(node.rntnparams['softmax'])
        node.rntnparams['dtop'][node.label] -= 1.
    return cost


def unsupervised_softmax_cost(tree):
    '''
    a costfunction for unsupervised training
    '''
    cost = 0.
    for node in tree.nodes:
        windices = node.rntnparams['wIndices']
        # print np.log(node.rntnparams['softmax'][windices])
        cost -= np.mean(np.log(node.rntnparams['softmax'][windices]))
     
        node.rntnparams['dtop'] = np.copy(node.rntnparams['softmax'])
        node.rntnparams['dtop'][windices] -= 1./len(windices)
    return cost


class stepper(object):

    def __init__(self, lr=1e-2, reg={'Ws': 1e-4, 'L':1e-4, 'W':1e-3, 'V':1e-3, 'bs':1e-4, 'b':1e-3}, learning_algo='adagrad', momentum=.9):
        self.lr = lr
        self.start_lr = lr # for SGD decreasing
        self.reg = reg

        self.historical_gparams = None
        self.learning_algo = learning_algo
        self.momentum = momentum

    def step(self,model, minibatch, costfunction, epoch=0):
        '''
        performs a single step given the model and the minibatch

        minibatch is an iterable of PTBtrees

        returns the cost on the minibatch
        '''
        # iterate through batch and go forward,backward,cost
        cost, gparams = self.cost_fbprop(model, minibatch, costfunction)

        # # get regularization cost and gparams
        # reg_cost, reg_gparams = self.cost_regularize(model)

        # # add the two gparam dicts and scale by minibatch size (so that larger minibatches doesn't greatly change anything)
        # minibatch_scale = 1/float(len(minibatch)) 
        # for k in reg_gparams:
        #     # only add the reg gparams to gparams; non reg gparams stay as they were
        #     gparams[k] += reg_gparams[k]
        #     gparams[k] * minibatch_scale
        # now update the model
        if self.learning_algo == 'adagrad':
            # if reset_adagrad is not None:
                # if epoch > 0 and (epoch % reset_adagrad == 0):
                #     print "resetting adagrad"
                #     self.historical_gparams = None
            self.update_model_adagrad(model, gparams)
        elif self.learning_algo == 'sgd':
            self.lr = self.start_lr * (.98**epoch) # trick to scale the lr down

            self.update_model_sgd(model,gparams)

        # return the cost (scaled by minibatch size)
        return cost

    def update_model_sgd(self, model, gparams):
        '''
        sgd with momentum
        '''
        # first we get current updates:
        upds = {}
        for p in gparams:
            upds[p] = -1*self.lr * gparams[p]

        # now add momentum term
        if self.historical_gparams is None:
            self.historical_gparams = upds
        else:
            for p in upds:
                upds[p] += self.momentum * self.historical_gparams[p]
                self.historical_gparams[p] = upds[p]

        for p in upds:
            model[p] += upds[p]

    def update_model_adagrad(self, model, gparams):
        '''
        updates a model using adagrad learning algorithm
        '''
        # we're gonna use AdaGrad because that's what is listed in the paper
        # note that I have never really used AdaGrad before so I'm going off the original paper + this helpful blog: http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
        epsilon = 1e-6
        if self.historical_gparams is None:
            adj_gparams = gparams
            self.historical_gparams = {} # set the historical gparams
            for key in gparams:
                self.historical_gparams[key] = (gparams[key] ** 2)
        else:
            adj_gparams = {}
            for key in gparams:
                grad = gparams[key]
                # add square of current gradient to historical gradient

                self.historical_gparams[key] += (grad**2)
            
                # calculate adjusted gradient
                if np.isinf(self.historical_gparams[key]).sum() > 0 or np.isnan(self.historical_gparams[key]).sum()> 0:
                    print key
                    raise Exception ("fuck.")
                adj_gparams[key] = grad / (epsilon + np.sqrt(self.historical_gparams[key]))

        # now one by one update the params
        for key in adj_gparams:
            model[key] -= self.lr * adj_gparams[key]

    def cost_regularize(self, model):
        '''
        computes the cost and fills a dict of gparams for the regularization
        '''
        cost = 0.
        gparams = {}
        for p in model['regularize']:
            gparams[p] = model[p] * self.reg[p]
            cost += np.sum(model[p]**2) * (self.reg[p]) * .5 # .5 so that the gradient is just reg*model[p], otherwise would be 2x

        return cost, gparams


    def cost_fbprop(self,model,minibatch,costfunction):
        '''
        total cost and gparams
        '''
        cost, gparams = self.cost_fbprop_softmax(model, minibatch, costfunction)
        reg_cost,reg_gparams = self.cost_regularize(model)

        # add the two gparam dicts and scale by minibatch size (so that larger minibatches doesn't greatly change anything)
        minibatch_scale = 1/float(len(minibatch)) 
        for k in gparams:
            gparams[k] *= minibatch_scale
        for k in reg_gparams:
            # only add the reg gparams to gparams; non reg gparams stay as they were
            gparams[k] += reg_gparams[k]

        cost = cost*minibatch_scale + reg_cost
        # cost *= minibatch_scale
        return cost, gparams


    def cost_fbprop_softmax(self, model, minibatch,costfunction):
        '''
        iterates through the batch and goes forward, backward, and computes cost

        returns cost and an aggregated gparam dict
        '''
        cost = 0.
        gparams = {k:np.zeros(model[k].shape) for k in model['update']}
        # below is commented because in practice it was slower.
        batch_fprop(model, minibatch) 
        for tree in minibatch:
            # fprop, bprop
            # fprop(model, tree)
            cost += costfunction(tree) # this should add node.rntnparams['dh']
            local_gparams = bprop(model, tree)
            # accumulate the dicts
            for k in local_gparams:
                gparams[k] += local_gparams[k]
            # cost += self.compute_cost(tree)

        return cost, gparams

    def compute_cost(self, tree):
        ''' computes the cost of the tree, assumed to already be fpropd. note that this does NOT include regularization cost.'''
        cost = 0.
        for node in tree.nodes:
            if node.rntnparams['softmax'][node.label] == 0:
                print node.rntnparams
            cost -= np.log(node.rntnparams['softmax'][node.label])
        return cost

    def gradcheck(self, model, minibatch, costfunction, num_checks=5, epsilon=1e-5):
        '''
        does a gradcheck on the minibatch

        lot of the code is very similar to karpathy neuraltalk
        '''
        # run forward, backward to get grads & cost
        for it in range(num_checks):
            cost, gparams = self.cost_fbprop(model, minibatch,costfunction)


            for p in model['update']:
                mat = model[p]
                grad = gparams[p]
                assert mat.shape == grad.shape, "shapes dont match"
                # let's also do a max to get out of numerical stuff (hopefully)
                ri = np.argmax(grad.flat)
                old = mat.flat[ri]
                # add epsilon
                mat.flat[ri] = old + epsilon
                c1,_ = self.cost_fbprop(model, minibatch,costfunction)
                # subtract epsilon
                mat.flat[ri] = old - epsilon
                c2, _=self.cost_fbprop(model, minibatch,costfunction)
                # back to normal
                mat.flat[ri] = old
                analytic = grad.flat[ri]
                numerical = (c1 - c2) / (2*epsilon)
                print "MAX: param: %s. analytical grad: %s. numerical grad: %s. relative error: %s" % (p, str(analytic), str(numerical), str(abs(analytic - numerical) / abs(numerical + analytic)))




                ri = np.random.randint(mat.size)
                old = mat.flat[ri]
                # add epsilon
                mat.flat[ri] = old + epsilon
                c1,_ = self.cost_fbprop(model, minibatch,costfunction)
                # subtract epsilon
                mat.flat[ri] = old - epsilon
                c2, _=self.cost_fbprop(model, minibatch,costfunction)
                # back to normal
                mat.flat[ri] = old

                analytic = grad.flat[ri]
                numerical = (c1 - c2) / (2*epsilon)

                print "param: %s. analytical grad: %s. numerical grad: %s. relative error: %s" % (p, str(analytic), str(numerical), str(abs(analytic - numerical) / abs(numerical + analytic)))


if __name__ == '__main__':
    model1 = initRNTN(25, 16582, 5, activation='tanh', wordActivations=False)
    # model2= initRNTN(25, 16582, 5, activation='relu')

    train = pkl.load(open('formatted/train.pkl'))
    s1 = stepper()
    # s2 = stepper()

    # test step
    mbsize = 27
    minibatches = [train.trees[i:i+mbsize] for i in range(0,len(train.trees), mbsize)][:2]
    for mb in minibatches:
        s1.step(model1, mb, softmax_crossentropy_cost) 
    # s2.step(model2, train.trees[:50])

    # test grad
    s1.gradcheck(model1,train.trees[51:71],softmax_crossentropy_cost)
    print "\n\n"
    # s2.gradcheck(model2, train.trees[51:71])

