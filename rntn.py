'''
Ankit Kumar
ankitk@stanford.edu

RNTN model implementation

This implementation follows the style of https://github.com/karpathy/neuraltalk somewhat
'''

import numpy as np


def initUniform(shape, r):
    ''' returns a matrix sampled from uniform -r, r '''
    return r * ((np.random.rand(*shape) * 2) - 1)

def initGauss(shape, r):
    ''' returns a matrix sample from the normal distribution centered at 0 with std 1, scaled by r '''
    return r * np.random.randn(*shape)

def initb(n):
    # this should be 0s
    return np.zeros(n)

def build_theano_gpu_func():
    import theano
    import theano.tensor as T

    '''
    this is a theano function to compute the softmax vectors for an entire batch
    '''
    # instantiate variables
    vecs = T.matrix()
    Ws = T.matrix()
    bs = T.vector() # matrix because it's a column vector in this formulation, which theano doesn't supportq

    # in this formulation, vecs is num_vecs x vec_size, ws is vec_size x output_size (i.e opposite formulation from the rest of this implementation)
    scores = T.nnet.softmax(T.dot(vecs,Ws) + bs)

    fn = theano.function([vecs,Ws,bs],[scores],allow_input_downcast=True)
    return fn


def initRNTN(d=25, vocab_size=16582, output_size=5, transform_eye=True, regularize_bias=True, activation='tanh', gpu=False, wordActivations=True):
    '''
    creates an RNTN model represented as a python dict
    '''
    model = {}
    Vscale = 1./(d*4)
    Wscale = 1./(np.sqrt(d)*2)
    Wsscale = 1./(np.sqrt(d))
    # learnable params
    # I went through the stanford coreNLP source code to find some of these initializations
    model['L'] = initGauss((d,vocab_size),.1)
    model['V'] = initUniform((d,2*d,2*d), Vscale) # making dx2dx2d rather than 2dx2dxd as in paper for intuitive V[i] slicing
    model['W'] = initUniform((d,2*d),Wscale)
    if transform_eye:
        # this was in the coreNLP source code, not in the paper; i'll try it.
        model['W'] += np.hstack([np.eye(d),np.eye(d)])
    model['b'] = initb(d) # only need one bias here as adding a bias for W and V multiplications would be redundant (would be b1+b2)
    model['Ws'] = initUniform((output_size,d),Wsscale)
    model['bs'] = initb(output_size)

    # activation func
    model['activation'] = activation
    model['wordActivations'] = wordActivations
    # what to learn
    model['update'] = ['L','V','W','b','Ws','bs']
    model['regularize'] = ['L','V','W','Ws'] # i'm not 100% sure why but many implementations of different models i see don't regularize the biases, so i'm going to go with that

    if regularize_bias:
        model['regularize'] += ['b','bs']
    # todo: should i regularize b? maybe do validation to see.
    if gpu:
        model['gpu'] = True
        model['gpu_func'] = build_theano_gpu_func()
    else:
        model['gpu'] = False

    return model

def predict(model, batch):
    '''
    predicts a batch of trees
    '''
    batch_fprop(model,batch)
    for tree in batch:
        for node in tree.nodes:
            node.rntnparams['predicted'] = np.argmax(node.rntnparams['softmax'])

def accuracy(model, batch):
    '''
    predicts and calculates some accuracy statistics
    '''
    # for all nodes
    nodes_total = 0
    nodes_corr = 0
    # for only root nodes
    roots_total = 0
    roots_corr = 0
    predict(model,batch)
    # todo: what does the paper mean by 'ignore neutral sentences'?
    for tree in batch:
        for node in tree.nodes:
            nodes_total += 1
            if node.label == node.rntnparams['predicted']:
                nodes_corr += 1
        roots_total += 1
        if tree.root.label == tree.root.rntnparams['predicted']:
            roots_corr += 1
    an = float(nodes_corr)/nodes_total
    rn = float(roots_corr)/roots_total
    print "accuracy on all nodes: %s" % str(an)
    print "accuracy on just toplevel nodes (sentences): %s" % str(rn)
    return (an,rn)


def batch_fprop(model, batch):
    '''
    forward pass of the model on a batch
    '''
    # fprop all the trees to get their hidden vectors
    nodes =[]
    for tree in batch:
        fprop(model, tree)
        nodes += tree.nodes
    Ws = model['Ws']
    bs = model['bs']    
    stacked_vecs = np.hstack([node.rntnparams['vec'][:,np.newaxis] for node in nodes])

    if model['gpu']:
        # gpu softmax computation via theano (run with correct flags)
        p = model['gpu_func'](stacked_vecs.T, Ws.T, bs)[0]
        for i in range(len(nodes)):
            node = nodes[i]
            node.rntnparams['softmax'] = p[i]
   

    else:
        # batch softmax computation
        scores = np.dot(Ws, stacked_vecs) + bs[:,np.newaxis]
        scores -= np.max(scores, axis=0) # for numerical stability; see e.g http://cs231n.github.io/linear-classify/
        e = np.exp(scores)
        p = e/np.sum(e,axis=0)
        for i in range(len(nodes)):
            node = nodes[i]
            node.rntnparams['softmax'] = p[:,i]

def fprop(model, tree):
    '''
    forward pass of the model on a tree. does this in a recursive fashion.

    stores important values in the nodes of the tree
    '''
    # Ws = model['Ws']
    # bs = model['bs']
    rfprop(model, tree.root)

    # ''' todo: vectorize this across whole batch'''
    # stacked_vecs = np.hstack([node.rntnparams['vec'][:,np.newaxis] for node in tree.nodes])
    # scores = np.dot(Ws, stacked_vecs) + bs[:,np.newaxis]
    # scores -= np.max(scores, axis=0)
    # e = np.exp(scores)
    # p = e/np.sum(e,axis=0)

    # for i in range(len(tree.nodes)):
    #     node = tree.nodes[i]
    #     node.rntnparams['softmax'] = p[:,i]

def rfprop(model, node):
    '''
    recursive function that traverses the tree and computes the hidden vectors at each step
    '''
    L = model['L']
    W = model['W']
    V = model['V']
    b = model['b']
    Ws = model['Ws']
    bs = model['bs']

    # find activation
    if model['activation'] == 'tanh':
        f = np.tanh
    elif model['activation'] == 'relu':
        f = lambda vec: np.maximum(0,vec)


    # leaf base case
    if node.leaf:
        if model['wordActivations']:
            node.rntnparams['vec'] = f(L[:,node.rntnparams['wIndex']]) # tanh activation on word vectors
        else:
            node.rntnparams['vec'] = L[:,node.rntnparams['wIndex']] # tanh activation on word vectors
    
    # else recurse down
    else:
        # fprop left and right
        # if not node.left.rntnparams['fprop']:
        rfprop(model, node.left)
        # else:
            # print "I dont think it should ever get here."
        # if not node.right.rntnparams['fprop']:
        rfprop(model, node.right)
        # else:
            # print "I dont think it should ever get here."

        # now left, right vecs should exist
        lvec = node.left.rntnparams['vec']
        rvec = node.right.rntnparams['vec']
        # stack the vectors for the matrix multiplications
        stacked = np.hstack([lvec,rvec])
        Wmul = np.dot(W, stacked)
        Vmul = np.tensordot(V, np.outer(stacked,stacked), ([1,2],[0,1])) # ******TODO: make sure that this works as i think it works/want it to work
        # now do the nonlinearity
        node.rntnparams['vec'] = f(Wmul + Vmul + b)
        
    # now go to softmax calcs
    # '''
    # TODO: vectorize this over the whole tree; should be easy enough to do, no more recursion here
    # '''
    # vec = node.rntnparams['vec'] #this should exist now
    # scores = np.dot(Ws, vec) + bs
    # scores -= np.max(scores) # for numerical stability; see e.g http://cs231n.github.io/linear-classify/
    # e = np.exp(scores)
    # p = e / np.sum(e)
    # node.rntnparams['softmax'] = p

    # node.rntnparams['fprop'] = True

def bprop(model, tree):
    '''
    backward pass of the model on a tree. accumulates gradients and returns them.
    '''
    # initialize gparams to zeros
    gparams = {k:np.zeros(model[k].shape) for k in model['update']}
    # recurse down the tree and accumulate gradients
    rbprop(model, tree.root, gparams)
    return gparams

def rbprop(model, node, gparams, dparent=None):
    ''' recursive worker function to accumulate gradients and store them in gparams.

    dparent is incoming errors from parents
    '''
    Ws = model['Ws']
    V = model['V']
    W = model['W']
    h = node.rntnparams['vec']
    dtop = node.rntnparams['dtop'] # should be here from the stepper, this is the top level derivatives

    # get derivative function
    if model['activation'] == 'tanh':
        fprime = lambda vec: 1 - vec**2
    elif model['activation'] == 'relu':
        fprime = lambda vec: np.array(vec > 0, np.int)
    # backproping means that the node will be ready for another fprop
    # node.rntnparams['fprop'] = False

    # backprop cost to Ws, bs
    gparams['Ws'] += np.outer(dtop, h)
    gparams['bs'] += dtop # if we used the bias notation trick this would be the same as above but the vec would just be 1s

    # now backprop to V,W,b
    dh = np.dot(Ws.T, dtop)
    # dh *= fprime(h) # now doing this after

    if dparent is not None:
        dh += dparent 

    if model['wordActivations']:
        # need to do *= fprime(h) on all
        dh *= fprime(h)
    else:
        # only should do *= fprime(h) on non-leafs
        if not node.leaf:
            dh *= fprime(h)


    # now we either continue down recursively or we're at a leaf node
    if node.leaf:
        # backprop to L matrix
        # this would be np.outer(dH, one_hot_encoding_of_vocab) which is equivalent to:
        gparams['L'][:,node.rntnparams['wIndex']] += dh
        

    else:
        # backprop to W, b
        stacked = np.hstack([node.left.rntnparams['vec'], node.right.rntnparams['vec']])
        # this is similart to softmax
        gparams['W'] += np.outer(dh, stacked)
        gparams['b'] += dh

        # backprop to V
        # little trickier. need each slice dV_i to be the dH_i * np.outer(stacked,stacked)
        # can do this by adding a new axis to np.outer(stacked,stacked) and then broadcasting the multiplication
        gparams['V'] += (np.outer(stacked,stacked)[:,:,np.newaxis] * dh).transpose()
       
        # now we need to send the error down to the children and recurse
        dstacked = np.dot(W.T, dh)
        Vtranspose = V.transpose((0,2,1)) # keep last axis same, transpose the 2dx2d matrices
        VplusVT = Vtranspose + V
        S = np.tensordot(VplusVT, np.outer(dh,stacked).T,([1,0],[0,1]))
        next_dparent = dstacked + S

        # next_dparent *= fprime(stacked)  (now doing this from the child)
                  
        rbprop(model,node.left, gparams, next_dparent[:W.shape[0]])
        rbprop(model,node.right,gparams,next_dparent[W.shape[0]:])



if __name__ == '__main__':
    from data import *
    import cPickle as pkl
    train = pkl.load(open('formatted/train.pkl'))

    import time
    mbsize=27
    minibatches = [train.trees[i:i+mbsize] for i in range(0,len(train.trees), mbsize)]
    rntn = initRNTN(gpu=True)
    tick = time.time()
    # rntn = initRNTN(gpu=True)
    
   
    for b in minibatches:
        batch_fprop(rntn, b)
    tock = time.time()
    print "time: %f" % (tock-tick)




