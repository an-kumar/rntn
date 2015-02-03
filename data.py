'''
Ankit Kumar
ankitk@stanford.edu

util functions/classes to deal with the PTB formatted trees that comprise the dataset.

note that i'm taking no effort to optimize the speed here because everything will only be needed to be done once.
'''

def find_index_of_close_bracket(rep, index_of_open_bracket=0):
	'''
	finds the index of the close bracket that matches the open bracket found at rep[index_of_open_bracket]def
	'''
	if rep[index_of_open_bracket] != '(':
		print index_of_open_bracket
		raise Exception("find_index_of_close_bracket didn't get a valid index of open bracket")
	obrackets = 0
	cbrackets = 0
	found = 0
	for j in range(len(rep[index_of_open_bracket:])):
		i = j+index_of_open_bracket
		char = rep[i]
		if char == '(':
			obrackets += 1
		elif char == ')':
			cbrackets += 1
		if obrackets == cbrackets:
			found = -1 # break when we find equality
			break
	if found == 0:
		raise Exception("find_index_of_close_bracket never got to equality of brackets")
	else:
		return i


def is_well_formed_PTB_rep(rep):
	'''
	checks some things regarding well-formed PTB representation
	'''
	# check that the first char is a (
	if rep[0] != '(':
		print "representation didn't start with ("
		return False

	# check that the second char is a number
	if not rep[1].isdigit():
		print "representation didn't have a label"
		return False

	# check that this is the representation of only one node; i.e, make sure the close bracket for the first open bracket is the last close bracket
	# this also makes sure that the rep ends with )
	index_of_close_bracket = find_index_of_close_bracket(rep)
	if index_of_close_bracket != (len(rep) - 1):
		print index_of_close_bracket
		print "representation wasn't of only one node"
		return False

	return True


def is_leaf(rep):
	'''
	checks if the representation is of a leaf node or not
	'''
	return rep.count('(') == 1

def find_left_right_reps(rep):
	'''
	assumes rep is the full representation of a node with two children
	'''
	# get first child's open bracket index (1: strips first bracket)
	first_child_open_bracket = rep[1:].find('(') + 1
	first_child_close_bracket = find_index_of_close_bracket(rep, first_child_open_bracket)

	second_child_open_bracket = rep[first_child_close_bracket+1:].find('(') + first_child_close_bracket+1
	second_child_close_bracket = find_index_of_close_bracket(rep, second_child_open_bracket)

	return rep[first_child_open_bracket:first_child_close_bracket+1], rep[second_child_open_bracket:second_child_close_bracket+1]

class PTBNode(object):
	'''
	a node of a PTB tree
	'''
	def __init__(self, rep, parent=None):
		'''
		string is expected to be a full string representation of a PTB node; i.e, is of the format (# (...)(...)) where # is the label and (...)(...) represents the two child nodes (if the exist)

		this will be called recursively from the top of the tree.
		'''
		rep = rep.strip() #strip newlines
		assert is_well_formed_PTB_rep(rep), "got a malformed PTB representation"
		self.parent=parent # none means root
		self.label = int(rep[1])
		self.leaf = is_leaf(rep)
		
		self.rntnparams = {'wIndex':None, 'vec':None, 'fprop':False} #rntnparams will be used by the RNTN to store info
		
		
		if not self.leaf:
			left_rep, right_rep = find_left_right_reps(rep)
			self.left = PTBNode(left_rep, self)
			self.right = PTBNode(right_rep, self)
			self.word = None
		else:
			self.left = None
			self.right = None
			self.word = rep.split()[1][:-1] # hacky but works

		

			

class PTBTree(object):
	'''
	whole PTB tree representation
	'''
	def __init__(self,rep):
		self.root = PTBNode(rep)
		self.leaves = []
		self.nodes = []
		self.traverse_tree_find_leaves()

	def traverse_tree_find_leaves(self):
		'''
		recurse down tree, add leaves to self.leaves and all nodes to self.nodes

		doesn't consider edge cases really (maybe add if needed)
		'''
		start = self.root
		self.recurse_find_leaves(start)

	def recurse_find_leaves(self,node):
		if node.leaf:
			self.leaves.append(node)
			self.nodes.append(node)
		else:
			self.nodes.append(node)
			self.recurse_find_leaves(node.left) # go left
			self.recurse_find_leaves(node.right) # go right

	def clear_rntn(self):
		for node in self.nodes:
			node.rntnparams['vec'] = None


class PTBDataset(object):
	'''
	whole dataset representation
	'''
	def __init__(self, filepath):
		self.trees = []
		f = open(filepath)
		for rep in f:
			self.trees.append(PTBTree(rep.strip().lower())) #lowercase tokens




if __name__ == '__main__':
	# ds = PTBDataset('trees/train.txt')

	### script to pickle some datasets & vocab to index/index to vocab maps:
	
	import cPickle as pkl
	# first do train set to build vocab
	ds = PTBDataset('trees/train.txt')
	words = set()
	for tree in ds.trees:
		for node in tree.leaves:
			words.add(node.word)
	i2v = list(words) + ['UNK']
	v2i = {i2v[i]:i for i in range(len(i2v))}

	# now go back through and set the word indices
	for tree in ds.trees:
		for node in tree.leaves:
			node.rntnparams['wIndex'] = v2i[node.word]
	
	# now go back and add node.rntnparams['wIndices']
	# this is sort of hacky but works.
	def rFindIndices(node,leaves):
		if node.leaf:
			leaves.append(node.rntnparams['wIndex'])
		else:
			rFindIndices(node.left,leaves)
			rFindIndices(node.right,leaves)

	def find_wIndices_from_node(node):
		leaves = []
		rFindIndices(node,leaves)
		return leaves




	for tree in ds.trees:
		for node in tree.nodes:
			node.rntnparams['wIndices'] = find_wIndices_from_node(node)

	pkl.dump(ds, open('formatted/train.pkl','w'))
	# now do the others 
	for dspath in 'dev,test'.split(','):
		ds = PTBDataset('trees/%s.txt' % dspath)
		# save the words of the train set
		for tree in ds.trees:
			for node in tree.leaves:
				if node.word in v2i:
					node.rntnparams['wIndex'] = v2i[node.word]
				else:
					node.rntnparams['wIndex'] = v2i['UNK']
		# now go and add the indices
		
		for tree in ds.trees:
			for node in tree.nodes:
				node.rntnparams['wIndices'] = find_wIndices_from_node(node)

		pkl.dump(ds, open('formatted/%s.pkl' % dspath,'w'))

	# 

	

		

	# # build index to vocab mapping in case it's needed
	# v2i = {i2v[i]:i for i in range(len(i2v))}
	# pkl.dump(i2v, open('formatted/i2v.pkl','w'))
	# pkl.dump(v2i, open('formatted/v2i.pkl','w'))
	


