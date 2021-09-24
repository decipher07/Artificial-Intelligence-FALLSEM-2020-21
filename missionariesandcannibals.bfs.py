import math
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

class State():
	def __init__(self, cannibals_in_left, missionaries_in_left, boat, cannibals_in_right, missionaries_in_right):
		self.cannibals_in_left = cannibals_in_left
		self.missionaries_in_left = missionaries_in_left
		self.boat = boat
		self.cannibals_in_right = cannibals_in_right
		self.missionaries_in_right = missionaries_in_right
		self.parent = None
		self.tree_node = Node(str(self))

	def is_goal(self):
		if self.cannibals_in_left == 0 and self.missionaries_in_left == 0:
			return True
		else:
			return False

	def is_valid(self):
		if self.missionaries_in_left >= 0 and self.missionaries_in_right >= 0 and self.cannibals_in_left >= 0 and self.cannibals_in_right >= 0 \
        	and (self.missionaries_in_left == 0 or self.missionaries_in_left >= self.cannibals_in_left) \
			and (self.missionaries_in_right == 0 or self.missionaries_in_right >= self.cannibals_in_right):
			return True
		else:
			return False

	def get_tree_node(self):
		return self.tree_node

	def set_tree_parent(self, parent):
		self.tree_node.parent = parent.get_tree_node()

	def __str__(self):
		return "("+ str(self.missionaries_in_left) + "," + str(self.cannibals_in_left) + "," + str(self.missionaries_in_right) + "," + str(self.cannibals_in_right) + "," + (self.boat) + ")" + (" <- Goal State" if self.is_goal() else "")

	def __eq__(self, other):
		return self.cannibals_in_left == other.cannibals_in_left and self.missionaries_in_left == other.missionaries_in_left \
                   and self.boat == other.boat and self.cannibals_in_right == other.cannibals_in_right \
                   and self.missionaries_in_right == other.missionaries_in_right

	def __hash__(self):
		return hash((self.cannibals_in_left, self.missionaries_in_left, self.boat, self.cannibals_in_right, self.missionaries_in_right))

def get_children(cur_state):
	children = [];
	
	if cur_state.boat == 'left':
		## Two missionaries cross left to right.
		new_state = State(cur_state.cannibals_in_left, cur_state.missionaries_in_left - 2, 'right', cur_state.cannibals_in_right, cur_state.missionaries_in_right + 2)
		if new_state.is_valid() and new_state != cur_state :
			new_state.parent= cur_state
			children.append(new_state)
		
		## Two cannibals cross left to right.
		new_state = State(cur_state.cannibals_in_left - 2, cur_state.missionaries_in_left, 'right', cur_state.cannibals_in_right + 2, cur_state.missionaries_in_right)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
		
		## One missionary and one cannibal cross left to right.
		new_state = State(cur_state.cannibals_in_left - 1, cur_state.missionaries_in_left - 1, 'right', cur_state.cannibals_in_right + 1, cur_state.missionaries_in_right + 1)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
		
		## One missionary crosses left to right.
		new_state = State(cur_state.cannibals_in_left, cur_state.missionaries_in_left - 1, 'right', cur_state.cannibals_in_right, cur_state.missionaries_in_right + 1)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
		
		## One cannibal crosses left to right.
		new_state = State(cur_state.cannibals_in_left - 1, cur_state.missionaries_in_left, 'right', cur_state.cannibals_in_right + 1, cur_state.missionaries_in_right)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
	else:
		## Two missionaries cross right to left.
		new_state = State(cur_state.cannibals_in_left, cur_state.missionaries_in_left + 2, 'left', cur_state.cannibals_in_right, cur_state.missionaries_in_right - 2)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
		
		## Two cannibals cross right to left.
		new_state = State(cur_state.cannibals_in_left + 2, cur_state.missionaries_in_left, 'left', cur_state.cannibals_in_right - 2, cur_state.missionaries_in_right)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
		
		## One missionary and one cannibal cross right to left.
		new_state = State(cur_state.cannibals_in_left + 1, cur_state.missionaries_in_left + 1, 'left', cur_state.cannibals_in_right - 1, cur_state.missionaries_in_right - 1)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)
		
		## One missionary crosses right to left.
		new_state = State(cur_state.cannibals_in_left, cur_state.missionaries_in_left + 1, 'left', cur_state.cannibals_in_right, cur_state.missionaries_in_right - 1)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent= cur_state
			children.append(new_state)

		## One cannibal crosses right to left.
		new_state = State(cur_state.cannibals_in_left + 1, cur_state.missionaries_in_left, 'left', cur_state.cannibals_in_right - 1, cur_state.missionaries_in_right)
		if new_state.is_valid() and new_state != cur_state:
			new_state.parent = cur_state
			children.append(new_state)

	return children

def breadth_first_search(initial_state):
	if initial_state.is_goal():
		return initial_state

	queue = list()
	explored = set()

	# Add state to the queue
	queue.append(initial_state)

	while queue:
		# Explore all children in this state.
		state = queue.pop(0)

		# Until we find a goal or we reach the end of the queue
		if state.is_goal():
			return state

		explored.add(state)
		children = get_children(state)

		for child in children:
			if (child not in explored) or (child not in queue):
				# If child is not explored and not in queue then
				# Add the child to the queue
				# And set it's tree parent to current state
				queue.append(child)
				child.set_tree_parent(state)
	return None

def main():
	# Initial State
	initial_state = State(3,3,'left',0,0)

	breadth_first_search(initial_state)

	print("GRAPHS IS AS FOLLOWS : ")
	
	# Printing Graph
	for pre, fill, node in RenderTree(initial_state.get_tree_node()):
		print("%s%s" %(pre,node.name))

	

# if called from the command line, call main()
if __name__ == "__main__":
    main()