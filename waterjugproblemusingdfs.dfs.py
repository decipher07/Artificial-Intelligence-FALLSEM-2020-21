from anytree import Node, RenderTree

class State():
    def __init__( self, current_water_in_left_jug, current_water_in_right_jug, max_water_left_jug, max_water_right_jug, final_water_left, final_water_right):
        self.current_water_in_left_jug = current_water_in_left_jug
        self.current_water_in_right_jug = current_water_in_right_jug
        self.max_water_left_jug = max_water_left_jug
        self.max_water_right_jug = max_water_right_jug
        self.final_water_left = final_water_left
        self.final_water_right = final_water_right
        self.parent = None
        self.tree_node = Node(str(self))

    def is_goal(self):
        if ( (self.current_water_in_left_jug == self.final_water_left and self.current_water_in_right_jug == self.final_water_right) or (self.current_water_in_left_jug == self.final_water_right and self.current_water_in_right_jug == self.final_water_left )):
            return True 
        else:
            return False

    def is_valid(self):
        if ((self.current_water_in_left_jug <= self.max_water_left_jug and self.current_water_in_right_jug <= self.max_water_right_jug)):
            return True 
        else:
            return False
    
    def get_tree_node(self):
        return self.tree_node

    def set_tree_parent(self, parent):
        self.tree_node.parent = parent.get_tree_node()

    def __str__(self):
        return "(" + str(self.current_water_in_left_jug) + " , " + str(self.current_water_in_right_jug) + ")" + (" <- Goal State" if self.is_goal() else "")
    
    def __eq__(self, other):
        return self.current_water_in_left_jug == other.current_water_in_left_jug and self.current_water_in_right_jug == other.current_water_in_right_jug \
                   and self.max_water_left_jug == other.max_water_left_jug and self.max_water_right_jug == other.max_water_right_jug \
                   and self.final_water_right == other.final_water_right and self.final_water_left == other.final_water_left
    
    def __hash__(self):
        return hash((self.current_water_in_left_jug, self.current_water_in_right_jug, self.max_water_left_jug, self.max_water_right_jug, self.final_water_left, self.final_water_right))

def get_children(cur_state):
    children = [];
    # 1. Fill The First To Max and Second to Remain As It is
    new_state = State(cur_state.max_water_left_jug, cur_state.current_water_in_right_jug, cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right)
    if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)
    
    # 2. Fill the Second to Max and First to remain as it is
    new_state = State(cur_state.current_water_in_left_jug, cur_state.max_water_right_jug, cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right)
    if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)
    
    # 3. Empty the Left to 0
    new_state = State(0, cur_state.current_water_in_right_jug, cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right)
    if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)

    # 4. Empty the Right to 0
    new_state = State(cur_state.current_water_in_left_jug, 0,  cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right)
    if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)

    # 5. Pour Water from right into left until the left is full 
    x = cur_state.current_water_in_left_jug
    y = cur_state.current_water_in_right_jug
    max_left = cur_state.max_water_left_jug 
    max_right = cur_state.max_water_right_jug

    if ( x + y >= max_left and y > 0 ):
        new_state = State(max_left, y - (max_left - x ), cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right )
        if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)

    # 6. Pour water from left to right until the right is full 
    if ( x + y >= max_right and x > 0 ):
        new_state = State( x - (max_right - y ), max_right, cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right )
        if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)

    # 7. Pour all water from left to right
    if ( x + y <= max_right and x > 0 ):
        new_state = State( 0 , x + y , cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right )
        if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)

    # 8. Pour all water from right to left 
    if ( x + y <= max_left and y > 0 ):
        new_state = State( x + y , 0, cur_state.max_water_left_jug, cur_state.max_water_right_jug, cur_state.final_water_left, cur_state.final_water_right )
        if new_state.is_valid() and new_state != cur_state :
            new_state.parent = cur_state
            children.append(new_state)

    return children

def depth_first_search(initial_state):

    if initial_state.is_goal():
        return initial_state 
    
    stack = list()
    explored = set()

	# Add state to the stack
    stack.append(initial_state)
    
    while stack :
        # Explore all children in this state.
        state = stack.pop()

        # Until we find a goal or we reach the end of the queue
        if state.is_goal():
            return state
        
        explored.add(state)
        children = get_children(state)

        for child in children:
            if ( child not in explored ):
                # if child is not explored and not in stack then
                # Add the child to the stack
                # And set it's tree parent to current state
                stack.append(child)
                child.set_tree_parent(state)
    return None


def main():
    
    x = int(input("Please Enter the Volume of X Jug : "))
    y = int(input("Please Enter the Volume of Y Jug : "))
    z = int(input("Please Enter the Z litres of Water You want : "))
    
    # Initial State
    initial_state = State(0, 0, x, y, z, 0 )
    
    depth_first_search(initial_state)
    
    print("GRAPHS IS AS FOLLOWS : ")
    
    # Printing Graph
    for pre, fill, node in RenderTree(initial_state.get_tree_node()):
        print("%s%s" %(pre,node.name))


# if called from the command line, call main()
if __name__ == "__main__":
    main()