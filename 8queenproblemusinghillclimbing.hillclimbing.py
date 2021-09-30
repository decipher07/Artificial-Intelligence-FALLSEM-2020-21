import numpy as np
from anytree import Node, RenderTree
import random

def attacks_from_queen ( current_row_filled_list ):
    attack = 0 
    for i in range ( 8 ):
        for j in range ( i + 1 , 8 ):
            if current_row_filled_list[i] == current_row_filled_list[j]:
                attack += 1 
            elif abs( i - j ) == abs(current_row_filled_list[i] - current_row_filled_list[j]):
                attack += 1
        
    return attack 


class State():
    
    def __init__ ( self, row_filled_list, row_filled_matrix, objective_value ):
        self.row_filled_list = row_filled_list 
        self.row_filled_matrix = row_filled_matrix
        self.objective_value = objective_value
        self.parent = None 
        self.tree_node = Node(str(self))
        
    def objective_goal ( self ):
        current_row_filled_list = self.row_filled_list
        return attacks_from_queen(current_row_filled_list)
        
    def is_goal ( self ):
        current_row_filled_list = self.row_filled_list
        if ( attacks_from_queen(current_row_filled_list) == 0 ):
            return True 
        else:
            return False
        
    def get_tree_node ( self ):
        return self.tree_node 

    def set_tree_parent ( self , parent ):
        self.tree_node.parent = parent.get_tree_node()
    
    def __str__ (self):
        return "(" + str ( self.objective_value ) + " " + str(self.row_filled_list) + " )" + "-----------------> " + str(self.objective_goal()) 
    
    def __eq__ (self, other ):
        return self.row_filled_list == other.row_filled_list and self.objective_value == other.objective_value
    
    def __hash__ (self):
        converted_list_string = str(self.row_filled_list)
        converted_matrix_string = str(self.row_filled_matrix)
        total_string = converted_list_string + converted_matrix_string
        return hash((total_string))
    
    def __lt__ (self, other):
        return self.objective_value < other.objective_value
    
def get_children ( cur_state ):
    children = []
    
    current_list_neighbours = cur_state.row_filled_list.copy()
    
    temp_neighbours_matrix = cur_state.row_filled_matrix.copy() 
    temp_neighbours_list = cur_state.row_filled_list.copy()
    
    for i in range (8):
        for j in range (8):
            
            # Ignoring the Row For Sequences
            if j != current_list_neighbours[i] :
                
                temp_neighbours_list[i] = j 
                temp_neighbours_matrix[temp_neighbours_list[i]][i] = 1
                temp_neighbours_matrix[current_list_neighbours[i]][i] = 0
                temp_objective_value = attacks_from_queen(temp_neighbours_list)
                new_state = State(temp_neighbours_list, temp_neighbours_matrix, temp_objective_value)
                
                if new_state != cur_state:
                    new_state.parent = cur_state
                    children.append(new_state)
                
                # Backtracking
                temp_neighbours_list[i] = current_list_neighbours[i]
                temp_neighbours_matrix[temp_neighbours_list[i]][i] = 0
                temp_neighbours_matrix[current_list_neighbours[i]][i] = 1
     
    return children

def hill_climbing_search ( initial_state ):
    
    if ( initial_state.is_goal() ):
        return initial_state 

    search_space = list()
    explored = set()
    
    # Add State to the Space
    search_space.append(initial_state)
    
    while search_space :
        # Explore all children in Search_space
        state = search_space.pop(0)
        
        # Until we find a goal or we reach the end of the search_space
        if state.is_goal():
            return state 
        
        explored.add(state)
        children = get_children(state)
        
        current_objective_function = state.objective_value
        
        for child in children :
            if ( child not in explored ):
                if ( child.objective_value < current_objective_function ):
                    search_space.append(child)
                    child.set_tree_parent(state)
                    current_objective_function = child.objective_value
                    break
    
       
        search_space.sort()
    return None 
        
def main():
    
    # Initial State 
    row_filled_list = []
    matrix = np.zeros((8,8))
    
    for j in range(8):
        a = random.randint(0,7)
        row_filled_list.append(a)
    
    for i in range(8):
        matrix[row_filled_list[i]][i] = 1
    
    initial_objective_value = attacks_from_queen(row_filled_list)
    
    print (row_filled_list)
    print (initial_objective_value)
    
    initial_state = State(row_filled_list, matrix, initial_objective_value )
    
    hill_climbing_search(initial_state)
    
    print("GRAPHS IS AS FOLLOWS : ")	
    # Printing Graph
    for pre, fill, node in RenderTree(initial_state.get_tree_node()):
		    print("%s%s" %(pre,node.name))

# if called from the command line, call main()
if __name__ == "__main__":
    main()