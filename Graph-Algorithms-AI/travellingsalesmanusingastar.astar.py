from anytree import Node, RenderTree
from anytree.exporter import DotExporter

def objective_composition_function ( tsp_solution_list, tsp_matrix ):
    # Considering the Estimated Total Cost Of Path to Goal from the Sequence to 50
    # f(n) = g(n) + h(n)
    # g(n) = 50 -> Assume
    # h(n) = cost so far.
    
    route_length = 50
    for i in range(len(tsp_solution_list)):
        route_length += tsp_matrix[tsp_solution_list[i - 1]][tsp_solution_list[i]]
    return route_length

class State():
    def __init__ (self, tsp_matrix, tsp_solution_list, objective_value, is_goal ):
        self.tsp_matrix = tsp_matrix 
        self.tsp_solution_list = tsp_solution_list
        self.objective_value = objective_value
        self.is_goal = is_goal
        self.parent = None
        self.tree_node = Node(str(self))
    
    def get_tree_node(self):
        return self.tree_node

    def set_tree_parent( self, parent ):
        self.tree_node.parent = parent.get_tree_node()
    
    def __str__(self):
        return "[" + str( self.tsp_solution_list ) + "] --> " + str(self.objective_value)
    
    def __eq__(self, other):
        return self.tsp_matrix == other.tsp_matrix and self.tsp_solution_list == other.tsp_solution_list and self.objective_value == other.objective_value
    
    def __hash__(self):
        convert_list_string = str(self.tsp_solution_list)
        convert_matrix_string = str(self.tsp_matrix)
        total_string = convert_list_string + convert_matrix_string
        return hash(total_string)
    
    def __lt__ (self, other):
        return self.objective_value < other.objective_value

def get_children(cur_state):
    children = [];
    
    current_solution_list = cur_state.tsp_solution_list
    tsp_matrix = cur_state.tsp_matrix

    for i in range (len(current_solution_list)):
        for j in range ( i + 1 , len(current_solution_list)):
            temp_neighbour = current_solution_list.copy()
            temp_neighbour[i] = current_solution_list[j]
            temp_neighbour[j] = current_solution_list[i]
            
            temp_objective_value = objective_composition_function (temp_neighbour, tsp_matrix)
            new_state = State(tsp_matrix, temp_neighbour, temp_objective_value, True)
            
            if new_state != cur_state:
                new_state.parent = cur_state
                children.append(new_state)
    
    return children

def a_star_algorithm ( initial_state ):
    
    search_space = list()
    explored = set()
    
    # Add State to the Search Space
    search_space.append(initial_state)
    
    value_of_store_state = 10000000000000
    store_the_goal_state = initial_state
    
    while search_space :
        # Exploring all children in search_space
        state = search_space.pop(0)
        
        if ( value_of_store_state > state.objective_value ):
            store_the_goal_state.is_goal = False
            value_of_store_state = state.objective_value
            store_the_goal_state = state
            store_the_goal_state.is_goal = True
        
        explored.add(state)
        children = get_children(state)
        
        for child in children :
            if ( child not in explored ) and ( child not in search_space ):
                search_space.append(child)
                child.set_tree_parent(state)
        
        search_space.sort()
    
    print("\nThe Goal or the Optimium Distance is : " + str(value_of_store_state) + " with the Configuration : " + str(store_the_goal_state.tsp_solution_list) + "\n")

    return None

def main():
    # Initial State
    tsp_matrix = [
        [0, 400, 500, 300],[400, 0, 300, 500],[500, 300, 0, 400],[ 300, 500, 400, 0]
    ]
    
    initial_solution = [0, 2, 1, 3]
    
    initial_objective_value = objective_composition_function(initial_solution, tsp_matrix)
    
    initial_state = State(tsp_matrix, initial_solution, initial_objective_value, False )
    
    a_star_algorithm(initial_state)
    
    print("GRAPHS IS AS FOLLOWS : ")	
    # Printing Graph
    for pre, fill, node in RenderTree(initial_state.get_tree_node()):
		    print("%s%s" %(pre,node.name))
    
# if called from the command line, call main()
if __name__ == "__main__":
    main()