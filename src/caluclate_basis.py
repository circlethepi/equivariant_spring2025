import torch

# global: collections of functions included in groups 
# TODO: is there a library for this? (probably)


class GroupActions:

    def __init__(self):

        self.actions = [] # list(functions)

        return
    
    def add_action(self, new_action):
        if new_action not in self.actions:
            self.actions.append(new_action)
    
    def calculate_group_basis(self, function:torch.tensor):
        """
        calculates group basis for finite groups following approach of Finzi et
        al. 2021:

        :param function: torch.tensor - function to compute basis for
        """
        # get basis for function
        # TODO: figure out if the tensor (more than 2D way will work generally)


        C_list = [action(function) for action in self.actions]
        

        basis = None

        return basis

# group averaging:
## need 1) all actions in the group 2) basis for the function
## for linear functions this is good

def make_matrix_basis(mat):
    """
    make basis for R^nxm matrix
    """
    dims = mat.size()

    basis = None

    return basis

# option 1: group averaging over the weights/function itself
# (what Wilson has done before)


# option 2: caluculating equivariant basis, then learning weights to project
# the model into the equivariant basis


