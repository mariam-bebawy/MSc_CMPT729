import torch
import torch.nn as nn
import torch.nn.functional as F

# the standard way to create a network in PyTorch is to define a class 
class DQN(nn.Module):
    # inherits from nn.Module

    # init function defines the layers of the network
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        # state_dim : size of input layer (states)
        # action_dim : size of output layer (actions)
        
        super(DQN, self).__init__()
    
        # initialize layers
        # in PyTorch the input layer is implicit so no need to define it
        # what we need to do it is to define the following layers
    
        self.fc1 = nn.Linear(state_dim, hidden_dim)     # defining hidden layer
        self.fc2 = nn.Linear(hidden_dim, action_dim)    # defining output layer

    # forward function defines the forward pass (calculations)
    def forward(self, x):
        # x : input to the network (the state)
        x = F.relu(self.fc1(x))
        # self.fc1 : hidden layer
        # self.fc1(x) : pass the input to the hidden layer
        # F.relu : activation function
        x = self.fc2(x)
        # self.fc2 : output layer
        # input is the output of the hidden layer
        # self.fc2(x) : pass the input to the output layer
        return x
    
# we should now do a simple test to make sure the network works 
# so define a main function

# ctrl+i : prompt codeium command
def main():
    """
    test function
    """
    state_dim = 12
    action_dim = 2

    # create an instance of the network
    net = DQN(state_dim=state_dim, action_dim=action_dim)
    state = torch.randn(1, state_dim)           # create a random state (input)
    print(f"state : {state}, shape : {state.shape}")
    # shape : torch.Size([1, 12])
    # why is the shape [1, 12] ? why not [12] ?
    # second dimension is state dimension (12)
    # first dimension is batch size (1)
    # meaning that we can send in a whole batch of states and PyTorch will handle them all at once
    # that is much more efficient than sending them one state at a time

    # try:
    # state = torch.randn(10, state_dim)

    q_values = net(state)                       # pass the state to the network and get the output
    print(f"output : {q_values}")               # print the output

if __name__ == "__main__":
    main()
