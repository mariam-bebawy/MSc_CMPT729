# to train a deep network we need to send in a lot of examples in order for the network to generalize the pattern and learn from it
# we cannot show it one instance of a certain situation
# we need to show it the same or similar instances over and over again before it can learn
# to overcome this challenge we use experience replay

# experience is composed of (state, action, new_state, reward, terminated)
# we take this combination and save it into memory (python deque)
# a deque is a double ended list (queue)
# we keep adding experiences to the front 
# it's gonna keep pushing and eventually the deque will be full and will start purging the old stuff
# in other words it's first in first out
# this way we will never run out of memory even if we train for a long time

# define memory for experience replay
from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed=None):
        # maxlen : maximum length of the memory to initialize the deque
        # seed : optional seed for reproducibility (controlling randomness)
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        # transition : (state, action, new_state, reward, terminated)
        self.memory.append(transition)      # appending experience to the memory

    def sample(self, batch_size):
        # randomly sampling experiences from the memory
        # batch_size : number of experiences to sample
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)             # return the length of the memory