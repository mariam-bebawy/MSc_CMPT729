# when training a RL algorithm ourselves we should not try to train a complicated env to begin with
# b/c we don't know if there are bugs in our code or if the env just takes a long time to train
# during coding we wanna test on a simpler env

import flappy_bird_gymnasium
import gymnasium

import itertools
# import yaml for hyperparameters
import yaml

# import DQN network that we built in dqn.py
from dqn import DQN

# import the replay memory
from experience_replay import ReplayMemory

# since we're using PyTorch we wanna see if we can use the GPU processing
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda" if torch.cuda.is_available() else "cpu"

# any game compatible with gymnasium will follow this general pattern
class Agent:

    def __init__(self, hyperparameter_set):
        with open("hyperparameters.yml", "r") as f:
            all_hyperparameters_sets = yaml.safe_load(f)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]
        
        # now we need to save the loaded hyperparameters
        self.replay_memory_size = hyperparameters["replay_memory_size"]     # size of the replay memory
        self.mini_batch_size = hyperparameters["mini_batch_size"]           # size of the training dataset sampled from the replay memory
        self.epsilon_init = hyperparameters["epsilon_start"]                # 1 = 100% random actions
        self.epsilon_decay = hyperparameters["epsilon_decay"]               # epsilon decay rate
        self.epsilon_min = hyperparameters["epsilon_min"]                   # minimum epsilon value

    # will use the run function for training and running tests after training
    def run(self, is_training=True, render=False):

        # create instance of environment
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=True)
        # env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        # declare policy network
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_net = DQN(num_states, num_actions).to(device)
        # policy_net = DQN(num_states, num_actions).to_device(device)
        # whatever the device is (cpu or gpu) we can the network to the device for processing

        rewards_per_episode = []
        if is_training:
            # create instance of replay memory
            memory = ReplayMemory(maxlen=self.replay_memory_size)
        
        # for episode in range(1000):           # to train for 1000 episodes
        for episode in itertools.count():     # to train indefinitely
            # in this case we choose to train for an infinite amount of episodes and manually stop when satisfied

            # initialize env
            state, _ = env.reset()
            terminated = False

            # in the beginning of each episode we want to reset the environment 
            # and start counting the reward
            episode_reward = 0.0

            while not terminated:
                # this loop is considered one episode
                # in order to train efficiently we need multiple episodes
                # so wrap this loop into another episode-loop

                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample()      # get a random action
                # right now we're just choosing an action randomly
                # so we need to implement the episilon-greedy policy (algorithm) here
                # before we do that we need to import some hyperparameters ==> the .yml file

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)     # pass action to step function for execution
                # obs : (observation) what the next state is
                # reward : how much we got from the last action
                # terminated : if the bird hit the ground or hit a pipe
                # _ : if the game ended on a time-basis manner
                # info : extra infor that could be used for debugging or smth
                
                # accumulate reward
                episode_reward += reward
                
                # after taking an action we want to save it to the memory if we are in training mode
                if is_training:
                    # add experience to replay memory
                    memory.append((state, action, new_state, reward, terminated))
                
                # move to new state to keep track
                state = new_state
            rewards_per_episode.append(episode_reward)

        # env.close()         # comment if training indefinitely b/c we don't need to close the env