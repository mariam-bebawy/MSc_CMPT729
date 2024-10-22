# when training a RL algorithm ourselves we should not try to train a complicated env to begin with
# b/c we don't know if there are bugs in our code or if the env just takes a long time to train
# during coding we wanna test on a simpler env

import flappy_bird_gymnasium
import gymnasium

# any game compatible with gymnasium will follow this general pattern

# create instance of environment
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
# env = gymnasium.make("CartPole-v1", render_mode="human")

# initialize env
obs, _ = env.reset()
while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()      # get a random action

    # Processing:
    obs, reward, terminated, _, info = env.step(action)     # pass action to step function for execution
    # obs : (observation) what the next state is
    # reward : how much we got from the last action
    # terminated : if the bird hit the ground or hit a pipe
    # _ : if the game ended on a time-basis manner
    # info : extra infor that could be used for debugging or smth
    
    # Checking if the player is still alive
    if terminated:
        break

env.close()