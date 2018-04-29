# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

    def state_RL(self):
        state = self.last_state
        top_dist = state['tree']['top'] - state['monkey']['top']
        bot_dist = state['tree']['bot'] - state['monkey']['bot']
        return (top_dist, bot_dist, state['tree']['dist'], state['monkey']['vel'])


def eps_greedy_q_learning_with_table(env, num_episodes=500):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table



    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        last_reward = 0
        state = self.last_state

        if state['monkey']['bot'] + state['monkey']['vel'] < 0 or state['monkey']['top'] + state['monkey']['vel'] > 400:
            last_reward = -10

        elif state['tree']['dist'] == 0:
            if state['monkey']['top'] >= state['tree']['top'] or state['monkey']['bot'] >= state['tree']['bot']:
                last_reward = -5
            else:
                last_reward = 1

        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games.
	run_games(agent, hist, 20, 10)

	# Save history.
	np.save('hist',np.array(hist))

# Ali is Dumb
