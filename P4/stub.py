# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import copy
import random
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from SwingyMonkey import SwingyMonkey




class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.two_ago = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = .5
        self.sarsa = []
        self.counter = 0
        self.game = 0
        self.gravity = 0
        self.posReward = False
        self.lastPos = 0
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=7, kernel_initializer='normal', activation='relu'))
        # self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(2, activation='sigmoid'))
        self.model.compile(loss='mse', optimizer='adam', metrics=['mae'])


    def reset(self):
        self.game += 1

        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        a,b,c,d,_ = self.sarsa[-1]
        self.sarsa[-1] = (a,b,c,d,1)
        if len(self.sarsa) > 512:
            self.sarsa = self.sarsa[-512:]
        if self.posReward:
            self.lastPos = self.game
        elif self.game - self.lastPos > 5 and self.epsilon < .1:
            self.epsilon = .3
            self.sarsa = []
        self.posReward = False


    def state_RL(self, state):
        top_dist = state['tree']['top'] - state['monkey']['top']
        bot_dist = state['tree']['bot'] - state['monkey']['bot']

        new_state = np.array([[state['monkey']['top']], [state['monkey']['bot']], [top_dist], [bot_dist], [state['tree']['dist']], [state['monkey']['vel']], [self.gravity]])

        return new_state




    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        new_state = self.state_RL(state)
        model = self.model
        count = self.counter


        epsilon = self.epsilon/(1+count*.000005)
        self.epsilon = epsilon
        # print(self.epsilon)

        if np.random.rand() < epsilon:
            # a = np.random.randint(0,2)
            a = 0
        else:
            a = np.argmax(model.predict(new_state.T))

        if self.last_action == a and a == 0:
            self.gravity = self.last_state['monkey']['vel'] - state['monkey']['vel']

        if self.gravity == 4:
            self.gravity = 0
        self.last_action = a
        self.two_ago = copy.copy(self.last_state)
        self.last_state  = state
        self.counter += 1
        return self.last_action

    def reward_callback(self, reward):
        last_reward = 0
        state = self.last_state
        old_state = self.two_ago

        ## Calculate Reward ##
        if state['monkey']['bot'] + state['monkey']['vel'] < 0 or state['monkey']['top'] + state['monkey']['vel'] > 400:
            last_reward = -10

        elif state['tree']['dist'] <= 270 and state['tree']['dist'] >= 150:
            if state['monkey']['top'] + state['monkey']['vel'] >= state['tree']['top'] or state['monkey']['bot'] + state['monkey']['vel'] >= state['tree']['bot']:
                last_reward = -5
            else:
                last_reward = 1
                # print self.counter, self.lastPos
                self.posReward = True

        self.last_reward = reward

        if old_state == None:
            return



        ## Update NN ##
        model = self.model
        old_state = self.state_RL(old_state).T
        last_state = self.state_RL(state).T
        Qvalues = model.predict(old_state)
        Q_val = self.last_reward + .9*(np.max(model.predict(last_state)))
        Qvalues[0][self.last_action] = Q_val
        model.fit(old_state, Qvalues, epochs = 1, verbose = 0)
        last_step = 0

        ## Add SARSA entries ##
        self.sarsa.append((old_state, self.last_action, last_state, reward, last_step))


    def long_learn(self):
        size = min(int(len(self.sarsa) / 4), 128)
        Q_matrix = np.empty((1,2))
        Q_states = np.empty((1,7))
        for old_state, action, last_state, reward, last_step in random.sample(self.sarsa, size):
            model = self.model
            Qvalues = model.predict(old_state)
            if last_step == 1:
                # print "hit"
                Q_val = reward
            else:
                Q_val = reward + .9*(np.max(model.predict(last_state)))
            Qvalues[0][action] = Q_val
            model.fit(old_state, Qvalues, epochs = 1, verbose = 0)

        # if self.game % 10 == 0:
        #     for old_state, action, last_state, reward, last_step in self.sarsa:
        #         Q_states = np.hstack((Q_states, old_state))
        #         Qvalues = model.predict(old_state)
        #         if last_step == 1:
        #             # print "hit"
        #             Q_val = reward
        #         else:
        #             Q_val = reward + .9*(np.max(model.predict(last_state)))
        #         Qvalues[0][action] = Q_val
        #         Q_matrix = np.hstack((Q_matrix, Qvalues))
        #     model.fit(np.array(Q_states), np.array(Q_matrix), epochs=100, verbose = 1)




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

        # Long term learning
        learner.long_learn()

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
	run_games(agent, hist, 5000, 1)

	# Save history.
	np.save('hist',np.array(hist))
