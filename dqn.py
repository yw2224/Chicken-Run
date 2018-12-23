import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_chicken_run as game
import random
import numpy as np
from collections import deque


import json

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

GAME = 'chicken' # the name of the game being played for log files
# CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 80, 80
img_channels = 4


def CNN():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding='same', input_shape = (img_rows, img_cols, img_channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    return model


def process_image(x_t):
    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t, (80, 80))
    x_t = skimage.exposure.rescale_intensity(x_t, out_range=(0, 255))  # rescale pixel value to (0, 255)
    x_t = x_t / 255.0

    return x_t



def DQN(model, arg):
    game_state = game.GameState()
    D = deque()

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing) # state, reward, terminal = True/False

    x_t = process_image(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # 80 * 80 * 4
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1 * 80 * 80 * 4, state feature, input for Q(s)

    if arg['mode'] == 'Train':
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON
    else:
        OBSERVE = sys.maxsize    # keep observing, no train
        epsilon = FINAL_EPSILON
        print ("Loading weight...")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully!")

    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
            else:
                q = model.predict(s_t) # input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
            a_t[action_index] = 1 # choose an action (jump or not) using epsilon-greedy policy

        if epsilon > FINAL_EPSILON and t > OBSERVE: # reduce epsilon gradually
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        x_t1 = process_image(x_t1_colored)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) # 1 * 80 * 80 * 1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) # append the latest image to the front, new state s'

        # save experience
        D.append((s_t, action_index, r_t, s_t1, terminal))

        if len(D) > REPLAY_MEMORY: # discard old memory
            D.popleft()

        # training
        if t > OBSERVE:
            # sampling
            minibatch = random.sample(D, BATCH)
            state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
            state_t = np.concatenate(state_t)
            state_t1 = np.concatenate(state_t1)

            targets = model.predict(state_t) # Q
            Q_sa = model.predict(state_t1)
            targets[range(BATCH), action_t] = reward_t + GAMMA * np.max(Q_sa, axis=1) * np.invert(terminal) # Q'

            loss += model.train_on_batch(state_t, targets) # input and output, minimize |model(state_t) - targets|

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Saving model...")
            model.save_weights("model.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

        # print info
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")


def playGame(args):
    model = CNN()
    if args['mode'] == 'Play':
        game_state = game.GameState()
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        while True:
            game_state.frame_step(do_nothing)
    else:
        DQN(model,args)

def main():
    parser = argparse.ArgumentParser(description='Chicken Run')
    parser.add_argument('-m','--mode', help='Train / Run / Play', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()
