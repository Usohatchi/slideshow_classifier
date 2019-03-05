# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from collections import deque, Counter
import numpy as np
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow import keras

from src.game import init, step, step_index, init_index

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

def read_file():
    # Read file
    #with open("c_memorable_moments.txt") as f:
    #with open("a_example.txt") as f:
    with open("b_lovely_landscapes.txt") as f:
        photos = f.readlines()
        photos = [x.strip() for x in photos]

    # Turn each photo into an list of its values
    photos.pop(0)
    photos = [x.split() for x in photos]

    # List of only the tags for each photo
    tags = [x[2:] for x in photos]
    tags = [item for sublist in tags for item in sublist]

    enc_photos = []
    photos_as_tags = [x[2:] for x in photos]

    # Get percentage distribution of letters
    letters_dist = Counter([letter for photo in tags for tag in photo for letter in tag])
    total_letters = sum(letters_count.values())
    for key, value in letters_count.items():
        letters_dist[key] = value / total_letters

    #print(photos_as_tags)

    for el in photos_as_tags:
        m = map(lambda x: tags.index(x), el)
        enc_photos.append(set(m))
    enc_photos = np.array(enc_photos)

    return enc_photos, letters_dist

def sample(photos):
    sample = np.random.choice(photos, SAMPLE_SIZE)
    return sample

def build_model():
    # Build input layers
    input_matrix = keras.layers.Input(shape=[SAMPLE_SIZE, SAMPLE_SIZE, 1], name="input_matrix")
    input_vectors = keras.layers.Input(shape=[SAMPLE_SIZE * 2], name="input_vectors")

    # Conv2d layers
    conv1 = keras.layers.Conv2D(10, kernel_size=(1, SAMPLE_SIZE), activation='relu')(input_matrix)
    conv2 = keras.layers.Conv2D(10, kernel_size=(SAMPLE_SIZE, 1), activation='relu')(conv1)
    flatten = keras.layers.Flatten()(conv2)

    # Combine layers
    combine = keras.layers.concatenate([flatten, input_vectors])

    # Build dense layers
    x = keras.layers.Dense(units=1000, activation='relu', kernel_initializer='glorot_uniform', name="layer_1", use_bias=False)(combine)
    x = keras.layers.Dense(units=1000 , activation='relu', kernel_initializer='glorot_uniform', name="layer_2", use_bias=False)(x)

    # Build output layers
    out = keras.layers.Dense(units=SAMPLE_SIZE, activation='softmax', kernel_initializer='RandomNormal', name="out")(x)

    # Build optimizer
    rms = keras.optimizers.RMSprop(lr=LEARNING_RATE)

    # Define custom loss function
    def custom_loss(y_true, y_pred):
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
        return K.mean(cross_entropy, keepdims=True)

    # Build and compile training model
    model = Model(inputs=[input_matrix, input_vector], outputs=out)
    model.compile(loss=custom_loss, optimizer=rms, metrics=['accuracy'])

    return model

def preprocess(_state):
    # Get data
    matrix = _state[0]
    vector_1 = _state[1]
    vector_2 = _state[2]

    # Reshape data
    matrix = matrix.reshape(1, len(matrix), len(matrix[0]), 1)
    vectors = np.concatenate((vector_1, vector_2), axis=None)
    vectors = vectors.reshape(1, len(vectors))
    
    return matrix, vectors

def play(model, photos):
    while True:
        print("==========STARTING ROLLOUT==========")
        sample_photos = sample(photos)

        # init game
        _state = init_index(sample_photos)
        _matrix_state, _vector_state = preprocess(_state)
        _done = False
        total_reward = 0
        count = 0

        while not _done and count < SAMPLE_SIZE * 2:
            _predict = model.predict([_matrix_state, _vector_state], batch_size=1)[0]
            _action = np.argmax(_predict)
            print("==========")
            print("Action being taken: {}".format(_action))
            print("State:")
            print(_state[0])
            print(_state[1])
            print(_state[2])
            print("==========")
            _state, _reward, _done  = step_index(_state, _action)
            _matrix_state, _vector_state = preprocess(_state)
            total_reward += _reward
            count += 1
        print("Total reward: {}".format(total_reward))
        input()
        
### FROM GOOGLE ###
def create_distance_callback(dist_matrix, max_reward):
  # Create the distance callback.

  def distance_callback(from_node, to_node):
    return int(max_reward - dist_matrix[from_node][to_node])

  return distance_callback

def gen_solution(matrix, tsp_size):
    # Make callbacks
    matrix = np.squeeze(matrix)

    # Consider distance as (max_reward - reward)
    # so the program prioritizes large rewards
    m_max = np.amax(matrix)
    dist_callback = create_distance_callback(matrix, m_max)

    # Make the solver
    routing = pywrapcp.RoutingModel(tsp_size, 1, 0)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Get solution
    solution = []
    node = routing.Start(0)
    while not routing.IsEnd(node):
        solution.append(node)
        node = assignment.Value(routing.NextVar(node))

    # Remove "useless" moves
    prev = solution[0]
    ret = []
    for index in range(1, len(solution) - 1):
        # The move is usefull if either moving to it gives us reward, or if our next move gives us reward
        # Otherwise, we can just skip it
        if (matrix[prev][solution[index]] != 0):
            ret.append(solution[index])
        elif (matrix[solution[index]][solution[index + 1]] != 0):
            ret.append(solution[index])
        prev = solution[index]

    # Check if last move is usefull
    if len(ret) > 0 and matrix[ret[-1]][solution[-1]] != 0:
        ret.append(solution[-1])

    return ret

def gen_frames(frames, photos, verbose=False):
    while True:
        # Play games until we have the amount of frames we want
        epoch_memory = []
        while len(epoch_memory) < frames:
            # Generate new game
            sample_photos = sample(photos)

            # init game
            _state = init_index(sample_photos)
            _matrix_state, _vector_state = preprocess(_state)
            game_memory = []
            _done = False

            # While there are no rewards in our sample, reroll
            while (np.amax(_matrix_state) == 0):
                sample_photos = sample(photos)
                _state = init_index(sample_photos)
                _matrix_state, _vector_state = preprocess(_state)
                    

            # init solution
            _solved = gen_solution(_matrix_state, SAMPLE_SIZE)
            if verbose: print(_solved)

            # Take solution actions
            for index in _solved:
                if verbose:
                    print(_state[0])
                    print(_state[1])
                    print(_state[2])
                _action = index
                _state, _reward, _done  = step_index(_state, _action)
                if verbose:
                    print("Action being taken: {}".format(_action))
                    print("Reward: {}".format(_reward))
                    print("Done? {}".format(_done))
                game_memory.append((_matrix_state, _vector_state, _reward, _action))
                _matrix_state, _vector_state = preprocess(_state)

            # Once the game has finished, process the rewards then save to epoch_memory
            _m_s, _v_s, _rewards, _labels = zip(*game_memory)
            PERIOD_REWARD.append(sum(_rewards))
            epoch_memory.extend(zip(_m_s, _v_s, _labels))

        # Shuffle and return frames
        epoch_memory = [tuple(ex) for ex in np.array(epoch_memory)[np.random.permutation(len(epoch_memory))]]
        _matrixs, _vectors, _labels = zip(*epoch_memory)

        # Cut returning arrays down to frames length to give a costant expected shape for tensors
        _matrixs = np.array(_matrixs).reshape(len(_matrixs), SAMPLE_SIZE, SAMPLE_SIZE, 1)
        _vectors = np.array(_vectors)[:frames].reshape(ROLLOUT_SIZE, SAMPLE_SIZE * 2)
        _labels = np.squeeze(np.array(_labels))

        yield ({"input_matrix": _matrixs[:frames], "input_vectors": _vectors}, _labels[:frames])

def build_dataset():
    memory = deque([], maxlen=MEMORY_SIZE)
    def next():
        for data in memory:
            yield data

    dataset = tf.data.Dataset.from_generator(
        next,
        output_types=({"input_matrix": tf.float32, "input_vectors": tf.float32}, tf.int32),
        output_shapes=({"input_matrix": (ROLLOUT_SIZE, SAMPLE_SIZE, SAMPLE_SIZE, 1), "input_vectors": (ROLLOUT_SIZE, SAMPLE_SIZE * 2)}, (ROLLOUT_SIZE))
        )
    dataset.batch(BATCH_SIZE).repeat()
    return memory, dataset

def main(args):
    global BATCH_SIZE, LEARNING_RATE, PERIOD_REWARD, GAMMA, SAMPLE_SIZE
    print('args: {}'.format(args))

    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate

    GAMMA = args.gamma

    PERIOD_REWARD = []
    SAMPLE_SIZE = args.sample_size

    # Build model
    model = build_model()

    # Read photos
    photos = read_file()

    # Build callbacks
    tbCallBack = callbacks.TensorBoard(log_dir=args.output_dir, histogram_freq=0, write_graph=True, write_images=True)
    filepath="new_model/checkpoint"
    saveCallBack = callbacks.ModelCheckpoint(filepath.format(args.output_dir), monitor='val_loss', period=5)

    # Build generator and dataset
    memory, dataset = build_dataset()
    gen = gen_frames(ROLLOUT_SIZE, model, photos)

    # Train!
    for e in range(args.n_epoch):
        print("Generating data...")
        memory.append(next(gen))

        PERIOD_REWARD = []
        print("Training...")
        model_train.fit(
            dataset,
            batch_size=ROLLOUT_SIZE,
            initial_epoch=e,
            epochs=(e + 1),
            steps_per_epoch=1,
            verbose=1,
            callbacks=[tbCallBack, saveCallBack])

        #print(model_train.train_on_batch(dataset))
        #print("All rewards: {}".format(PERIOD_REWARD))
        print("Average epoch reward: {}".format(sum(PERIOD_REWARD) / len(PERIOD_REWARD)))

    play(model, photos)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('pizza cutter trainer')
    parser.add_argument(
        '--n-epoch',
        type=int,
        default=1000)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=3000)
    parser.add_argument(
        '--period',
        type=int,
        default=5)
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='/tmp/pizza_output')
    parser.add_argument(
        '--restore',
        default=False,
        action='store_true')
    parser.add_argument(
        '--play',
        default=False,
        action='store_true')
    parser.add_argument(
        '--save-checkpoint-steps',
        type=int,
        default=1)
    parser.add_argument(
        '--sample-size',
        type=int,
        default=4)
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.9)
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-2)

    args = parser.parse_args()

    # save all checkpoints
    args.max_to_keep = args.n_epoch // args.save_checkpoint_steps

    main(args)
