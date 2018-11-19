'''
Octavia, musical generation A.I.
    Joseph Kopplin
    Braxton Lazar
    Jorge Barrios (Sisyphus Igor)
'''

""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras import optimizers

def train_network():
    """ Train a Neural Network to generate music """
    with open('data/notes','rb') as filepath:
        notes = pickle.load(filepath)
    notes = notes[:int(len(notes) * .01)]
    # get amount of pitch names
    n_vocab = len(set(notes))
    
    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(Bidirectional(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    )))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    optimizer = optimizers.RMSprop(lr = 0.001,rho = 0.9,epsilon = None,decay = 0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    '''
    model.add(Conv1D(
        filters = 512,
        kernel_size = 4,
        strides = 4,
        activation = 'relu',
        input_shape = (network_input.shape[1],network_input.shape[2])))
    print(model.shape)

    model.add(Bidirectional(LSTM(
        512,
        input_shape = (network_input.shape[1],network_input.shape[2]),
        return_sequences = True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512, return_sequences = True)))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dropout(0.3))
    #model.add(Conv1D(1024,4,strides = 4, activation = 'relu'))
    model.add(Bidirectional(LSTM(512, return_sequences = True)))
    model.add(Dense(256))
    model.add(Dense(n_vocab))
    model.add(Activation('softplus'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
    '''
    
    return model

def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=10, batch_size=64, callbacks=callbacks_list)


train_network()