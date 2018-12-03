
from music21 import converter, instrument, note, chord
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
import Zadavia

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    midi = converter.parse('test_output.mid')

    notes_to_parse = None

    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

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


def discriminator(input_shape, n_vocab):
    disk_model = Sequential()
    input_shape = (input_shape)
    disk_model.add(Bidirectional(LSTM(
        512,
        input_shape=(input_shape),
        return_sequences=True
    )))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dense(n_vocab))
    return disk_model
    
    
def discriminator_model(notes,n_vocab):
    optimizer = RMSprop(lr=0.0002, decay=0.0)
    DM = Sequential()
    DM.add(discriminator(prepare_sequences(notes,n_vocab),n_vocab))
    DM.compile(loss='categorical_crossentropy', optimizer=optimizer,\
        metrics=['accuracy'])
    return DM
    
def main():
    notes = get_notes()
    # get amount of pitch names
    n_vocab = len(set(notes))
    DM = discriminator_model(notes,n_vocab)
    
    

if __name__ == '__main__':
    main()