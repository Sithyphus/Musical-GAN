'''
Octavia, musical generation A.I.
    Joseph Kopplin
    Braxton Lazar
    Jorge Barrios (Sisyphus Igor)
'''

#importing required dependancies
#import keras #main library for neural networks
import numpy as np #mathematical structures, primarily arrays
import matplotlib.pyplot as plt #optional for visualisation 
import glob #used for parsing data from folders
import music21 as m21 #needed for parsing information from MIDI files into useful formats
import pickle


#Data prep
#This code was extrapolated from https://github.com/Skuldur/Classical-Piano-Composer/blob/master/lstm.py, written by Skuldur
def get_notes(folder):
    ''' 
    This, for now until I can more easily manuver the music21 package, takes all the notes from the "Raw_Files" folder
    '''
    notes = []

    for file in glob.glob(folder + "/*.mid"):
        midi = m21.converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = m21.instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, m21.note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, m21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

schist = get_notes("RawFiles")

pitchnames = sorted(set(item for item in schist))
print(pitchnames)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(schist)