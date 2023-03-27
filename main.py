import os
import shutil
import pickle
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dropout,Dense, BatchNormalization, Activation
from music21 import converter, instrument, note, chord

DATA_FOLDER_PATH = 'data/' # Dependant folder.
WEIGHTS_FILE_PATH = DATA_FOLDER_PATH+'model_weights.hdf5' # Training checkpoint.
NOTES_FILE_PATH = DATA_FOLDER_PATH+'notes' # This file is simply all the songs in one huge list dumped here.
NETWORK_INPUT_FILE_PATH = DATA_FOLDER_PATH + 'network_input' # This contains a list of song sequences.
NETWORK_OUTPUT_FILE_PATH = DATA_FOLDER_PATH + 'network_output' # This contains target notes for the network based on the sequences.
MIDI_FILE_PATH = 'midi_songs/' # Put your midi tracks in this folder. 
SEQUENCE_LENGTH = 100 # The model should take this length of notes/samples before trying to guess what should be next.
EPOCH = 2
BATCH_SIZE = 20 #128
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
CALLBACKS_LIST = [checkpoint]


def convert_midi_to_list(path_to_files):
    '''
    This function will simply take the list of midi files and convert them into a long list of notes/chords.
    '''
    # Get the list of midi files
    try:
        files = [file for file in os.listdir(path_to_files) if file.endswith('.mid')]
    except:
        print(f'Folder {path_to_files} not found in root directory. Exiting ...')
        exit()
    
    # if there's no midi files, exit with message.
    if len(files)==None:
        print(f'Empty folder found at {path_to_files}. Please ensure there exists your midi tracks (.mid files) are in this folder')
        exit()
    else:
        print(f'Midi files detected in {path_to_files} ....')

    notes = [] # Init return list.

    for file in files:
        notes_to_parse = None

        midi = converter.parse(path_to_files+file) 
        print("Parsing %s" % file)

        # We want only 1 instrument to work with.
        try: # If there's multiple, take the first.
            part = instrument.partitionByInstrument(midi)
            notes_to_parse = part.parts[0].recurse() 
        except: # If it fails, file has notes in a flat structure.
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note): # Find note and append
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord): # Find chord and append
                notes.append('.'.join(str(n) for n in element.normalOrder))
            elif isinstance(element, note.Rest): # Find rest and append
                notes.append('r')
               
    # Dump the output into file. Creates a new one if it doesn't exist.
    with open(NOTES_FILE_PATH, 'wb+') as filepath:
        pickle.dump(notes, filepath)

    return notes

def sequence_and_target(notes, size_of_vocabulary, sequence_length=100):
    '''
    Here, we will break the input list of notes into many lists of pre-defined lengths of sequences 
    and separately, a note for each of them.

    The idea is the model will use that sequence to try and guess the note. 
    '''
    # get a sorted vocabulary list
    vocabulary = sorted(set(note for note in notes))

    # Create a dictionary of note/number pairs for encoding.
    note_to_number_dictionary = dict((note, number) for number, note in enumerate(vocabulary))

    network_input = []
    network_output = []

    # Create sequences and corresponding target note.
    for i in range(0, len(notes) - sequence_length, 1):
        sequence = notes[i:i + sequence_length] # Slice list to make the sequence
        target = notes[i + sequence_length] # Get the note right after that sequence.

        network_input.append([note_to_number_dictionary[char] for char in sequence]) # This list will store the sequence after encoding it to numbers.
        network_output.append(note_to_number_dictionary[target]) # This list will have the corresponding target notes after encoding.

    number_of_sequences = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (number_of_sequences, sequence_length, 1))
    # normalize input
    # network_input = network_input / float(n_vocab)

    # One Hot Encoding of sorts.
    network_output = np_utils.to_categorical(network_output) 

    # Dump the output into file. 
    with open(NETWORK_INPUT_FILE_PATH, 'wb+') as filepath:
        pickle.dump(network_input, filepath)
    # Dump the output into file. 
    with open(NETWORK_OUTPUT_FILE_PATH, 'wb+') as filepath:
        pickle.dump(network_output, filepath)

    # Give out the resultant network training data.
    return (network_input,network_output)
    
def create_model(network_input,size_of_vocabulary):
    '''
    This function will create the model and return it, that's all.
    '''
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(size_of_vocabulary))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


# This is the main control flow.
if __name__ == '__main__':    

    # In case there's new midi files added, you might want to delete the data folder and calculate everything again.
    if int(input("Do you want to re-initialize the data folder for new midi files? Press 1 else 0\n")):
        print(f'Trying to delete the {DATA_FOLDER_PATH}...')
        try:
            shutil.rmtree(DATA_FOLDER_PATH)
            print(f'{DATA_FOLDER_PATH} deleted successfully!')
        except:
            print(f'{DATA_FOLDER_PATH} did not exist.')


    # This will create the data folder if it doesn't exist
    if not os.path.exists(DATA_FOLDER_PATH):
        print(f'folder {DATA_FOLDER_PATH} not found. Making one now ...')
        os.makedirs(DATA_FOLDER_PATH) # Make data folder.

    # Check for notes backup file
    if not os.path.exists(NOTES_FILE_PATH): 
        print(f'Notes backup file not found at {NOTES_FILE_PATH}. Parsing MIDI files now ...')
        notes = convert_midi_to_list(MIDI_FILE_PATH) # Process midi files since backup not found.
    else:
        with open(NOTES_FILE_PATH, 'rb') as filepath:
            notes = pickle.load(filepath)
    
    # Check if there's any checkpoint file.
    #if os.path.isfile(WEIGHTS_FILE_PATH):
    #    print(f"{WEIGHTS_FILE_PATH} exists!\nWill train from here.")
    #else:
    #    print(f"{WEIGHTS_FILE_PATH} does not exist.\nWill create new checkpoint.")
    
    # Get the number of unique notes.
    size_of_vocabulary = len(set(notes))

    # Check for network_input backup file
    if not os.path.exists(NETWORK_INPUT_FILE_PATH): 
        print(f'Network input sequences backup file not found at {NETWORK_INPUT_FILE_PATH} and {NETWORK_OUTPUT_FILE_PATH}. Creating sequences now ...')
        network_input, network_output = sequence_and_target(notes, size_of_vocabulary, SEQUENCE_LENGTH)
    else:
        with open(NETWORK_INPUT_FILE_PATH, 'rb') as filepath:
            network_input = pickle.load(NETWORK_INPUT_FILE_PATH)
        with open(NETWORK_OUTPUT_FILE_PATH, 'rb') as filepath:
            network_output = pickle.load(NETWORK_OUTPUT_FILE_PATH)

    
    # This fellow does the hard work
    model = create_model(network_input, size_of_vocabulary)

    # Ask if this program should run in training mode or should it sing?
    # TODO
    

    model.fit(network_input, network_output, epochs=EPOCH, batch_size=BATCH_SIZE, callbacks=CALLBACKS_LIST)

