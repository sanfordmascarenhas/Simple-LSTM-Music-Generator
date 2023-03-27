import os
from music21 import converter, instrument, note, chord

WEIGHTS_FILE_PATH = 'data/model_weights.hdf5'
MIDI_FILE_PATH = 'midi_songs/'

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
               
    # Implement save notes feature

    return notes


# This is the main control flow.
if __name__ == '__main__':

    # First check if there's any checkpoint file.
    if os.path.isfile(WEIGHTS_FILE_PATH):
        print(f"{WEIGHTS_FILE_PATH} exists!\nWill train from here.")
    else:
        print(f"{WEIGHTS_FILE_PATH} does not exist.\nWill create new checkpoint.")
    
    notes = convert_midi_to_list(MIDI_FILE_PATH)
    print(notes)

