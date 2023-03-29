# Simple LSTM Music Generator
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

A very simple deep learning based midi generator that trains a Recurrent Neural Network(RNN) to learn from midi files and then generate its own music in the same style. 

## Requirements
- Python 3.7+
- Music21 (```pip install music21```)
- Keras (```pip install keras```)
- Numpy (```pip install numpy```)

## Setup
1. Clone this repository using or download the source code from github.
2. Install all dependencies using the ```pip install``` commands mentioned above.
3. Create a folder called ```midi_songs``` in the root directory of the project and place your MIDI files in it.
4. Run the ```midi_generator.py``` file.

## Configuration
- ```DATA_FOLDER_PATH``` - Path to the folder where data files will be stored.
- ```NOTES_FILE_PATH``` - Filepath of the notes file.
- ```NETWORK_INPUT_FILE_PATH``` - Filepath of the input network file. This contains a list of song sequences.
- ```NETWORK_OUTPUT_FILE_PATH``` - Filepath of the output network file. This contains target notes for the network based on the sequences.
- ```VOCABULARY_FILE_PATH``` - Filepath of the vocabulary file of all the unique notes in the files.
- ```NOTE_TO_NUMBER_DICTIONARY_FILE_PATH``` - Filepath of the note to number dictionary file. This contains the conversion table of notes to an encoded number.
- ```MIDI_FILE_PATH``` - Path to the folder where MIDI files are stored.
- ```SEQUENCE_LENGTH``` - Length of each sequence. Default is ```100```.
- ```EPOCH``` - Number of epochs to train for. Default is ```200```.
- ```BATCH_SIZE``` - Batch size to use during training. Default is ```70```.
- ```LENGTH_OF_SONG``` - Length of the generated song. Default is ```100```.
- ```OFFSET_INCREMENTS``` - List of offsets to select from when generating a song. Default is ```[0.25, 0.5, 0.75, 1]```. 
- ```WEIGHTS_FILE_PATH``` - Path to the file where weights will be saved. Default is ```"weights.h```.

<br>
<hr>

## How it Works
The generator first takes in a list of midi files and converts them into a long list of notes and chords using the ```convert_midi_to_list``` function. The output is then dumped into a file for later use. All these files are stored in the `data` folder. Next, the ```sequence_and_target``` function is called with the note list, size of the vocabulary, and the sequence length as arguments. This function breaks the input list of notes into many lists of predefined lengths of sequences and separately, a note for each of them.

The model then uses that sequence to try and guess the note. The model architecture consists of an LSTM layer, a Dropout layer for regularization, a Dense layer with batch normalization and activation, and finally, a Softmax activation layer. The weights of the best performing model are saved in the ```weights.hdf5``` file.

## Usage
1. Run the ```midi_generator.py``` file.
2. You have a choice of training the model or to run it directly with the weights.
3. If you choose to train, the generator will train for a specified number of epochs (default is 200). You can change this by modifying the EPOCH variable in the code.
3. Once training is complete, the generator will use the trained model to generate a new midi file.
4. The generated midi file will be saved in the root directory of the project.
5. You will need an application like MuseScore to play the track.


<br>
<hr>
 This is a derivative project from https://github.com/Skuldur/Classical-Piano-Composer with some modifications.

