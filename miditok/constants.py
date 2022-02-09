"""Constants for data encoding

"""

CURRENT_PACKAGE_VERSION = '1.3.4'  # used un setup and when saving the config of a tokenizer

# MIDI encodings default parameters, used when tokenizing a dataset and using tokens
# These are the parameters from which a MIDI file will be tokenized
PITCH_RANGE = range(21, 109)  # the recommended pitches for piano in the GM2 specs are from 21 to 108
BEAT_RES = {(0, 4): 8, (4, 12): 4}  # samples per beat
NB_VELOCITIES = 32  # nb of velocity bins, velocities values from 0 to 127 will be quantized
ADDITIONAL_TOKENS = {'Chord': False,
                     'Rest': False,
                     'Tempo': False,
                     'TimeSignature': False,
                     'Program': False,
                     # rest params
                     'rest_range': (2, 8),  # (/min_rest, max_rest_in_BEAT), first divides a whole note/rest
                     # tempo params
                     'nb_tempos': 32,  # nb of tempo bins for additional tempo tokens, quantized like velocities
                     'tempo_range': (40, 250),  # (min_tempo, max_tempo)
                     # time signature params
                     'time_signature_range': {4: 4},  # {denom_i: num_i / num_list_i / (min_num_i, max_num_i)}
                     'nb_beats': 8}  # max number of 4th note beats for position tokens


# Defaults values when writing new MIDI files
TIME_DIVISION = 384  # 384 and 480 are convenient as divisible by 4, 8, 12, 16, 24, 32
TEMPO = 120
TIME_SIGNATURE = (4, 4)
MIDI_BEAT_RESOLUTION = 4  # in MIDI beat resolution is 4th note

CHORD_MAPS = {'min': (0, 3, 7),
              'maj': (0, 4, 7),
              'dim': (0, 3, 6),
              'aug': (0, 4, 8),
              'sus2': (0, 2, 7),
              'sus4': (0, 5, 7),

              '7dom': (0, 4, 7, 10),
              '7min': (0, 3, 7, 10),
              '7maj': (0, 4, 7, 11),
              '7halfdim': (0, 3, 6, 10),
              '7dim': (0, 3, 6, 9),
              '7aug': (0, 4, 8, 11),

              '9maj': (0, 4, 7, 10, 14),
              '9min': (0, 4, 7, 10, 13)}

# http://newt.phys.unsw.edu.au/jw/notes.html
# https://www.midi.org/specifications

# index i = program i+1 in the GM2 specs (7. Appendix A)
# index i = program i as retrieved by packages like mido or miditoolkit
MIDI_INSTRUMENTS = [{'name': 'Acoustic Grand Piano', 'pitch_range': range(21, 109)},
                    {'name': 'Bright Acoustic Piano', 'pitch_range': range(21, 109)},
                    {'name': 'Electric Grand Piano', 'pitch_range': range(21, 109)},
                    {'name': 'Honky-tonk Piano', 'pitch_range': range(21, 109)},
                    {'name': 'Electric Piano 1', 'pitch_range': range(28, 104)},
                    {'name': 'Electric Piano 2', 'pitch_range': range(28, 104)},
                    {'name': 'Harpsichord', 'pitch_range': range(41, 90)},
                    {'name': 'Clavi', 'pitch_range': range(36, 97)},

                    # Chromatic Percussion
                    {'name': 'Celesta', 'pitch_range': range(60, 109)},
                    {'name': 'Glockenspiel', 'pitch_range': range(72, 109)},
                    {'name': 'Music Box', 'pitch_range': range(60, 85)},
                    {'name': 'Vibraphone', 'pitch_range': range(53, 90)},
                    {'name': 'Marimba', 'pitch_range': range(48, 85)},
                    {'name': 'Xylophone', 'pitch_range': range(65, 97)},
                    {'name': 'Tubular Bells', 'pitch_range': range(60, 78)},
                    {'name': 'Dulcimer', 'pitch_range': range(60, 85)},

                    # Organs
                    {'name': 'Drawbar Organ', 'pitch_range': range(36, 97)},
                    {'name': 'Percussive Organ', 'pitch_range': range(36, 97)},
                    {'name': 'Rock Organ', 'pitch_range': range(36, 97)},
                    {'name': 'Church Organ', 'pitch_range': range(21, 109)},
                    {'name': 'Reed Organ', 'pitch_range': range(36, 97)},
                    {'name': 'Accordion', 'pitch_range': range(53, 90)},
                    {'name': 'Harmonica', 'pitch_range': range(60, 85)},
                    {'name': 'Tango Accordion', 'pitch_range': range(53, 90)},

                    # Guitars
                    {'name': 'Acoustic Guitar (nylon)', 'pitch_range': range(40, 85)},
                    {'name': 'Acoustic Guitar (steel)', 'pitch_range': range(40, 85)},
                    {'name': 'Electric Guitar (jazz)', 'pitch_range': range(40, 87)},
                    {'name': 'Electric Guitar (clean)', 'pitch_range': range(40, 87)},
                    {'name': 'Electric Guitar (muted)', 'pitch_range': range(40, 87)},
                    {'name': 'Overdriven Guitar', 'pitch_range': range(40, 87)},
                    {'name': 'Distortion Guitar', 'pitch_range': range(40, 87)},
                    {'name': 'Guitar Harmonics', 'pitch_range': range(40, 87)},

                    # Bass
                    {'name': 'Acoustic Bass', 'pitch_range': range(28, 56)},
                    {'name': 'Electric Bass (finger)', 'pitch_range': range(28, 56)},
                    {'name': 'Electric Bass (pick)', 'pitch_range': range(28, 56)},
                    {'name': 'Fretless Bass', 'pitch_range': range(28, 56)},
                    {'name': 'Slap Bass 1', 'pitch_range': range(28, 56)},
                    {'name': 'Slap Bass 2', 'pitch_range': range(28, 56)},
                    {'name': 'Synth Bass 1', 'pitch_range': range(28, 56)},
                    {'name': 'Synth Bass 2', 'pitch_range': range(28, 56)},

                    # Strings & Orchestral instruments
                    {'name': 'Violin', 'pitch_range': range(55, 94)},
                    {'name': 'Viola', 'pitch_range': range(48, 85)},
                    {'name': 'Cello', 'pitch_range': range(36, 73)},
                    {'name': 'Contrabass', 'pitch_range': range(28, 56)},
                    {'name': 'Tremolo Strings', 'pitch_range': range(28, 94)},
                    {'name': 'Pizzicato Strings', 'pitch_range': range(28, 94)},
                    {'name': 'Orchestral Harp', 'pitch_range': range(23, 104)},
                    {'name': 'Timpani', 'pitch_range': range(36, 58)},

                    # Ensembles
                    {'name': 'String Ensembles 1', 'pitch_range': range(28, 97)},
                    {'name': 'String Ensembles 2', 'pitch_range': range(28, 97)},
                    {'name': 'SynthStrings 1', 'pitch_range': range(36, 97)},
                    {'name': 'SynthStrings 2', 'pitch_range': range(36, 97)},
                    {'name': 'Choir Aahs', 'pitch_range': range(48, 80)},
                    {'name': 'Voice Oohs', 'pitch_range': range(48, 80)},
                    {'name': 'Synth Voice', 'pitch_range': range(48, 85)},
                    {'name': 'Orchestra Hit', 'pitch_range': range(48, 73)},

                    # Brass
                    {'name': 'Trumpet', 'pitch_range': range(58, 95)},
                    {'name': 'Trombone', 'pitch_range': range(34, 76)},
                    {'name': 'Tuba', 'pitch_range': range(29, 56)},
                    {'name': 'Muted Trumpet', 'pitch_range': range(58, 83)},
                    {'name': 'French Horn', 'pitch_range': range(41, 78)},
                    {'name': 'Brass Section', 'pitch_range': range(36, 97)},
                    {'name': 'Synth Brass 1', 'pitch_range': range(36, 97)},
                    {'name': 'Synth Brass 2', 'pitch_range': range(36, 97)},

                    # Reed
                    {'name': 'Soprano Sax', 'pitch_range': range(54, 88)},
                    {'name': 'Alto Sax', 'pitch_range': range(49, 81)},
                    {'name': 'Tenor Sax', 'pitch_range': range(42, 76)},
                    {'name': 'Baritone Sax', 'pitch_range': range(37, 69)},
                    {'name': 'Oboe', 'pitch_range': range(58, 92)},
                    {'name': 'English Horn', 'pitch_range': range(52, 82)},
                    {'name': 'Bassoon', 'pitch_range': range(34, 73)},
                    {'name': 'Clarinet', 'pitch_range': range(50, 92)},

                    # Pipe
                    {'name': 'Piccolo', 'pitch_range': range(74, 109)},
                    {'name': 'Flute', 'pitch_range': range(60, 97)},
                    {'name': 'Recorder', 'pitch_range': range(60, 97)},
                    {'name': 'Pan Flute', 'pitch_range': range(60, 97)},
                    {'name': 'Blown Bottle', 'pitch_range': range(60, 97)},
                    {'name': 'Shakuhachi', 'pitch_range': range(55, 85)},
                    {'name': 'Whistle', 'pitch_range': range(60, 97)},
                    {'name': 'Ocarina', 'pitch_range': range(60, 85)},

                    # Synth Lead
                    {'name': 'Lead 1 (square)', 'pitch_range': range(21, 109)},
                    {'name': 'Lead 2 (sawtooth)', 'pitch_range': range(21, 109)},
                    {'name': 'Lead 3 (calliope)', 'pitch_range': range(36, 97)},
                    {'name': 'Lead 4 (chiff)', 'pitch_range': range(36, 97)},
                    {'name': 'Lead 5 (charang)', 'pitch_range': range(36, 97)},
                    {'name': 'Lead 6 (voice)', 'pitch_range': range(36, 97)},
                    {'name': 'Lead 7 (fifths)', 'pitch_range': range(36, 97)},
                    {'name': 'Lead 8 (bass + lead)', 'pitch_range': range(21, 109)},

                    # Synth Pad
                    {'name': 'Pad 1 (new age)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 2 (warm)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 3 (polysynth)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 4 (choir)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 5 (bowed)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 6 (metallic)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 7 (halo)', 'pitch_range': range(36, 97)},
                    {'name': 'Pad 8 (sweep)', 'pitch_range': range(36, 97)},

                    # Synth SFX
                    {'name': 'FX 1 (rain)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 2 (soundtrack)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 3 (crystal)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 4 (atmosphere)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 5 (brightness)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 6 (goblins)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 7 (echoes)', 'pitch_range': range(36, 97)},
                    {'name': 'FX 8 (sci-fi)', 'pitch_range': range(36, 97)},

                    # Ethnic Misc.
                    {'name': 'Sitar', 'pitch_range': range(48, 78)},
                    {'name': 'Banjo', 'pitch_range': range(48, 85)},
                    {'name': 'Shamisen', 'pitch_range': range(50, 80)},
                    {'name': 'Koto', 'pitch_range': range(55, 85)},
                    {'name': 'Kalimba', 'pitch_range': range(48, 80)},
                    {'name': 'Bag pipe', 'pitch_range': range(36, 78)},
                    {'name': 'Fiddle', 'pitch_range': range(55, 97)},
                    {'name': 'Shanai', 'pitch_range': range(48, 73)},

                    # Percussive
                    {'name': 'Tinkle Bell', 'pitch_range': range(72, 85)},
                    {'name': 'Agogo', 'pitch_range': range(60, 73)},
                    {'name': 'Steel Drums', 'pitch_range': range(52, 77)},
                    {'name': 'Woodblock', 'pitch_range': range(0, 128)},
                    {'name': 'Taiko Drum', 'pitch_range': range(0, 128)},
                    {'name': 'Melodic Tom', 'pitch_range': range(0, 128)},
                    {'name': 'Synth Drum', 'pitch_range': range(0, 128)},
                    {'name': 'Reverse Cymbal', 'pitch_range': range(0, 128)},

                    # SFX
                    {'name': 'Guitar Fret Noise, Guitar Cutting Noise', 'pitch_range': range(0, 128)},
                    {'name': 'Breath Noise, Flute Key Click', 'pitch_range': range(0, 128)},
                    {'name': 'Seashore, Rain, Thunder, Wind, Stream, Bubbles', 'pitch_range': range(0, 128)},
                    {'name': 'Bird Tweet, Dog, Horse Gallop', 'pitch_range': range(0, 128)},
                    {'name': 'Telephone Ring, Door Creaking, Door, Scratch, Wind Chime', 'pitch_range': range(0, 128)},
                    {'name': 'Helicopter, Car Sounds', 'pitch_range': range(0, 128)},
                    {'name': 'Applause, Laughing, Screaming, Punch, Heart Beat, Footstep',
                     'pitch_range': range(0, 128)},
                    {'name': 'Gunshot, Machine Gun, Lasergun, Explosion', 'pitch_range': range(0, 128)}]

INSTRUMENT_CLASSES = [{'name': 'Piano', 'program_range': range(0, 8)},  # 0
                      {'name': 'Chromatic Percussion', 'program_range': range(8, 16)},
                      {'name': 'Organ', 'program_range': range(16, 24)},
                      {'name': 'Guitar', 'program_range': range(24, 32)},
                      {'name': 'Bass', 'program_range': range(32, 40)},
                      {'name': 'Strings', 'program_range': range(40, 48)},  # 5
                      {'name': 'Ensemble', 'program_range': range(48, 56)},
                      {'name': 'Brass', 'program_range': range(56, 64)},
                      {'name': 'Reed', 'program_range': range(64, 72)},
                      {'name': 'Pipe', 'program_range': range(72, 80)},
                      {'name': 'Synth Lead', 'program_range': range(80, 88)},  # 10
                      {'name': 'Synth Pad', 'program_range': range(88, 96)},
                      {'name': 'Synth Effects', 'program_range': range(96, 104)},
                      {'name': 'Ethnic', 'program_range': range(104, 112)},
                      {'name': 'Percussive', 'program_range': range(112, 120)},
                      {'name': 'Sound Effects', 'program_range': range(120, 128)},  # 15
                      {'name': 'Drums', 'program_range': range(-1, 0)}]

# To easily get the class index of any instrument program
CLASS_OF_INST = [i for i, inst_class in enumerate(INSTRUMENT_CLASSES) for _ in inst_class['program_range']]

# index i = program i+1 in the GM2 specs (8. Appendix B)
# index i = program i as retrieved by packages like mido or miditoolkit
DRUM_SETS = {0: 'Standard', 8: 'Room', 16: 'Power', 24: 'Electronic', 25: 'Analog', 32: 'Jazz', 40: 'Brush',
             48: 'Orchestra', 56: 'SFX'}

# Control changes list (without specifications):
# https://www.midi.org/specifications-old/item/table-3-control-change-messages-data-bytes-2
# Undefined and general control changes are not considered here
# All these attributes can take values from 0 to 127, with some of them being on/off
CONTROL_CHANGES = {
    # MSB
    0: 'Bank Select',
    1: 'Modulation Depth',
    2: 'Breath Controller',
    4: 'Foot Controller',
    5: 'Portamento Time',
    6: 'Data Entry',
    7: 'Channel Volume',
    8: 'Balance',
    10: 'Pan',
    11: 'Expression Controller',

    # LSB
    32: 'Bank Select',
    33: 'Modulation Depth',
    34: 'Breath Controller',
    36: 'Foot Controller',
    37: 'Portamento Time',
    38: 'Data Entry',
    39: 'Channel Volume',
    40: 'Balance',
    42: 'Pan',
    43: 'Expression Controller',

    # On / Off control changes, ≤63 off, ≥64 on
    64: 'Damper Pedal',
    65: 'Portamento',
    66: 'Sostenuto',
    67: 'Soft Pedal',
    68: 'Legato Footswitch',
    69: 'Hold 2',

    # Continuous controls
    70: 'Sound Variation',
    71: 'Timbre/Harmonic Intensity',
    72: 'Release Time',
    73: 'Attack Time',
    74: 'Brightness',
    75: 'Decay Time',
    76: 'Vibrato Rate',
    77: 'Vibrato Depth',
    78: 'Vibrato Delay',
    84: 'Portamento Control',
    88: 'High Resolution Velocity Prefix',

    # Effects depths
    91: 'Reverb Depth',
    92: 'Tremolo Depth',
    93: 'Chorus Depth',
    94: 'Celeste Depth',
    95: 'Phaser Depth',

    # Registered parameters numbers
    96: 'Data Increment',
    97: 'Data Decrement',
    #  98: 'Non-Registered Parameter Number (NRPN) - LSB',
    #  99: 'Non-Registered Parameter Number (NRPN) - MSB',
    100: 'Registered Parameter Number (RPN) - LSB',
    101: 'Registered Parameter Number (RPN) - MSB',

    # Channel mode controls
    120: 'All Sound Off',
    121: 'Reset All Controllers',
    122: 'Local Control On/Off',
    123: 'All Notes Off',
    124: 'Omni Mode Off',  # + all notes off
    125: 'Omni Mode On',  # + all notes off
    126: 'Mono Mode On',  # + poly off, + all notes off
    127: 'Poly Mode On'  # + mono off, +all notes off
}
