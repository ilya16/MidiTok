""" Octuple encoding method, as introduced in MusicBERT
https://arxiv.org/abs/2106.05630

"""

import json
import math
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from .midi_tokenizer_base import MIDITokenizer, Vocabulary, Event
from .constants import *


class Octuple(MIDITokenizer):
    """ Octuple encoding method, as introduced in MusicBERT
    https://arxiv.org/abs/2106.05630

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (time signature, tempo)
    :param sos_eos_tokens: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param mask_token: adds Mask (MASK) token to the vocabulary
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 sos_eos_tokens: bool = False, mask_token: bool = False, params=None):
        additional_tokens['Chord'] = False  # Incompatible additional token
        additional_tokens['Rest'] = False
        # used in place of positional encoding
        self.max_bar_embedding = 60  # this attribute might increase during encoding
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens, mask_token, params)

    def save_params(self, out_dir: Union[str, Path, PurePath]):
        """ Override the parent class method to include additional parameter drum pitch range
        Saves the base parameters of this encoding in a txt file
        Useful to keep track of how a dataset has been tokenized / encoded
        It will also save the name of the class used, i.e. the encoding strategy

        :param out_dir: output directory to save the file
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(PurePath(out_dir, 'config').with_suffix(".txt"), 'w') as outfile:
            json.dump({'pitch_range': (self.pitch_range.start, self.pitch_range.stop),
                       'beat_res': {f'{k1}_{k2}': v for (k1, k2), v in self.beat_res.items()},
                       'nb_velocities': len(self.velocities),
                       'additional_tokens': self.additional_tokens,
                       'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding},
                      outfile)

    def midi_to_tokens(self, midi: MidiFile) -> List[List[int]]:
        """ Override the parent class method
        Converts a MIDI file in a tokens representation, a sequence of "time steps".
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Program (track)
            4: Position
            5: Bar
            (6: Tempo)
            (7: TimeSignature)

        :param midi: the MIDI objet to convert
        :return: the token representation, i.e. tracks converted into sequences of tokens
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self.durations_ticks:
            self.durations_ticks[midi.ticks_per_beat] = np.array([(beat * res + pos) * midi.ticks_per_beat // res
                                                                  for beat, pos, res in self.durations])

        # Preprocess the MIDI file
        self.preprocess_midi(midi)

        # Register MIDI metadata
        self.current_midi_metadata = {'time_division': midi.ticks_per_beat,
                                      'tempo_changes': midi.tempo_changes,
                                      'time_sig_changes': midi.time_signature_changes,
                                      'key_sig_changes': midi.key_signature_changes}

        # Convert each track to tokens
        tokens = []
        for track in midi.instruments:
            tokens += self.track_to_tokens(track)

        tokens.sort(key=lambda x: (x[0].time, x[0].desc, x[0].value))  # Sort by time then track then pitch

        # Convert pitch events into tokens
        for time_step in tokens:
            time_step[0] = self.vocab.event_to_token[f'{time_step[0].type}_{time_step[0].value}']

        return tokens

    def track_to_tokens(self, track: Instrument) -> List[List[Union[Event, int]]]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch (as an Event object for sorting purpose afterwards)
            1: Velocity
            2: Duration
            3: Program (track)
            4: Position
            5: Bar
            (6: Tempo)
            (7: TimeSignature)

        :param track: track object to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        time_division = self.current_midi_metadata['time_division']
        ticks_per_sample = time_division / max(self.beat_res.values())
        dur_bins = self.durations_ticks[time_division]

        events = []
        current_tick = -1
        current_bar = -1
        current_pos = -1
        current_tempo_idx = 0
        current_tempo = self.current_midi_metadata['tempo_changes'][current_tempo_idx].tempo
        current_time_sig_idx = 0
        current_time_sig_tick = 0
        current_time_sig_bar = 0
        current_time_sig = self.current_midi_metadata['time_sig_changes'][current_time_sig_idx]
        ticks_per_bar = self.compute_ticks_per_bar(current_time_sig, time_division)

        for note in track.notes:
            # Positions and bars
            if note.start != current_tick:
                pos_index = int(((note.start - current_time_sig_tick) % ticks_per_bar) / ticks_per_sample)
                current_tick = note.start
                current_bar = current_time_sig_bar + (current_tick - current_time_sig_tick) // ticks_per_bar
                current_pos = pos_index

                # Check bar embedding limit, update if needed
                if self.max_bar_embedding <= current_bar:
                    self.vocab.add_event(f'Bar_{i}' for i in range(self.max_bar_embedding, current_bar + 1))
                    self.max_bar_embedding = current_bar + 1

            # Note attributes
            duration = note.end - note.start
            dur_index = np.argmin(np.abs(dur_bins - duration))
            event = [Event(type_='Pitch', time=note.start, value=note.pitch,
                           desc=-1 if track.is_drum else track.program),
                     self.vocab.event_to_token[f'Velocity_{note.velocity}'],
                     self.vocab.event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}'],
                     self.vocab.event_to_token[f'Program_{-1 if track.is_drum else track.program}'],
                     self.vocab.event_to_token[f'Position_{current_pos}'],
                     self.vocab.event_to_token[f'Bar_{current_bar}']]

            # (Tempo)
            if self.additional_tokens['Tempo']:
                # If the current tempo is not the last one
                if current_tempo_idx + 1 < len(self.current_midi_metadata['tempo_changes']):
                    # Will loop over incoming tempo changes
                    for tempo_change in self.current_midi_metadata['tempo_changes'][current_tempo_idx + 1:]:
                        # If this tempo change happened before the current moment
                        if tempo_change.time <= note.start:
                            current_tempo = tempo_change.tempo
                            current_tempo_idx += 1  # update tempo value (might not change) and index
                        elif tempo_change.time > note.start:
                            break  # this tempo change is beyond the current time step, we break the loop
                event.append(self.vocab.event_to_token[f'Tempo_{current_tempo}'])

            # (TimeSignature)
            if self.additional_tokens['TimeSignature']:
                # If the current time signature is not the last one
                if current_time_sig_idx + 1 < len(self.current_midi_metadata['time_sig_changes']):
                    # Will loop over incoming time signature changes
                    for time_sig in self.current_midi_metadata['time_sig_changes'][current_time_sig_idx + 1:]:
                        # If this time signature change happened before the current moment
                        if time_sig.time <= note.start:
                            current_time_sig = time_sig
                            current_time_sig_idx += 1  # update time signature value (might not change) and index
                            current_time_sig_bar += (time_sig.time - current_time_sig_tick) // ticks_per_bar
                            current_time_sig_tick = time_sig.time
                            ticks_per_bar = self.compute_ticks_per_bar(time_sig, time_division)
                        elif time_sig.time > note.start:
                            break  # this time signature change is beyond the current time step, we break the loop
                event.append(self.vocab.event_to_token[f'TimeSig_{current_time_sig.numerator}/{current_time_sig.denominator}'])

            events.append(event)

        return events

    def tokens_to_midi(self, tokens: List[List[int]], _=None, output_path: Optional[str] = None,
                       time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        """ Override the parent class method
        Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Program (track)
            4: Position
            5: Bar
            (6: Tempo)
            (7: TimeSignature)

        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param _: unused, to match parent method signature
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_sample = time_division // max(self.beat_res.values())

        if self.additional_tokens['Tempo']:
            tempo_changes = [TempoChange(int(self.tokens_to_events(tokens[0])[6].value), 0)]
        else:  # default
            tempo_changes = [TempoChange(TEMPO, 0)]

        if self.additional_tokens['TimeSignature']:
            time_sig = self.parse_token_time_signature(self.tokens_to_events(tokens[0])[-1].value)
        else:  # default
            time_sig = TIME_SIGNATURE
        time_sig_changes = [TimeSignature(*time_sig, 0)]
        ticks_per_bar = self.compute_ticks_per_bar(time_sig_changes[0], time_division)

        current_time_sig_tick = 0
        current_time_sig_bar = 0

        tracks = dict([(n, []) for n in range(-1, 128)])
        for time_step in tokens:
            events = self.tokens_to_events(time_step)

            # Note attributes
            pitch = int(events[0].value)
            vel = int(events[1].value)
            duration = self._token_duration_to_ticks(events[2].value, time_division)

            # Time and track values
            program = int(events[3].value)
            current_pos = int(events[4].value)
            current_bar = int(events[5].value)
            current_tick = current_time_sig_tick + (current_bar - current_time_sig_bar) * ticks_per_bar \
                           + current_pos * ticks_per_sample

            # Append the created note
            tracks[program].append(Note(vel, pitch, current_tick, current_tick + duration))

            # Tempo, adds a TempoChange if necessary
            if self.additional_tokens['Tempo']:
                tempo = int(events[6].value)
                if tempo != tempo_changes[-1].tempo:
                    tempo_changes.append(TempoChange(tempo, current_tick))

            # Time Signature, adds a TimeSignatureChange if necessary
            if self.additional_tokens['TimeSignature']:
                time_sig = self.parse_token_time_signature(events[-1].value)
                if time_sig != (time_sig_changes[-1].numerator, time_sig_changes[-1].denominator):
                    current_time_sig_tick += (current_bar - current_time_sig_bar) * ticks_per_bar
                    current_time_sig_bar = current_bar
                    time_sig = TimeSignature(*time_sig, current_time_sig_tick)
                    ticks_per_bar = self.compute_ticks_per_bar(time_sig, time_division)
                    time_sig_changes.append(time_sig)

        # Tempos
        midi.tempo_changes = tempo_changes

        # Time Signatures
        midi.time_signature_changes = time_sig_changes

        # Appends created notes to MIDI object
        for program, notes in tracks.items():
            if len(notes) == 0:
                continue
            if int(program) == -1:
                midi.instruments.append(Instrument(0, True, 'Drums'))
            else:
                midi.instruments.append(Instrument(int(program), False, MIDI_INSTRUMENTS[int(program)]['name']))
            midi.instruments[-1].notes = notes

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def tokens_to_track(self, tokens: List[List[int]], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False)) -> Tuple[Instrument, List[TempoChange]]:
        """ NOT RELEVANT / IMPLEMENTED IN OCTUPLE
        Use tokens_to_midi instead

        :param tokens: sequence of tokens to convert
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param program: the MIDI program of the produced track and if it drum, (default (0, False), piano)
        :return: the miditoolkit instrument object and tempo changes
        """
        raise NotImplementedError('tokens_to_track not implemented for Octuple, use tokens_to_midi instead')

    def _create_vocabulary(self, sos_eos_tokens: bool = False, mask_token: bool = False) -> Vocabulary:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :param mask_token: will include Mask (MASK) token
        :return: the vocabulary object
        """
        vocab = Vocabulary({'PAD_None': 0})

        # PITCH
        vocab.add_event(f'Pitch_{i}' for i in self.pitch_range)

        # VELOCITY
        vocab.add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        vocab.add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # POSITION
        max_nb_beats = max(map(lambda ts: math.ceil(4 * ts[0] / ts[1]), self.time_signatures))
        nb_positions = max(self.beat_res.values()) * max_nb_beats
        vocab.add_event(f'Position_{i}' for i in range(nb_positions))

        # TEMPO
        if self.additional_tokens['Tempo']:
            vocab.add_event(f'Tempo_{i}' for i in self.tempos)

        # TIME_SIGNATURE
        if self.additional_tokens['TimeSignature']:
            vocab.add_event(f'TimeSig_{i[0]}/{i[1]}' for i in self.time_signatures)

        # PROGRAM
        vocab.add_event(f'Program_{program}' for program in range(-1, 128))

        # SOS & EOS
        if sos_eos_tokens:
            vocab.add_sos_eos()

        # MASK
        if mask_token:
            vocab.add_mask()

        # BAR --- MUST BE LAST IN DIC AS THIS MIGHT BE INCREASED
        vocab.add_event('Bar_None')  # new bar token
        vocab.add_event(f'Bar_{i}' for i in range(self.max_bar_embedding))  # bar embeddings (positional encoding)

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        """ Returns a graph (as a dictionary) of the possible token
        types successions.
        Not relevant for Octuple.

        :return: the token types transitions dictionary
        """
        return {}  # not relevant for this encoding

    def token_types_errors(self, tokens: List[List[int]]) -> float:
        """ Checks if a sequence of tokens is constituted of good token values and
        returns the error ratio (lower is better).
        The token types are always the same in Octuple so this methods only checks
        if their values are correct:
            - a bar token value cannot be < to the current bar (it would go back in time)
            - same for positions
            - a pitch token should not be present if the same pitch is already played at the current position

        :param tokens: sequence of tokens to check
        :return: the error ratio (lower is better)
        """
        err = 0
        current_bar = current_pos = -1
        current_pitches = []

        for token in tokens:
            has_error = False
            bar_value = int(self.vocab.token_to_event[token[5]].split('_')[1])
            pos_value = int(self.vocab.token_to_event[token[4]].split('_')[1])
            pitch_value = int(self.vocab.token_to_event[token[0]].split('_')[1])

            # Bar
            if bar_value < current_bar:
                has_error = True
            elif bar_value > current_bar:
                current_bar = bar_value
                current_pos = -1
                current_pitches = []

            # Position
            if pos_value < current_pos:
                has_error = True
            elif pos_value > current_pos:
                current_pos = pos_value
                current_pitches = []

            # Pitch
            if pitch_value in current_pitches:
                has_error = True
            else:
                current_pitches.append(pitch_value)

            if has_error:
                err += 1

        return err / len(tokens)
