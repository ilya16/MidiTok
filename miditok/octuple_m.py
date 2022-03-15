""" OctupleM encoding method, a modified Octuple encoding, as introduced in MusicBERT
https://arxiv.org/abs/2106.05630

"""
import copy
import json
import math
from fractions import Fraction
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditok import Octuple
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature, Marker

from .midi_tokenizer_base import Vocabulary, Event
from .constants import *


class OctupleM(Octuple):
    """ OctupleM: modified Octuple encoding method, introduced in MusicBERT
    https://arxiv.org/abs/2106.05630

    :param pitch_range: range of used MIDI pitches
    :param beat_res: beat resolutions, with the form:
            {(beat_x1, beat_x2): beat_res_1, (beat_x2, beat_x3): beat_res_2, ...}
            The keys of the dict are tuples indicating a range of beats, ex 0 to 3 for the first bar
            The values are the resolution, in samples per beat, of the given range, ex 8
    :param nb_velocities: number of velocity bins
    :param additional_tokens: specifies additional tokens (time signature, tempo)
    :param absolute_token_ids: encode token ids in Vocabulary absolute values instead of token type ids
    :param sos_eos_tokens: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param mask_token: adds Mask (MASK) token to the vocabulary
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 absolute_token_ids: bool = False, sos_eos_tokens: bool = False, mask_token: bool = False, params=None):
        self.absolute_token_ids = absolute_token_ids
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens, mask_token, params)
        self._compute_token_types_indexes()
        self.fill_unperformed_notes = True
        self.remove_duplicates = False

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
                       'nb_velocities': len(self.velocities) - 1,
                       'additional_tokens': self.additional_tokens,
                       'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding,
                       'absolute_token_ids': self.absolute_token_ids,
                       'remove_duplicates': self.remove_duplicates},
                      outfile)

    def midi_to_tokens(self, midi: MidiFile) -> List[List[int]]:
        """ Override the parent class method
        Converts a MIDI file in a tokens representation, a sequence of "time steps".
        A time step is a list of tokens where:
            (list index: token type)
            0: Bar
            1: Position
            2: Pitch
            3: Velocity
            4: Duration
            (5: Tempo)
            (6: TimeSignature)
            (7: Program (track))

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
                                      'key_sig_changes': midi.key_signature_changes,
                                      'bar_shift': 0}

        for m in midi.markers:
            if m.text.startswith('Anacrusis'):
                self.current_midi_metadata['bar_shift'] = m.time
                break

        # Convert each track to tokens
        tokens = []
        for track in midi.instruments:
            tokens += self.track_to_tokens(track)

        tokens.sort(key=lambda x: (x[2].time, x[2].desc, x[2].value))  # Sort by time then track then pitch

        # Convert pitch events into tokens
        for time_step in tokens:
            time_step[2] = self.vocab.event_to_token[f'{time_step[2].type}_{time_step[2].value}']

        if not self.absolute_token_ids:
            tokens = self.to_type_ids(tokens, inplace=True)  # encoding works only with absolute ids

        return tokens

    def preprocess_midi(self, midi: MidiFile):
        """ Will process a MIDI file to be used by the OctupleM encoding.
        Processes anacrusis and first downbeat, adds unperformed notes on a new track.

        :param midi: MIDI object to preprocess
        """
        # Process anacrusis if any
        for m in midi.markers:
            if m.text.startswith('Anacrusis'):
                num, denom = map(int, m.text.split('_')[-1].split('/'))
                time_sig = midi.time_signature_changes[0]
                if (num, denom) != (time_sig.numerator, time_sig.denominator):
                    midi.time_signature_changes = midi.time_signature_changes[1:]
                break

        # Insert unperformed notes on a new track
        if self.fill_unperformed_notes and midi.instruments[-1].name != 'Unperformed Notes':
            notes = []
            for m in midi.markers:
                if m.text.startswith('NoteS'):
                    pitch, start_tick, end_tick = map(int, m.text.split('_')[1:])
                    notes.append(Note(0, pitch, start_tick, end_tick))
            if notes:
                midi.instruments.append(Instrument(0, False, 'Unperformed Notes'))
                midi.instruments[-1].notes = notes

        # Do base preprocessing
        super().preprocess_midi(midi)

    def track_to_tokens(self, track: Instrument) -> List[List[Union[Event, int]]]:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        A time step is a list of tokens where:
            (list index: token type)
            0: Bar
            1: Position
            2: Pitch (as an Event object for sorting purpose afterwards)
            3: Velocity
            4: Duration
            (5: Tempo)
            (6: TimeSignature)
            (7: Program (track))

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

        bar_shift = self.current_midi_metadata['bar_shift']

        for note in track.notes:
            # Positions and bars
            if note.start != current_tick:
                pos_index = int(((note.start - current_time_sig_tick - bar_shift) % ticks_per_bar) / ticks_per_sample)
                current_tick = note.start
                current_bar = current_time_sig_bar + (current_tick - current_time_sig_tick - bar_shift) // ticks_per_bar
                if bar_shift > 0:
                    current_bar += 1
                current_pos = pos_index

                # Check bar embedding limit, update if needed
                if self.max_bar_embedding <= current_bar:
                    self.vocab.add_event(f'Bar_{i}' for i in range(self.max_bar_embedding, current_bar + 1))
                    self.max_bar_embedding = current_bar + 1

            # Note attributes
            duration = note.end - note.start
            dur_index = np.argmin(np.abs(dur_bins - duration))
            event = [self.vocab.event_to_token[f'Bar_{current_bar}'],
                     self.vocab.event_to_token[f'Position_{current_pos}'],
                     Event(type_='Pitch', time=note.start, value=note.pitch,
                           desc=-1 if track.is_drum else track.program),
                     self.vocab.event_to_token[f'Velocity_{note.velocity}'],
                     self.vocab.event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}']]

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
                            bar_shift = 0  # only for the first time signature
                        elif time_sig.time > note.start:
                            break  # this time signature change is beyond the current time step, we break the loop
                event.append(self.vocab.event_to_token[f'TimeSig_{current_time_sig.numerator}/{current_time_sig.denominator}'])

            # (Program)
            if self.additional_tokens['Program']:
                event.append(self.vocab.event_to_token[f'Program_{-1 if track.is_drum else track.program}'])

            events.append(event)

        return events

    def tokens_to_midi(self, tokens: List[List[int]], _=None, output_path: Optional[str] = None,
                       time_division: Optional[int] = TIME_DIVISION, bar_shift: Optional[int] = 0) -> MidiFile:
        """ Override the parent class method
        Convert multiple sequences of tokens into a multitrack MIDI and save it.
        The tokens will be converted to event objects and then to a miditoolkit.MidiFile object.
        A time step is a list of tokens where:
            (list index: token type)
            0: Pitch
            1: Velocity
            2: Duration
            3: Bar
            4: Position
            (5: Tempo)
            (6: TimeSignature)
            (7: Program (track))

        :param tokens: list of lists of tokens to convert, each list inside the
                       first list corresponds to a track
        :param _: unused, to match parent method signature
        :param output_path: path to save the file (with its name, e.g. music.mid),
                        leave None to not save the file
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create)
        :param bar_shift: shift of the second bar in ticks/beat (of the MIDI to create)
        :return: the midi object (miditoolkit.MidiFile)
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_sample = time_division // max(self.beat_res.values())

        if not self.absolute_token_ids:
            tokens = self.to_absolute_ids(tokens, inplace=False)  # decoding works only with absolute ids

        events = self.tokens_to_events(tokens[0])
        if self.additional_tokens['Tempo']:
            tempo = int(events[self.token_types_order['Tempo']].value)
        else:  # default
            tempo = TEMPO
        tempo_changes = [TempoChange(tempo, 0)]

        if self.additional_tokens['TimeSignature']:
            time_sig = self.parse_token_time_signature(events[self.token_types_order['TimeSig']].value)
        else:  # default
            time_sig = TIME_SIGNATURE
        time_sig_changes = [TimeSignature(*time_sig, 0)]
        ticks_per_bar = self.compute_ticks_per_bar(time_sig_changes[0], time_division)
        ticks_shift = ticks_per_bar - bar_shift if bar_shift > 0 else bar_shift

        current_tempo_bar = 0
        current_time_sig_tick = 0
        current_time_sig_bar = 0

        if self.additional_tokens['Program']:
            tracks = dict([(n, []) for n in range(-1, 128)])
        else:
            tracks = {0: []}

        for time_step in tokens:
            events = self.tokens_to_events(time_step)

            # Note attributes
            pitch = int(events[self.token_types_order['Pitch']].value)
            vel = int(events[self.token_types_order['Velocity']].value)
            duration = self._token_duration_to_ticks(events[self.token_types_order['Duration']].value, time_division)

            # Time values
            current_bar = int(events[self.token_types_order['Bar']].value)
            current_pos = int(events[self.token_types_order['Position']].value)
            current_bar_tick = current_time_sig_tick + (current_bar - current_time_sig_bar) * ticks_per_bar
            current_tick = current_bar_tick + current_pos * ticks_per_sample

            if ticks_shift > 0:
                current_bar_tick = max(0, current_bar_tick - ticks_shift)
                current_tick = max(0, current_tick - ticks_shift)

            # Track
            program = 0
            if self.additional_tokens['Program']:
                program = int(events[self.token_types_order['Program']].value)

            # Append the created note
            tracks[program].append(Note(vel, pitch, current_tick, current_tick + duration))

            # Tempo, adds a TempoChange if necessary
            if self.additional_tokens['Tempo']:
                tempo = int(events[self.token_types_order['Tempo']].value)
                if current_bar > current_tempo_bar and tempo != tempo_changes[-1].tempo:
                    current_tempo_bar = current_bar
                    tempo_changes.append(TempoChange(tempo, current_bar_tick))  # position at the start of the bar

            # Time Signature, adds a TimeSignatureChange if necessary
            if self.additional_tokens['TimeSignature']:
                time_sig = self.parse_token_time_signature(events[self.token_types_order['TimeSig']].value)
                if time_sig != (time_sig_changes[-1].numerator, time_sig_changes[-1].denominator):
                    current_time_sig_tick = current_bar_tick
                    current_time_sig_bar = current_bar
                    time_sig = TimeSignature(*time_sig, current_time_sig_tick)
                    ticks_per_bar = self.compute_ticks_per_bar(time_sig, time_division)
                    ticks_shift = 0
                    time_sig_changes.append(time_sig)

        # Tempos
        midi.tempo_changes = tempo_changes

        # Time Signatures
        if bar_shift > 0:
            time_sig = time_sig_changes[0]
            ticks_per_bar = self.compute_ticks_per_bar(time_sig, time_division)
            cut_time_sig = Fraction(time_sig.numerator / time_sig.denominator * bar_shift / ticks_per_bar)
            cut_time_sig = cut_time_sig.limit_denominator(64)
            time_sig_changes.insert(0, TimeSignature(cut_time_sig.numerator, cut_time_sig.denominator, 0))
            time_sig.time = bar_shift
            midi.markers = [Marker(text=f'Anacrusis_{time_sig.numerator}/{time_sig.denominator}', time=bar_shift)]

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

    def _create_vocabulary(self, sos_eos_tokens: bool = False, mask_token: bool = False) -> Vocabulary:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :param sos_eos_tokens: will include Start Of Sequence (SOS) and End Of Sequence (tokens)
        :param mask_token: will include Mask (MASK) token
        :return: the vocabulary object
        """
        vocab = Vocabulary({'PAD_None': 0})

        # MASK
        if mask_token:
            vocab.add_mask()

        # SOS & EOS
        if sos_eos_tokens:
            vocab.add_sos_eos()

        # PITCH
        vocab.add_event(f'Pitch_{i}' for i in self.pitch_range)

        # VELOCITY
        self.velocities = np.concatenate(([0], self.velocities))  # allow 0 velocity (unperformed note)
        vocab.add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        self.durations = [(0, 0, 0)] + self.durations  # allow 0 duration
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
        if self.additional_tokens['Program']:
            vocab.add_event(f'Program_{program}' for program in range(-1, 128))

        # BAR --- MUST BE LAST IN DIC AS THIS MIGHT BE INCREASED
        vocab.add_event(f'Bar_{i}' for i in range(self.max_bar_embedding))  # bar embeddings (positional encoding)

        self._compute_token_types_order()

        return vocab

    def _compute_token_types_order(self):
        """ Creates the mapping of token types and their order in the encoding."""
        token_types_order = {
            'Bar': 0,
            'Position': 1,
            'Pitch': 2,
            'Velocity': 3,
            'Duration': 4,
        }

        if self.additional_tokens['Tempo']:
            token_types_order['Tempo'] = max(token_types_order.values()) + 1

        if self.additional_tokens['TimeSignature']:
            token_types_order['TimeSig'] = max(token_types_order.values()) + 1

        if self.additional_tokens['Program']:
            token_types_order['Program'] = max(token_types_order.values()) + 1

        self.token_types_order = token_types_order

    def _compute_token_types_indexes(self):
        """ Computes token type index ranges for absolute and type index encodings."""
        self._absolute_token_types_indexes = self.vocab._token_types_indexes

        mask_token = 'MASK' in self._absolute_token_types_indexes
        sos_eos_tokens = 'SOS' and 'EOS' in self._absolute_token_types_indexes

        self._token_types_indexes = copy.copy(self._absolute_token_types_indexes)
        for token_type, abs_token_indexes in self._absolute_token_types_indexes.items():
            if token_type in self.token_types_order:
                min_index = abs_token_indexes[0]
                index_shift = 1 + int(mask_token) + 2 * int(token_type == 'Bar' and sos_eos_tokens)
                self._token_types_indexes[token_type] = [idx - min_index + index_shift for idx in abs_token_indexes]

    def _convert_tokens(self, tokens: List[List[int]], to_absolute: bool = False,
                        inplace: bool = False) -> List[List[int]]:
        """ Converts a token sequence between absolute and type ids.

        :param tokens: sequence of tokens to be converted
        :param to_absolute: convert from type to absolute ids, otherwise from absolute to type ids
        :param inplace: modify the sequence inplace
        :return: sequence of converted tokens
        """
        tokens = tokens if inplace else copy.deepcopy(tokens)

        source_indexes = self._token_types_indexes if to_absolute else self._absolute_token_types_indexes
        target_indexes = self._absolute_token_types_indexes if to_absolute else self._token_types_indexes

        for token_type, token_type_index in self.token_types_order.items():
            index_shift = target_indexes[token_type][0] - source_indexes[token_type][0]
            for time_step in tokens:
                if time_step[token_type_index] >= source_indexes[token_type][0]:
                    time_step[token_type_index] += index_shift

        return tokens

    def to_type_ids(self, tokens: List[List[int]], inplace: bool = False) -> List[List[int]]:
        """ Converts a token sequence with absolute token ids
        into a token sequence with token type absolute token ids.

        :param tokens: sequence of tokens to be converted
        :param inplace: modify the sequence inplace
        :return: sequence of converted tokens
        """
        return self._convert_tokens(tokens, to_absolute=False, inplace=inplace)

    def to_absolute_ids(self, tokens: List[List[int]], inplace: bool = False) -> List[List[int]]:
        """ Converts a token sequence with absolute token ids
        into a token sequence with token type absolute token ids.

        :param tokens: sequence of tokens to be converted
        :param inplace: modify the sequence inplace
        :return: sequence of converted tokens
        """
        return self._convert_tokens(tokens, to_absolute=True, inplace=inplace)

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
            bar_value = int(self.vocab.token_to_event[token[self.token_types_order['Bar']]].split('_')[1])
            pos_value = int(self.vocab.token_to_event[token[self.token_types_order['Position']]].split('_')[1])
            pitch_value = int(self.vocab.token_to_event[token[self.token_types_order['Pitch']]].split('_')[1])

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

    def validate_midi_time_signatures(self, midi: MidiFile) -> bool:
        """ Checks if MIDI files contains only time signatures supported by the encoding.

        :param midi: MIDI file
        :return: boolean indicating whether MIDI file could be processed by the Encoding
        """
        time_signatures = midi.time_signature_changes
        if contains_anacrusis(midi):  # hide anacrusis' time signature
            midi.time_signature_changes = time_signatures[1:]

        is_valid = super().validate_midi_time_signatures(midi)
        midi.time_signature_changes = time_signatures

        return is_valid


def contains_anacrusis(midi: MidiFile):
    """ Checks if MIDI file contains anacrusis

    :param midi: MIDI file
    :return: boolean indicating whether MIDI contains anacrusis marker
    """
    for m in midi.markers:
        if m.text.startswith('Anacrusis'):
            return True
    return False
