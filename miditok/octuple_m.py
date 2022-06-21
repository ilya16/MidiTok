""" OctupleM encoding method, a modified Octuple encoding, as introduced in MusicBERT
https://arxiv.org/abs/2106.05630

"""
import json
import math
from pathlib import Path, PurePath
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from miditok import Octuple
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

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
    :param max_bar_embedding: maximum bar embedding (might increase during encoding)
    :param sos_eos_tokens: adds Start Of Sequence (SOS) and End Of Sequence (EOS) tokens to the vocabulary
    :param mask: will add a MASK token to the vocabulary (default: False)
    :param params: can be a path to the parameter (json encoded) file or a dictionary
    """
    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, bool] = ADDITIONAL_TOKENS,
                 max_bar_embedding: int = 256, sos_eos_tokens: bool = False, mask: bool = False, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens,
                         max_bar_embedding, sos_eos_tokens, mask, params)
        self._fill_unperformed_notes = True
        self._remove_duplicates = False
        self._expand_vocab_on_bar_overflow = False

        self._num_reserved = 1 + int(self._mask)  # pad + (mask)
        self._num_reserved_bar = self._num_reserved + 2 * int(self._sos_eos)  # pad + (mask) + (sos/eos)

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
                       '_sos_eos': self._sos_eos,
                       '_mask': self._mask,
                       'encoding': self.__class__.__name__,
                       'max_bar_embedding': self.max_bar_embedding,
                       'remove_duplicates': self._remove_duplicates},
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
            self.durations_ticks[midi.ticks_per_beat] = np.array([
                (beat * res + pos) * midi.ticks_per_beat // res if res > 0 else 0
                for beat, pos, res in self.durations
            ])

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

        tokens.sort(key=lambda x: (x[2].time, x[2].desc, x[2].value))  # Sort by time then track then pitch

        # Convert pitch events into tokens
        for time_step in tokens:
            time_step[2] = self.vocab[2].event_to_token[f'{time_step[2].type}_{time_step[2].value}']

        return tokens

    def preprocess_midi(self, midi: MidiFile):
        """ Will process a MIDI file to be used by the OctupleM encoding.
        Adds unperformed notes on a new track.

        :param midi: MIDI object to preprocess
        """
        # Insert unperformed notes on a new track
        if self._fill_unperformed_notes and midi.instruments[-1].name != 'Unperformed Notes':
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
        current_bar_addon = 0
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
                    if self._expand_vocab_on_bar_overflow:
                        self.vocab[0].add_event(f'Bar_{i}' for i in range(self.max_bar_embedding, current_bar + 1))
                        self.max_bar_embedding = current_bar + 1
                        current_bar_addon = 0
                    else:
                        current_bar_addon = current_bar - self.max_bar_embedding + 1
                        current_bar = self.max_bar_embedding - 1

            # Note attributes
            duration = note.end - note.start
            dur_index = np.argmin(np.abs(dur_bins - duration))
            event = [self.vocab[0].event_to_token[f'Bar_{current_bar}'] + current_bar_addon,
                     self.vocab[1].event_to_token[f'Position_{current_pos}'],
                     Event(type_='Pitch', time=note.start, value=note.pitch,
                           desc=-1 if track.is_drum else track.program),
                     self.vocab[3].event_to_token[f'Velocity_{note.velocity}'],
                     self.vocab[4].event_to_token[f'Duration_{".".join(map(str, self.durations[dur_index]))}']]

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
                event.append(self.vocab[self.token_types_order['Tempo']].event_to_token[f'Tempo_{current_tempo}'])

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
                event.append(self.vocab[self.token_types_order['TimeSig']].event_to_token[
                    f'TimeSig_{current_time_sig.numerator}/{current_time_sig.denominator}'
                ])

            # (Program)
            if self.additional_tokens['Program']:
                event.append(self.vocab[self.token_types_order['Program']].event_to_token[
                    f'Program_{-1 if track.is_drum else track.program}'
                ])

            events.append(event)

        return events

    def tokens_to_midi(self, tokens: List[List[int]], _=None, output_path: Optional[str] = None,
                       time_division: Optional[int] = TIME_DIVISION) -> MidiFile:
        """ Convert multiple sequences of tokens into a multitrack MIDI and save it.
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
        :return: the midi object (miditoolkit.MidiFile)
        """
        assert time_division % max(self.beat_res.values()) == 0, \
            f'Invalid time division, please give one divisible by {max(self.beat_res.values())}'
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_sample = time_division // max(self.beat_res.values())

        tokens = np.array(tokens) if isinstance(tokens, list) else tokens

        # Time values
        bars = tokens[:, self.token_types_order['Bar']] - self.zero_bar_token
        positions = tokens[:, self.token_types_order['Position']] - self.zero_token

        # Process Time Signature changes
        # Compute change positions
        time_sig_idx = self.token_types_order['TimeSig']
        time_sig_indices = np.where(np.diff(tokens[:, time_sig_idx]))[0] + 1
        time_sig_indices = np.concatenate([[0], time_sig_indices])

        # Get time signatures
        time_sigs = tokens[time_sig_indices, time_sig_idx]
        time_sigs = np.array(self.time_signatures)[time_sigs - self.zero_token]

        # Compute time signature ticks
        ticks_per_bar = (time_division * 4 * time_sigs[:, 0] / time_sigs[:, 1]).astype(int)
        time_sig_bars = bars[time_sig_indices]
        time_sig_ticks = np.cumsum(ticks_per_bar[:-1] * np.diff(time_sig_bars))
        time_sig_ticks = np.concatenate([[0], time_sig_ticks])

        # Build Time Signature objects
        time_sig_changes = [
            TimeSignature(int(time_sigs[i][0]), int(time_sigs[i][1]), int(time_sig_ticks[i]))
            for i in range(len(time_sigs))
        ]

        # Compute ticks for each bar
        bar_ids = np.arange(bars[-1] + 1)
        bar_time_sig_ids = np.searchsorted(time_sig_bars, bar_ids, side='right') - 1
        bar_ticks = np.concatenate([[0], np.cumsum(ticks_per_bar[bar_time_sig_ids])])

        # Compute note positions in ticks
        note_on_ticks = bar_ticks[bars] + positions * ticks_per_sample

        # Note attributes
        pitches = tokens[:, self.token_types_order['Pitch']] - self.zero_token
        pitches += self.pitch_range[0]

        velocities = tokens[:, self.token_types_order['Velocity']] - self.zero_token
        velocities = self.velocities[velocities]

        durations = tokens[:, self.token_types_order['Duration']] - self.zero_token

        if time_division not in self.durations_ticks:
            self.durations_ticks[time_division] = np.array([
                (beat * res + pos) * midi.ticks_per_beat // res if res > 0 else 0
                for beat, pos, res in self.durations
            ])

        durations = self.durations_ticks[time_division][durations]
        note_off_ticks = note_on_ticks + durations

        # Process Tempo changes
        tempo_idx = self.token_types_order['Tempo']
        tempo_indices = np.where(np.diff(tokens[:, tempo_idx]))[0] + 1
        tempo_indices = np.concatenate([[0], tempo_indices])

        tempos = tokens[tempo_indices, tempo_idx]
        tempos = self.tempos[tempos - self.zero_token]

        if len(tempos) > 0:
            # Compute beat positions to tie Tempo change to them
            num_beats_in_bar = time_sigs[:, 0]
            num_beats_in_bar[num_beats_in_bar == 6] = 2
            num_beats_in_bar[np.isin(num_beats_in_bar, (9, 18))] = 3
            num_beats_in_bar[np.isin(num_beats_in_bar, (12, 24))] = 4
            ticks_per_beat = ticks_per_bar / num_beats_in_bar

            max_beat = np.sum(np.diff(np.concatenate([time_sig_bars, [bars[-1] + 1]])) * num_beats_in_bar)
            beat_ids = np.arange(max_beat + 1)
            beat_time_sig_ids = np.searchsorted(time_sig_bars, beat_ids, side='right') - 1
            beat_ticks = np.concatenate([[0], np.cumsum(ticks_per_beat[beat_time_sig_ids])])

            # Tempo ticks and Tempo changes
            tempo_ticks = note_on_ticks[tempo_indices]  # Note: position at the start of the beat
            tempo_ticks = beat_ticks[np.searchsorted(beat_ticks, tempo_ticks)]
            tempo_ticks[0] = 0
        else:
            tempo_ticks = [0]

        tempo_changes = [
            TempoChange(int(tempos[i]), int(tempo_ticks[i]))
            for i in range(len(tempos))
        ]

        # Process Programs
        if self.additional_tokens['Program']:
            _zero_program = f'Program_0'
            program_idx = self.token_types_order['Program']
            programs = tokens[:, program_idx] - self.zero_token
        else:
            programs = np.zeros_like(pitches)

        # Create Notes
        if self.additional_tokens['Program']:
            tracks = dict([(n, []) for n in range(-1, 128)])

            for program in np.unique(programs):
                program_ids = np.where(programs == program)[0]
                tracks[int(program)] = [
                    Note(vel, pitch, start, end)
                    for vel, pitch, start, end in zip(
                        velocities[program_ids], pitches[program_ids],
                        note_on_ticks[program_ids], note_off_ticks[program_ids]
                    )
                ]
        else:
            tracks = {0: [
                Note(velocities[i], pitches[i], note_on_ticks[i], note_off_ticks[i])
                for i in range(len(pitches))
            ]}

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

        midi.max_tick = note_off_ticks.max() + 1

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def _create_vocabulary(self) -> List[Vocabulary]:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training

        :return: the vocabulary object
        """
        vocab = [Vocabulary({'PAD_None': 0}, sos_eos=self._sos_eos and i == 0, mask=self._mask) for i in range(5)]

        # BAR
        vocab[0].add_event(f'Bar_{i}' for i in range(self.max_bar_embedding))  # bar embeddings (positional encoding)

        # POSITION
        max_nb_beats = max(map(lambda ts: math.ceil(4 * ts[0] / ts[1]), self.time_signatures))
        nb_positions = max(self.beat_res.values()) * max_nb_beats
        vocab[1].add_event(f'Position_{i}' for i in range(nb_positions))

        # PITCH
        vocab[2].add_event(f'Pitch_{i}' for i in self.pitch_range)

        # VELOCITY
        self.velocities = np.concatenate(([0], self.velocities))  # allow 0 velocity (unperformed note)
        vocab[3].add_event(f'Velocity_{i}' for i in self.velocities)

        # DURATION
        self.durations = [(0, 0, 0)] + self.durations  # allow 0 duration
        vocab[4].add_event(f'Duration_{".".join(map(str, duration))}' for duration in self.durations)

        # TEMPO
        if self.additional_tokens['Tempo']:
            vocab.append(Vocabulary({'PAD_None': 0}, mask=self._mask))
            vocab[-1].add_event(f'Tempo_{i}' for i in self.tempos)

        # TIME_SIGNATURE
        if self.additional_tokens['TimeSignature']:
            vocab.append(Vocabulary({'PAD_None': 0}, mask=self._mask))
            vocab[-1].add_event(f'TimeSig_{i[0]}/{i[1]}' for i in self.time_signatures)

        # PROGRAM
        if self.additional_tokens['Program']:
            vocab.append(Vocabulary({'PAD_None': 0}, mask=self._mask))
            vocab[-1].add_event(f'Program_{program}' for program in range(-1, 128))

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

        bar_idx = self.token_types_order['Bar']
        pos_idx = self.token_types_order['Position']
        pitch_idx = self.token_types_order['Pitch']
        for token in tokens:
            has_error = False
            bar_value = int(self.vocab[bar_idx].token_to_event[token[bar_idx]].split('_')[1])
            pos_value = int(self.vocab[pos_idx].token_to_event[token[pos_idx]].split('_')[1])
            pitch_value = int(self.vocab[pitch_idx].token_to_event[token[pitch_idx]].split('_')[1])

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

    @property
    def sizes(self):
        return {k: len(v) for k, v in zip(self.token_types_order, self.vocab)}

    @property
    def zero_token(self):
        return self._num_reserved

    @property
    def zero_bar_token(self):
        return self._num_reserved_bar
