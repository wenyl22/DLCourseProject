import sys
import utils.chord_recognition as chord_recognition
import numpy as np
import miditoolkit
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480

# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note', 
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# extract chord
def extract_chords(items):
    method = chord_recognition.MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return '{}_{}'.format(self.name, self.value)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Beat', 
                time=item.start,
                value=index,
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # pitch
                events.append(Event(
                    name='Note_Pitch',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note_Velocity',
                    time=item.start, 
                    value=velocity_index * 4,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note_Duration',
                    time=item.start,
                    value=(index + 1)* 60,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo < 30: tempo = 30
                if tempo >= 210: tempo = 209
                tempo_value = Event('Tempo', item.start, 
                    tempo, None)
                events.append(tempo_value)
    return events