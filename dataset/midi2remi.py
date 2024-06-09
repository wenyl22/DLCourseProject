import sys
import pickle
import os
import glob
import random
import shutil
import math
import struct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import read_items, quantize_items, extract_chords, group_items, item2event, Event

def add(event, ev2id, id2ev):
    if '{}_{}'.format(event.name, event.value) not in ev2id:
        id = len(ev2id)
        ev2id['{}_{}'.format(event.name, event.value)] = id
        id2ev[id] = '{}_{}'.format(event.name, event.value)

def quantize(event, ev2id):
    qt = 1
    if event.name == 'Note_Duration':
        qt = 60
    for i in range(100):
        if '{}_{}'.format(event.name, event.value - i * qt) in ev2id:
            event.value = event.value - i
            break
        if '{}_{}'.format(event.name, event.value + i * qt) in ev2id:
            event.value = event.value + i
            break
    return event

def midi2remi(src, ev2id, id2ev, force = False):
    with open(src, 'rb') as file:
        header = file.read(8)
        _, size = struct.unpack('>4sL', header)
        data = file.read(size)
        _, _, ticks_per_beat = struct.unpack('>hhh', data[:6])
    note_items, tempo_items = read_items(src, ticks_per_beat=ticks_per_beat)
    note_items = quantize_items(note_items, ticks=int(round(ticks_per_beat / 4)))
    chord_items = extract_chords(note_items)
    items = chord_items + tempo_items + note_items
    max_time = note_items[-1].end
    groups = group_items(items, max_time, ticks_per_bar=ticks_per_beat * 4)
    events = item2event(groups)
    bar_pos = []
    dicts = []
    for j, event in enumerate(events):
        if force:
            add(event, ev2id, id2ev)
            assert('{}_{}'.format(event.name, event.value) in ev2id)
        elif '{}_{}'.format(event.name, event.value) not in ev2id:
            event = quantize(event, ev2id)
        if event.name == 'Bar':
            bar_pos.append(j)
        dict = {'name': event.name, 'value': event.value}
        dicts.append(dict)
    return bar_pos, dicts

def main():
    ev2id = {}
    id2ev = {}
    paths = ['./midi/anime/', './midi/classic/*/', './midi/jazz/midi/studio/*/*/', './midi/pop/pop_piano/']
    dst = './processed_midi'
    train_files = []
    test_files = []
    if os.path.exists(dst):
       shutil.rmtree(dst)
    os.makedirs(dst)
    if not os.path.exists('./pickles'):
        os.makedirs('./pickles')
    if os.path.exists('./pickles/remi_vocab.pkl'):
        ev2id, id2ev = pickle.load(open('./pickles/remi_vocab.pkl', 'rb'))
        os.remove('./pickles/remi_vocab.pkl')
    if os.path.exists('./pickles/train_pieces.pkl'):
        train_files = pickle.load(open('./pickles/train_pieces.pkl', 'rb'))
        os.remove('./pickles/train_pieces.pkl')
    if os.path.exists('./pickles/val_pieces.pkl'):
        test_files = pickle.load(open('./pickles/val_pieces.pkl', 'rb'))
        os.remove('./pickles/val_pieces.pkl')
    if os.path.exists('./pickles/test_pieces.pkl'):
        os.remove('./pickles/test_pieces.pkl')

    for path in paths:
        src_files = glob.glob(path + '*.mid')
        src_files= src_files + (glob.glob(path + '*.midi'))
        random.shuffle(src_files)
        print("{} events found in {}".format(len(src_files), path))
        cnt = 0
        dup = math.floor(800 / len(src_files))
        dup = max(dup, 1)
        for i, src in enumerate(src_files):
            if cnt > 800:
                break
            if i % 50 == 0:
                print("Processing {}/{}".format(i, len(src_files)))
            bar_pos, dicts = midi2remi(src, ev2id, id2ev, force = True)
            if "classic" in path:
                dst_file = str(i) +'_classic.pkl'
            elif "pop" in path:
                dst_file = str(i) +'_pop.pkl'
            elif "jazz" in path:
                dst_file = str(i) +'_jazz.pkl'
            elif "anime" in path:
                dst_file = str(i) +'_anime.pkl'
            with open(dst + '/' + dst_file, 'wb') as f: 
                pickle.dump((bar_pos, dicts), f)
            if random.random() < 0.8:
                for _ in range(dup):
                    train_files.append(dst_file)
            else:
                for _ in range(dup):
                    test_files.append(dst_file)
            cnt += dup
    event = Event(name = 'EOS', value = 'None', time = 0, text = 'EOS')
    add(event, ev2id, id2ev)

    with open('./pickles/remi_vocab.pkl', 'wb') as f:
        pickle.dump((ev2id, id2ev), f)
    with open('./pickles/train_pieces.pkl', 'wb') as f:
        pickle.dump(train_files, f)
    with open('./pickles/test_pieces.pkl', 'wb') as f:
        pickle.dump(test_files, f)
    with open('./pickles/val_pieces.pkl', 'wb') as f:
        pickle.dump(test_files, f)
if __name__ == '__main__':
    main()
