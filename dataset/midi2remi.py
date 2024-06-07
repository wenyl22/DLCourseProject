import sys
import pickle
import os
import glob
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import read_items, quantize_items, extract_chords, group_items, item2event, Event

ev2id = {}
id2ev = {}
def add(event):
    if '{}_{}'.format(event.name, event.value) not in ev2id:
        id = len(ev2id)
        ev2id['{}_{}'.format(event.name, event.value)] = id
        id2ev[id] = '{}_{}'.format(event.name, event.value)
def main():
    paths =['./midi/pop/*/']
    dst = './processed_midi/'
    train_files = []
    test_files = []
    if not os.path.exists(dst):
        os.makedirs(dst)
    if not os.path.exists('./pickles'):
        os.makedirs('./pickles')
    if os.path.exists('./pickles/remi_vocab.pkl'):
        with open('./pickles/remi_vocab.pkl', 'rb') as f:
            ev2id, id2ev = pickle.load(f)
    if os.path.exists('./pickles/train_pieces.pkl'):
        with open('./pickles/train_pieces.pkl', 'rb') as f:
            train_files = pickle.load(f)
    if os.path.exists('./pickles/val_pieces.pkl'):
        with open('./pickles/val_pieces.pkl', 'rb') as f:
            test_files = pickle.load(f)
    os.remove('./pickles/remi_vocab.pkl')
    os.remove('./pickles/train_pieces.pkl')
    os.remove('./pickles/test_pieces.pkl')
    os.remove('./pickles/val_pieces.pkl')
    for path in paths:
        src_files = glob.glob(path + '*.mid')
        src_files= src_files + (glob.glob(path + '*.midi'))
        random.shuffle(src_files)
        print("{} events found in {}".format(len(src_files), path))
        for i, src in enumerate(src_files):
            if i % 50 == 0:
                print("Processing {}/{}".format(i, len(src_files)))
            note_items, tempo_items = read_items(src)
            note_items = quantize_items(note_items)
            chord_items = extract_chords(note_items)
            items = chord_items + tempo_items + note_items
            max_time = note_items[-1].end
            groups = group_items(items, max_time)
            events = item2event(groups)
            bar_pos = []
            dicts = []
            for j, event in enumerate(events):
                add(event)
                if event.name == 'Bar':
                    bar_pos.append(j)
                dict = {'name': event.name, 'value': event.value}
                dicts.append(dict)
            if "classic" in path:
                dst_file = dst + str(i) +'_classic.pkl'
            else:
                dst_file = dst + str(i) +'_pop.pkl'
            if i < len(src_files) * 0.8:
                train_files.append(dst_file)
            else:
                test_files.append(dst_file)
            with open(dst_file, 'wb') as f: 
                pickle.dump((bar_pos, dicts), f)
    event = Event(name = 'EOS', value = 'None', time = 0, text = 'EOS')
    add(event)
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
