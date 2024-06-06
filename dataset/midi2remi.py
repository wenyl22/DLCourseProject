import sys
import pickle
import os
import glob
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
    path ='./classic_midi/*/*.midi'
    dst = './processed_midi/classic/'
    if not os.path.exists(dst):
        os.makedirs(dst)
    src_files = glob.glob(path)
    print("{} events found in {}".format(len(src_files), path))
    for i, src in enumerate(src_files):
        if i > 20:
            break
        if i % 10 == 0:
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
        dst_file = dst + str(i) + '.pkl'
        #print(dicts)
        with open(dst_file, 'wb') as f:  
            pickle.dump((bar_pos, dicts), f)
    event = Event(name = 'EOS', value = 'None', time = 0, text = 'EOS')
    add(event)
    with open('./pickles/remi_vocab.pkl', 'wb') as f:
        pickle.dump((ev2id, id2ev), f)
if __name__ == '__main__':
    main()
