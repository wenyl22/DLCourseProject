import pickle
import os
from utils.utils import read_items, quantize_items, extract_chords, group_items, item2event, Event
from dataset.remi2midi import remi2midi

def add(event, ev2id, id2ev):
    if '{}_{}'.format(event.name, event.value) not in ev2id:
        id = len(ev2id)
        ev2id['{}_{}'.format(event.name, event.value)] = id
        id2ev[id] = '{}_{}'.format(event.name, event.value)

def main():
    src = '/local_data/wyl/DLCourseProject/1.mid'
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
        # add(event)
        if event.name == 'Bar':
            bar_pos.append(j)
        dict = {'name': event.name, 'value': event.value}
        dicts.append(dict)
    dst_file  = './event.pkl'
    with open(dst_file, 'wb') as f: 
        pickle.dump((bar_pos, dicts), f)
    dst_files = ['./event.pkl']
    with open('./files', 'wb') as f: 
        pickle.dump(dst_files, f)

    # bar, ev = pickle.load(open('./event.pkl', 'rb'))
    # print(ev)
    # midi = remi2midi(ev, output_midi_path='./a.midi', is_full_event=True)
    
    # src = ['./pickles/train_pieces.pkl', './pickles/val_pieces.pkl', './pickles/test_pieces.pkl']
    # dst = './processed_midi/'
    # for s in src:
    #     with open(s, 'rb') as f:
    #         files = pickle.load(f)
    #     for file in files:
    #         with open(dst + file, 'rb') as f:
    #             bar_pos, events = pickle.load(f)
    #         for event in events:
    #             ev = Event(name = event['name'], value = event['value'], time = 0, text = 'EOS')
    #             assert('{}_{}'.format(event['name'], event['value']) in ev2id)
                # add(ev)
        # with open(s, 'wb') as f:
        #     pickle.dump(n_files, f) 
    # event = Event(name = 'EOS', value = 'None', time = 0, text = 'EOS')
    # add(event)
    # print(len(ev2id))
    # pr
if __name__ == "__main__":
    main()