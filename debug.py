import pickle
import os
from utils.utils import read_items, quantize_items, extract_chords, group_items, item2event, Event
from dataset.remi2midi import remi2midi
ev2id = {}
id2ev = {}

def add(event):
    if '{}_{}'.format(event.name, event.value) not in ev2id:
        id = len(ev2id)
        ev2id['{}_{}'.format(event.name, event.value)] = id
        id2ev[id] = '{}_{}'.format(event.name, event.value)

def main():
    src = '/local_data/wyl/DLCourseProject/midi/pop/pop_piano/001.midi'
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
    dst_file  = './event.pkl'
    with open(dst_file, 'wb') as f: 
        pickle.dump((bar_pos, dicts), f)

    bar, ev = pickle.load(open('./event.pkl', 'rb'))
    print(ev)
    midi = remi2midi(ev, output_midi_path='./a.midi', is_full_event=True)

    # p =  pickle.load(open('./pickles/remi_vocab.pkl', 'rb'))
    # print(p)
    
    # if os.path.exists('./pickles/remi_vocab.pkl'):
    #     os.remove('./pickles/remi_vocab.pkl')
    # src = ['./pickles/train_pieces.pkl', './pickles/val_pieces.pkl']
    # for s in src:
    #     with open(s, 'rb') as f:
    #         files = pickle.load(f)
    #     for file in files:
    #         with open(file, 'rb') as f:
    #             bar_pos, events = pickle.load(f)
    #         for event in events:
    #             ev = Event(name = event['name'], value = event['value'], time = 0, text = 'EOS')
    #             add(ev)
    # event = Event(name = 'EOS', value = 'None', time = 0, text = 'EOS')
    # add(event)
    # event = Event(name = 'Bar', value = 'None', time = 0, text = 'Bar')
    # add(event)
    # print(ev2id, id2ev)
    # with open('./pickles/remi_vocab.pkl', 'wb') as f:
    #     pickle.dump((ev2id, id2ev), f)
if __name__ == "__main__":
    main()