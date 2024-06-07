import pickle
import os
from utils.utils import read_items, quantize_items, extract_chords, group_items, item2event, Event

ev2id = {}
id2ev = {}

def add(event):
    if '{}_{}'.format(event.name, event.value) not in ev2id:
        id = len(ev2id)
        ev2id['{}_{}'.format(event.name, event.value)] = id
        id2ev[id] = '{}_{}'.format(event.name, event.value)

def main():
    p =  pickle.load(open('./pickles/remi_vocab.pkl', 'rb'))
    print(p)
    
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