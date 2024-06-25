# DLCourseProject

## Introduction
This is the course project of "Deep Learning" in IIIS, Tsinghua University. The goal of the project is to implement a deep learning model that can generate music with hierarchical control. Here we implement a model with two types of low-level control(rhythm and polyphony) and one type of high-level control(style). You can listen to the generated music in `./demo` folder.

## Dataset
The music used in this project is in midi format. You can use your own midi files, then run the following command to convert them to the remi format, which the model can use:
```bash
python dataset/midi2remi.py <midi_file>
python dataset/attrbutes.py
```


## Generate Music
To generate music, you can run the following command:
```bash
python generate.py <config_file>
```
In the config file, you can specify the generation mode, where we provide three modes: `unconditional`, `interpolate`, and `transfer`. Unconditional generation is to generate music from scratch, interpolate is to generate continuous transfer from one piece of music to another, and transfer is to transfer the style/rhythmic intensity/polyphony of one piece of music.

## Train the Model
To train the model, you can run the following command:
```bash
python train.py <config_file>
```
