# Scripts for running imagen-pytorch.

This is currently in development, so things will probably break.

## Setup:
```bash
python3 -m pip install imagen-pytorch
```

## Running Inference:

```bash
python3 imagen.py --imagen yourmodel.pth --tags "1girl, red_hair" --output red_hair.png
```

## Training:

Currently, this is set up to use danbooru-style tags such as:

```
1girl, blue_dress, super_artist
```

The dataloader expects a directory with images and tags laid out like this:

```
dataset/
   tags/img1.txt
   tags/img2.txt
   ...
   imgs/img1.png
   imgs/img2.png
```

The subdirectories doesn't really matter, only the filenames matter.

### To train:

```bash
python3 imagen.py --train --source /path/to/dataset --imagen yourmodel.pth
```

## gel_fetch.py

Included is a tool to fetch data from *booru-style websites and creates tag files 
in the expected format.


### Setup:
You will need GelbooruViewer and pybooru to run.

First clone GelbooruViewer into the deep-imagen repo:
```bash
cd path/to/deep-imagen/
git clone https://github.com/ArchieMeng/GelbooruViewer
````

```bash
python3 -m pip install git+https://github.com/LuqueDaniel/pybooru
```

### Usage

```bash
python3 gel_fetch.py --tags "holo" --txt holo/tags --img holo/imgs --danbooru
```


