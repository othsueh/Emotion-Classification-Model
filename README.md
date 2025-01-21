# Emotion classification on MSP-Podcast(Temporarily)
## Introduction
The repository is for the emotion classification task on MSP-Podcast dataset. The dataset is not public yet, so you have to prepare corpus file first. The code is written in PyTorch.
## File Structure
```
.
├── config.toml.example
├── display.py
├── net
├── normal.py
├── PerformanceLog
├── README.md
├── test.ipynb
├── train.py
├── todo.todo
└── utils.py
```
### Files
`config.toml.example` : Example configuration file. Copy this file to `config.toml` and modify it to your needs.

`train.py` : Main train file.

`utils.py` : Utility functions. Store function like PyTorch dataset, mixup, train report, etc.

`display.py` : Display functions, for terminal display.

`test.ipynb` : Jupyter notebook for testing/writing scratchy code.

`todo.todo` : My working log.

### Folders
`net` : Store all the network architecture files here.

`PerformanceLog` : Store all the performance logs here.