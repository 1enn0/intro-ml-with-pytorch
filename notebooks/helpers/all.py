import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import ipywidgets as widgets
import librosa
import tempfile

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torchvision

from fastai.vision.all import *
from fastai.data.all import *
from fastai.metrics import accuracy
from fastaudio.core.all import *
from fastaudio.augment.signal import AudioPadType

from .dirs import *
from .train_utils import *
from .helpers import *
