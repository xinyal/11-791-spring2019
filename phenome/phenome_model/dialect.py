import torch
from torch.optim import Adam
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import librosa

from model import PhonemeModel


class DialectModel(nn.Module):
    def __init__(self, pt):
        phenome_model = torch.load(pt)
        