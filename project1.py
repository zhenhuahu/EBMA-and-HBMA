import numpy as np
import cv2

from EBMA import ebma
from EBMA_halfPel import ebma_halfPel
from HBMA import hbma

import display as dp

eBMA = ebma('akiyo_352x288_30.yuv', 16, 8)
mvx, mvy = eBMA.match()

#eBMA_halfPel = ebma_halfPel('akiyo_352x288_30.yuv', 16, 8)
#mvx, mvy = eBMA_halfPel.match()

#hBMA = hbma('akiyo_352x288_30.yuv', 16, 8)
#mvx, mvy = hBMA.match()