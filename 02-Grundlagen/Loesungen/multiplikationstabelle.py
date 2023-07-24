"""Gibt eine Multiplikationstabelle für das kleine 1x1 aus. """

import numpy as np

a = np.arange(1, 10) 
b = np.arange(1, 10)
b = b.reshape(-1, 1)
print(a * b)
