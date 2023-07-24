"""Das Sieb des Eratosthenes. """

import math
import numpy as np

# Zahl bis zu der nach Primzahlen gesucht werden soll.
n = 100

# Array das angibt, welche Zahlen als Primzahlen in Frage kommen.
prim = np.ones(n, dtype=bool)

# 0 und 1 sind keine Primzahlen.
prim[:2] = False

# Markiere nacheinander die Vielfachen von ganzen Zahlen.
for i in range(2, int(math.sqrt(n)+1)):
    prim[i * i::i] = False

# Primzahlen sind die Zahlen, die am Ende noch als prim
# markiert sind. Alternativ könnte man hier auch die Funktion
# np.where benutzen.
zahlen = np.arange(n)
primzahlen = zahlen[prim]

# Gib die Primzahlen aus.
for i in primzahlen:
    print(i)

