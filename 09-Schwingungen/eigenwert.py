"""Bestimmung von Eigenwerten und -vektoren mit NumPy. """

import numpy as np

omega_0 = 1.0
omega_k = 0.1

Lam = np.array([[omega_0 ** 2 + omega_k ** 2, -omega_k ** 2],
                [-omega_k ** 2, omega_0 ** 2 + omega_k ** 2]])

# Bestimme die Eigenwerte w und die Eigenvektoren v.
eigenwerte, eigenvektoren = np.linalg.eig(Lam)

# v[:, i] ist der i-te Eigenvektor. Damit die Funktion
# zip die Eigenwerte und Eigenvektoren richtig zuordnet,
# muss v transponiert werden.
for lam, vec in zip(eigenwerte, eigenvektoren.T):
    print(f'Eigenwert {lam:5f}: '
          f'Eigenvektor ({vec[0]: .3}, {vec[1]: .3})')
