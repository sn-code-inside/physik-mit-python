﻿"""Lineare, progressive und degressive Federkennlinien. """

import numpy as np
import matplotlib.pyplot as plt

# Anfängliche Federhärte [N/m].
D0 = 40.0

# Konstante für den nichtlinearen Term [m].
alpha = 0.5


def F_nonlin(x, D0, alpha):
    """Definition der nichtlinearen Federkennlinie. """
    return D0 * alpha * np.expm1(np.abs(x)/alpha) * np.sign(x)


# Berechne die drei Kennlinien.
x = np.linspace(-0.75, 0.75, 500)
F_lin = D0 * x
F_pro = F_nonlin(x, D0, alpha)
F_deg = F_nonlin(x, D0, -alpha)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$-F$ [N]')
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(F_pro), np.max(F_pro))
ax.grid()

#  Plotte die Kennlinien.
ax.plot(x, F_pro, '-r', label='progressiv')
ax.plot(x, F_lin, '--k', label='linear')
ax.plot(x, F_deg, '-b', label='degressiv')
ax.legend()

# Zeige die Grafik an.
plt.show()
