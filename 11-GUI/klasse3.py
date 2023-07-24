"""Definition einer Klasse Planet, die von der Klasse
Koerper abgeleitet wird. """

import math
from klasse2 import Koerper


class Planet(Koerper):
    def __init__(self, name, masse, radius):
        super().__init__(name, masse)
        self.radius = radius

    def volumen(self):
        return 4 / 3 * math.pi * self.radius ** 3

    def dichte(self):
        return self.masse / self.volumen()


if __name__ == '__main__':
    jupiter = Planet('Jupiter', 1.89813e27,  6.9911e7)
    rho = jupiter.dichte()

    print(jupiter)
    print(f'Dichte: {rho/1e3} g/cm³')


