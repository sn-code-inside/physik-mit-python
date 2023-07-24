"""Einfache Definition einer Klasse Koerper. """


class Koerper:
    def __init__(self, name, masse):
        self.name = name
        self.masse = masse


k = Koerper('Jupiter', 1.89813e27)
print(k.name)
print(k.masse)


