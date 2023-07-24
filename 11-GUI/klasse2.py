"""Definition der Klasse Koerper mit weiteren Methoden. """


class Koerper:
    def __init__(self, name, masse):
        self.name = name
        self.masse = masse

    def erdmassen(self):
        return self.masse / 5.9722e24

    def __str__(self):
        return f'Körper {self.name}: m = {self.masse} kg'


if __name__ == '__main__':
    jupiter = Koerper('Jupiter', 1.89813e27)
    print(jupiter)


