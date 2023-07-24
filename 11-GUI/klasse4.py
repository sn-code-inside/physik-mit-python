"""Definition einer Klasse Planet2, die von der Klasse Planet
abgeleitet wird. Hier wird die Methode __str__ überschrieben. """

from klasse3 import Planet


class Planet2(Planet):
    def __str__(self):
        return f'Planet {self.name}: m = {self.masse} kg'


if __name__ == '__main__':
    jupiter = Planet2('Jupiter', 1.89813e27,  6.9911e7)
    rho = jupiter.dichte()

    print(jupiter)
    print(f'Dichte: {rho/1e3} g/cm³')


