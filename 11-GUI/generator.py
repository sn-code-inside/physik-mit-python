"""Wirkungsweise von Generatoren. """


def g():
    yield 3
    yield 7
    yield 5


for k in g():
    print(k)
