"""For-Schleife über zwei Listen mit zip. """

lst1 = [2, 4, 6, 8, 10]
lst2 = [3, 5, 7, 9, 11]

for a, b in zip(lst1, lst2):
    print(f'{a:3d}    {b:3d}')
