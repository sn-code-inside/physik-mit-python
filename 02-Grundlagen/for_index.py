"""For-Schleife über zwei Listen mit Indizierung. """

lst1 = [2, 4, 6, 8, 10]
lst2 = [3, 5, 7, 9, 11]

for i in range(len(lst1)):
    a = lst1[i]
    b = lst2[i]
    print(f'{a:3d}    {b:3d}')
