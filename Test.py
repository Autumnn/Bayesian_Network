a = [5.5, 5, 4.6, 2.9, 2.4, 2.4, 4.5, 4.5 , 4.8 ,1.8,2.2,2.2, 5.8, 4.6, 4.4, 2.6, 3.0, 3.5,
     4.6, 4.8, 4.5 , 2.1 ,2.4, 3, 6.5, 6.1, 6.3, 2.7, 3.2, 2.6, 4.6, 5.1, 4.7 , 2, 2.3, 1.9,
     4.3, 4.2, 4, 3.8, 3.4, 3.8, 5, 5.2, 5.3, 3.8, 3.8, 4.2, 5.5, 4.4, 4.1, 3.7, 3.5, 3.8, 3.4,
     3.7, 3.6, 3, 3, 3.1, 5.4 , 5.3 , 4.8 , 3.3 , 3.6 , 4.5 , 4.3 , 4.5 , 4.2 , 2.9, 3.1, 3.2, 5.2,
     4.7, 4.3, 3.4, 3.4, 3.7, 4, 4.6, 4.4, 2.6, 2.1, 2.8]
ma = max(a)
mi = min(a)

for i in a:
    temp = (i - mi)*((4 - 1)/(ma - mi)) + 1
    print(temp)