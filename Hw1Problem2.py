def find_even_nums (n):
    list1 = []
    for i in range(n):
        if i % 2 == 0:
            list1.append(i)
    return list1

n = 0
print(find_even_nums(n))