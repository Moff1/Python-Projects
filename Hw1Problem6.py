def generate_palindromes(limit):
    sol = []
    lessthan10 = []
    if limit < 151:
        for i in range(0, limit+1):
            if i == int("".join(list(str(i))[::-1])):
                sol.append(i)
        return sol[-15::]
        
            
    for i in range(10, limit+1):
        if i == int("".join(list(str(i))[::-1])):
            sol.append(i)
    return sol[-15::]

print(generate_palindromes(30))