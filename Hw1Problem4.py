
import math
import collections
from collections import Counter
def factorize(n):
    i = 2
    factors = []
    freq = []
    settolist = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    num_count = Counter(factors)
    for i in num_count:
        settolist.append(i)
        
    
    for n, count in num_count.most_common(100):
        
        freq.append(count)
        
    combo = [pair for pair in zip(settolist,freq)]
    return combo
    
    
n = 10
print(factorize(n))