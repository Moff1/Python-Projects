keys = ["Paris", 3, 4.5] 
values = ["France", "is odd", "is half of 9"]
swap = True


def swap_d (keys, values, swap):
    dicts = {}
    if len(keys) != len(values):
        dicts = {}
        print(dicts)    
    
    elif (len(keys) == len(values)) and (swap == False):
        for i in range(len(keys)):
        
            dicts[keys[i]] = values[i]
            
    elif (len(keys) == len(values)) and (swap == True):
        for i in range(len(keys)):
        
            dicts[values[i]] = keys[i]
    
    print(dicts)  
swap_d(keys,values,swap)