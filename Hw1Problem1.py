def happy_number (string):
    counter = 0
    for i in string:
        count = 0
        i = 0
        if string[i] == ':' and string[i+1] == ')':
            counter = counter + 1
            count = count + 1
        elif string[i] == ':' and string[i+1] == '(':
            counter = counter - 1
            count = count - 1
        elif string[i] == '(' and string[i+1] == ':':
            counter = counter + 1
            count = count + 1
        elif string[i] == ')' and string[i+1] == ':':
            counter = counter - 1
            count = count - 1
        elif string[i] == ':' and string[i+1] == ':':
            counter = counter + 0
            count = count + 0
        elif string[i] == ':' and string[i+1] == ':':
            counter = counter + 0
            count = count + 0
        elif string[i] == '(' and string[i+1] == '(':
            counter = counter + 0
            count = count + 0
    
    print(count)
    
    
string = ':)'
happy_number (string)