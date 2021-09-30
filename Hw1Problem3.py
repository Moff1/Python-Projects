def count_vowels (string):
    vowel = set("aeiouAEIOU")
    count = 0
    for i in string:
        
        if i in "aeiouAEIOU":
            count = count + 1
    
    return count
string = 'Prediction'
count_vowels(string)
print("The word " + string + " has", count_vowels(string), "vowels")