# f = open("../../Corpus/e960401.htm", encoding='latin-1')
# print(f)

# text = f.read()
# print(text)


s = input('Enter a number: ')
num = int(s)

fac = list(range(1, num+1))
print('Before: ', fac)

fac = [ i for i in range(1, num+1)]
facAfter = []

for f in fac:
    if num%f != 0:
        facAfter.append(f)

print('After: ', facAfter)
