f = open("dims.txt", 'r')

text = ""
lines = f.readlines()
for line in lines:
    for i in range(len(line)):
        if line[i] == ",":
            text += ", "
        else:
            text += line[i]
f.close()
new_f = open("newDims.txt", 'w')
new_f.write(text)
new_f.close()
