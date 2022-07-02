f = open("dims.txt", "r")

text = ""
for line in f:
    thing = line.strip()
    text += thing.replace(" (", "(") + "\n"

f.close()

f = open("dims.txt", "w")

f.write(text)

f.close()
