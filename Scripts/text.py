import pickle

def loadText(File):
    f = open(File, "r")
    text = f.read()
    return text
    f.close()

Text = loadText('../Flickr8k/Flickr8k.token.txt')
descriptions = {}

for line in Text.split('\n'):
    if len(line) < 2:
        continue
    words = line.split()
    imageid = words[0].split('.')[0]
    if imageid not in descriptions:
        descriptions[imageid] = []
    description = words[1:]
    for i in range(len(description)):
        description[i] = description[i].lower()
    descriptions[imageid].append(description)


for ID, descs in descriptions.items():
    for i in range(len(descs)):
        d = descs[i]
        d = [word for word in d if word.isalpha()]
        descs[i] = d


# print(descriptions)
pickle.dump(descriptions, open('descriptions.pkl', 'wb'))
