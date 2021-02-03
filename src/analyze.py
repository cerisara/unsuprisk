txts = []
with open("textintest","r") as f:
    for l in f:
        s=l.strip()[10:]
        txts.append(s)

epochs = []
trepochs = []
with open("tmplog.6","r") as f:
    for l in f:
        if l.startswith("DETCVDD "):
            id2all = {}
            idordre = []
            st=l.strip().split('(')
            for s in st:
                x = s.split(',')
                if len(x)>=4:
                    idi = int(x[0])
                    idordre.append(idi)
                    lab = int(float(x[1]))
                    score = float(x[2])
                    i=x[3].find(')')
                    h=x[3][0:i]
                    id2all[idi] = [lab,score,h]
            epochs.append(id2all)
        elif l.startswith("DETRISK ") and False:
            st=l[10:].strip().split(',')
            assert len(st)==len(idordre)
            i=st[-1].find(']')
            st[-1]=st[-1][0:i]
            for i in range(len(st)):
                id2all[idordre[i]][1]=float(st[i])
        elif l.startswith("DETCVDDTRAIN"):
            trid2all = {}
            idordre = []
            st=l.strip().split('(')
            for s in st:
                x = s.split(',')
                if len(x)>=4:
                    idi = int(x[0])
                    idordre.append(idi)
                    lab = int(float(x[1]))
                    score = float(x[2])
                    i=x[3].find(')')
                    h=x[3][0:i]
                    trid2all[idi] = [lab,score,h]
        elif l.startswith("DETRISKTRAIN"):
            st=l[15:].strip().split(',')
            assert len(st)==len(idordre)
            i=st[-1].find(']')
            st[-1]=st[-1][0:i]
            for i in range(len(st)):
                trid2all[idordre[i]][1]=float(st[i])
            trepochs.append(trid2all)

print("nepochs ", len(epochs))
print("ntestsamples ", len(idordre))
print("maxidx ",max(idordre))

# epochs[nepoch][obsid] = (lab, score, head)


def printScoresTest():
    sc = [x[1] for x in epochs[-1].values() if x[0]==1]
    for s in sc: print(s)
def printScoresTrain():
    sc = [x[1] for x in trepochs[-2].values()]
    for s in sc: print(s)

def samplesExtreme():
    sortedids = sorted(epochs[-1].items(), key=lambda item: item[1])
    # sortedids = ( (id, (lab,score,head)) ...)

    print("extreme low...")
    for i in range(10):
        print(sortedids[i][1][0],txts[sortedids[i][0]])

    print("extreme high...")
    for i in range(1,11):
        print(sortedids[-i][1][0],txts[sortedids[-i][0]])

    n1 = sum([sortedids[i][1][0] for i in range(len(sortedids))])
    n0 = sum([1-sortedids[i][1][0] for i in range(len(sortedids))])
    print("n0 %d n1 %d" % (n0,n1))

def samplesMid():
    sortedids = sorted(epochs[-1].items(), key=lambda item: item[1])
    # sortedids = ( (id, (lab,score,head)) ...)

    x = [sortedids[i][0] for i in range(len(sortedids)) if sortedids[i][1][1] > 0.12]

    for i in range(5):
        print(txts[x[i]])


# printScoresTrain()
# printScoresTest()
# samplesExtreme()
samplesMid()


