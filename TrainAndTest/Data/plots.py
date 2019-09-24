import numpy
import matplotlib.pyplot as plt


def showDocsByLength(dynamic_store):
    train_docs = dynamic_store["train_docs"]
    test_docs = dynamic_store["test_docs"]
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(10 ,6))
    dictLens = dict()
    dictLens1 = dict()
    for doc in train_docs:
        lend = "%5d " %(len(doc.words))
        if not lend in dictLens:
            dictLens[lend] = 1
        else:
            dictLens[lend] += 1
    lens = sorted(list(dictLens.items()))
    lvars = [int(x[0]) for x in lens]
    locc = [x[1] for x in lens]
    plot1.set_title ("Documents by tokens in training set")
    plot1.set_ylabel("Documents")
    plot1.set_xlabel("Tokens")
    plot1.plot(lvars, locc, "b.-")
    for doc in test_docs:
        lend = "%5d " %(len(doc.words))
        if not lend in dictLens1:
            dictLens1[lend] = 1
        else:
            dictLens1[lend] += 1
    lens1 = sorted(list(dictLens1.items()))
    lvars1 = [int(x[0]) for x in lens1]
    locc1 = [x[1] for x in lens1]
    plot2.set_title ("Documents by tokens in testing set")
    # plot2.set_ylabel("Documents")
    plot2.set_xlabel("Tokens")
    plot2.yaxis.tick_right()
    plot2.plot(lvars1, locc1, "b.-")
    plt.show()

def showDocsByLabs(dynamic_store):
    train_docs = dynamic_store["train_docs"]
    test_docs = dynamic_store["test_docs"]
    categories = dynamic_store["predefined_categories"]
    fig, (plot1, plot2) = plt.subplots(1, 2, figsize=(10 ,6))
    dictLabs = dict()
    dictLabs1 = dict()
    for doc in train_docs:
        lab = "%5d " % doc.qLabs[0]
        if not lab in dictLabs:
            dictLabs[lab] = 1
        else:
            dictLabs[lab] += 1
    labs = sorted(list(dictLabs.items()))
    lvars1 = [int(x[0]) for x in labs]
    locc1 = [x[1] for x in labs]
    plot1.set_title ("Documents by labels in training set")
    plot1.set_ylabel("Documents")
    plot1.set_xlabel("Labels")
    plot1.set_xticks(numpy.arange(0, len(categories), step=1))
    plot1.plot(lvars1, locc1, "bo-")
    for doc in test_docs:
        lab = "%5d " %(doc.qLabs[0])
        if not lab in dictLabs1:
            dictLabs1[lab] = 1
        else:
            dictLabs1[lab] += 1
    labs1 = sorted(list(dictLabs1.items()))
    lvars2 = [int(x[0]) for x in labs1]
    locc2 = [x[1] for x in labs1]
    plot2.set_title ("Documents by labels in testing set")
    # plot2.set_ylabel("Documents")
    plot2.set_xlabel("Labels")
    plot2.set_xticks(numpy.arange(0, len(categories), step=1))
    plot2.yaxis.tick_right()
    plot2.plot(lvars2, locc2, "bo-")
    plt.show()
