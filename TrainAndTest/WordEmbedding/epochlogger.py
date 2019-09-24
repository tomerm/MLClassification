from gensim.models.callbacks import CallbackAny2Vec

class EpochLogger(CallbackAny2Vec):
    def __init__(self, epochs):
        self.epoch = 1
        self.epochs = epochs

    def on_epoch_begin(self, model):
        print("Epoch %d from %d" % (self.epoch, self.epochs), end='\r')

    def on_epoch_end(self, model):
        self.epoch += 1
