from abc import abstractmethod
class Model:
    @abstractmethod
    def train_with_weights(self, train_X, train_Y, weights):
        pass

    @abstractmethod
    def evaluate(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass
