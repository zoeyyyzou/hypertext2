from abc import abstractmethod, ABCMeta


class ModelBase(metaclass=ABCMeta):

    @abstractmethod
    def train(self, train_file: str):
        pass

    @abstractmethod
    def test(self, test_file: str):
        pass
