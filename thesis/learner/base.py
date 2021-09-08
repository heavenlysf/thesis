from abc import ABC, abstractmethod


class ILearner(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError
