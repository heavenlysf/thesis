from abc import ABC, abstractmethod


class IRunner(ABC):
    @abstractmethod
    def rollout(self):
        raise NotImplementedError
