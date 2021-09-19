from abc import ABC, abstractmethod


class IComa(ABC):
    @abstractmethod
    def run(self, iterations):
        raise NotImplementedError

    def demo(self, n_env):
        raise NotImplementedError

    def get_log(self):
        raise NotImplementedError

    def load_network(self, actor_state, critic_state):
        raise NotImplementedError

    def get_network(self):
        raise NotImplementedError
