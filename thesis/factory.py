from thesis.env_runner import CooperativeNavigationRunner
from thesis.model import CriticMlp, CriticMlpIndex, CriticGcn


def critic_factory(critic_type, **critic_kwarg):
    factories = {
        'mlp': CriticMlp(**critic_kwarg),
        'mlp_index': CriticMlpIndex(**critic_kwarg),
        'gcn': CriticGcn(**critic_kwarg)
    }
    return factories[critic_type]


def env_factory(env_name, **env_kwarg):
    factories = {
        'cooperative_navigation': CooperativeNavigationRunner(**env_kwarg)
    }
    return factories[env_name]
