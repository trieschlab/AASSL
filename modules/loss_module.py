from abc import ABC, abstractmethod


class LossModule(ABC):

    @abstractmethod
    def apply(self, batch_data, net_target=None, **kwargs):
        pass

    @classmethod
    def get_args(cls, parser):
        return parser

    def eval(self, network, *args):
        return {}