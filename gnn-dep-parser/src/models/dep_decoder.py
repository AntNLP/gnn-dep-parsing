from abc import ABCMeta, abstractmethod
from typing import List
import dynet as dy


class DependencyDecoder(metaclass=ABCMeta):
    """docstring for DependencyDecoder"""

    @abstractmethod
    def __call__(self, inputs: List[dy.Expression]) -> List[dy.Expression]:
        pass

