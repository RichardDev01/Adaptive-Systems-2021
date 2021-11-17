"""Class file for making policies."""
import abc
from typing import List


class BasePolicy(metaclass=abc.ABCMeta):
    """Most generic agent class with fill in template."""

    @abc.abstractmethod
    def decide_action(self, observation: List):
        """Take an action based on the given observation."""
        raise NotImplementedError
