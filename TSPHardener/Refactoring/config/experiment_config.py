from dataclasses import dataclass
from typing import List, Union


@dataclass
class ExperimentConfig:
    """Defines runtime experiement executution""" 
    citysize: int
    generation_type: str
    distribution: str
    mutation_type: str
    control: List[Union[float, int]]