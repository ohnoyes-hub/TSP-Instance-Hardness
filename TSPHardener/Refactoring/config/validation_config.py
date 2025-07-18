from dataclasses import dataclass
from typing import List, Union

@dataclass
class ValidationConfig:
    size: int
    ranges: List[Union[float, int]]
    mutations: int
    continuation: str
    tsp_type: str
    distribution: str
    mutation_strategy: str

    def __post_init__(self):
        # Validate numeric values
        if self.size not in {20, 30}:
            raise ValueError(f"Invalid size: {self.size}. Only testing 20 or 30 for now")
            
        if self.mutations != 1000:
            raise ValueError(f"Unexpected mutation count: {self.mutations}")

        # Validate categorical values
        validators = {
            'tsp_type': ['euclidean', 'asymmetric'],
            'distribution': ['uniform', 'lognormal'],
            'mutation_strategy': ['scramble', 'wouter', 'swap', 'random_sampling']
        }
        
        for field, allowed in validators.items():
            value = getattr(self, field)
            if value.lower() not in allowed:
                raise ValueError(f"Invalid {field}: {value}. Allowed: {allowed}")

        # Validate range matches distribution
        if self.distribution == 'lognormal' and any(isinstance(x, int) for x in self.ranges):
            raise ValueError("Lognormal distribution requires float ranges")
            
        if self.distribution == 'uniform' and any(isinstance(x, float) for x in self.ranges):
            raise ValueError("Uniform distribution requires integer ranges")