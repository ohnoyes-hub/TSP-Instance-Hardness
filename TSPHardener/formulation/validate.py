from dataclasses import dataclass
from typing import List, Union
import ast
import csv
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
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
            'mutation_strategy': ['scramble', 'wouter', 'swap']
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

def load_configs(csv_path: str) -> List[ExperimentConfig]:
    configs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            try:
                # Convert string representations to Python objects
                config = ExperimentConfig(
                    size=int(row['size']),
                    ranges=ast.literal_eval(row['range'].strip('"')),
                    mutations=int(row['mutations']),
                    continuation=row['continuation'],
                    tsp_type=row['tsp type'].lower(),
                    distribution=row['distribution'].lower(),
                    mutation_strategy=row['mutation strategy'].lower()
                )
                configs.append(config)
            except (ValueError, SyntaxError, KeyError) as e:
                logger.error(f"Error in row {row_idx}: {str(e)}")
                logger.error(f"Problematic row: {row}")
                continue
    return configs