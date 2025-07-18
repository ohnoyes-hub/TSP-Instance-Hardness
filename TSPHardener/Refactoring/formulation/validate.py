from dataclasses import dataclass
from typing import List, Union
import ast
import csv
from config.experiment_config import ExperimentConfig
from config.validation_config import ValidationConfig
import logging

logger = logging.getLogger(__name__)

def validation_to_experiment_config(val_config: ValidationConfig) -> ExperimentConfig:
    """Convert validated CSV config to runtime experiment config."""
    return ExperimentConfig(
        citysize=val_config.size,
        generation_type=val_config.tsp_type,
        distribution=val_config.distribution,
        mutation_type=val_config.mutation_strategy,
        control=val_config.ranges[0]  # Or iterate through ranges as needed
    )

def load_configs(csv_path: str) -> List[ExperimentConfig]:
    configs = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, 1):
            try:
                # Create ValidationConfig (triggers __post_init__ checks)
                val_config = ValidationConfig(
                    size=int(row['size']),
                    ranges=ast.literal_eval(row['range'].strip('"')),
                    mutations=int(row['mutations']),
                    continuation=row['continuation'],
                    tsp_type=row['tsp type'].lower(),
                    distribution=row['distribution'].lower(),
                    mutation_strategy=row['mutation strategy'].lower()
                )
                
                # convert to ExperimentConfig
                exp_config = validation_to_experiment_config(val_config)
                configs.append(exp_config)
                
            except (ValueError, SyntaxError, KeyError) as e:
                logger.error(f"Row {row_idx} invalid: {str(e)}")
                continue
    return configs