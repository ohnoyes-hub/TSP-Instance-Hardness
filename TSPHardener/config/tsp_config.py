from dataclasses import dataclass

@dataclass
class TSPConfig:
    distribution: str
    control: float
    tsp_type: str
    lognormal_mean: float = 10