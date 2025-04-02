from dataclasses import dataclass

@dataclass
class TSPConfig:
    distribution: str
    control: float
    tsp_type: str

@dataclass
class EuclideanConfig(TSPConfig):
    city_size: int
    lognormal_mean: float = 10
