import os

def get_result_path(citysize, rang, distribution, tsp_type, mutation_strategy, is_final=False):
    """Generate standardized file paths"""
    folder = "Phase-Trans-Results" if is_final else "Phase-Trans-Continuation"
    return os.path.join(
        folder,
        f"{distribution}_{tsp_type}",
        f"city{citysize}_range{rang}_{mutation_strategy}.json"
    )