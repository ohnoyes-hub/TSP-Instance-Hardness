import os
import glob
import json
from collections import defaultdict
from util.load_experiment import load_json, fill_missing_config  # Import your existing loader

def check_iteration_count(threshold=100):
    """
    Check if each configuration file has at least the specified number 
    of values in its 'all_iterations' array.
    """
    base_dirs = [
        "../data/Phase-Trans-Merged",
        # "../data/Phase-Trans-Continuation"
    ]
    
    # Track configuration completeness
    incomplete_files = []
    valid_files = 0
    invalid_files = 0
    
    print(f"Checking for at least {threshold} iterations per configuration...\n")
    
    for base_dir in base_dirs:
        pattern = os.path.join(base_dir, '**', '*.json')
        json_files = glob.glob(pattern, recursive=True)
        
        for file_path in json_files:
            data, errors, _ = load_json(file_path)
            
            if data is None or errors:
                invalid_files += 1
                print(f"🟥 Invalid: {file_path} (skipped due to errors)")
                continue
                
            try:
                iterations = data['results']['all_iterations']
                count = len(iterations)
                
                if count < threshold:
                    config = data['configuration']
                    incomplete_files.append({
                        'path': file_path,
                        'count': count,
                        'config': config
                    })
                    print(f"🟨 Incomplete: {file_path} ({count} iterations)")
                else:
                    valid_files += 1
                    # # Optional: Uncomment to see valid files
                    # print(f"🟩 Valid: {file_path} ({count} iterations)")
            except KeyError:
                invalid_files += 1
                print(f"🟥 Missing data: {file_path} (no 'all_iterations')")
    
    # Print summary
    print("\n" + "="*50)
    print(f"Total files checked: {len(json_files)}")
    print(f"Valid configurations: {valid_files} (≥{threshold} iterations)")
    print(f"Incomplete configurations: {len(incomplete_files)} (<{threshold} iterations)")
    print(f"Invalid/unloadable files: {invalid_files}")
    print("="*50)
    
    # Print incomplete files details
    if incomplete_files:
        print("\nIncomplete configurations:")
        for file_info in incomplete_files:
            config = file_info['config']
            print(f"\n- File: {file_info['path']}")
            print(f"  Iterations: {file_info['count']}")
            print("  Configuration:")
            print(f"    Distribution: {config.get('distribution')}")
            print(f"    Generation: {config.get('generation_type')}")
            print(f"    City size: {config.get('city_size')}")
            print(f"    Range: {config.get('range')}")
            print(f"    Mutation: {config.get('mutation_type')}")

base_dirs = [
    "../data/Last",
    "../data/Last-Cont",
    "../data/Phase-Trans_Continuation",
    "../data/Phase-Continuation-Redux",
    "../data/Phase_Continuation",
    "../data/Phase-Cont",
    "../data/Phase-Res",
    "../data/Phase-Continuation",
    "../data/Phase-Results",
    "../data/Rerun-Trans-Continuation",
    "../data/Rerun-Trans-Results",
    "../data/Phase-Trans-Results",
    "../data/Phase-Trans-Continuation"
]

OUTPUT_DIR = "../data/Phase-Trans-Merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def group_key(cfg):
    # Keep all distinguishing factors except mutation_type
    return (
        cfg.get('distribution'),
        cfg.get('generation_type'),   # <- this is your ETSP/ATSP distinction
        cfg.get('city_size'),
        cfg.get('range'),
    )

json_files = []
for base_dir in base_dirs:
    pattern = os.path.join(base_dir, '**', '*.json')
    json_files.extend(glob.glob(pattern, recursive=True))

groups = defaultdict(list)
for file_path in json_files:
    data, errors, _ = load_json(file_path)
    if data is None or errors:
        continue
    fill_missing_config(data, file_path)
    cfg = data['configuration']
    groups[group_key(cfg)].append((file_path, data))

print(f"Found {len(groups)} unique configs (ignoring mutation_type)")

for group_id, files in groups.items():
    all_iterations = []
    base_data = None

    for file_path, data in files:
        iters = data.get('results', {}).get('all_iterations', [])
        all_iterations.extend(iters)
        if base_data is None:
            base_data = data  # Use the first file as a template

    all_iterations = all_iterations[:100]

    # Build merged config (preserve all info except mutation_type)
    merged_config = dict(base_data['configuration'])
    merged_config['mutation_type'] = 'random_sampling'  # <-- As requested!

    # Build merged data object
    merged_data = dict(base_data)
    merged_data['configuration'] = merged_config
    merged_data['results']['all_iterations'] = all_iterations
    merged_data['results']['merged_from'] = [fp for fp, _ in files]

    dist, gen_type, city, rng = group_id
    subfolder = f"{dist}_{gen_type}"
    filename = f"city{city}_range{rng}_random_sampling.json"
    full_outdir = os.path.join(OUTPUT_DIR, subfolder)

    os.makedirs(full_outdir, exist_ok=True)
    outpath = os.path.join(full_outdir, filename)

    with open(outpath, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Saved merged file: {outpath} (n={len(all_iterations)})")

print("All groups processed.")

if __name__ == "__main__":
    check_iteration_count(threshold=100)

