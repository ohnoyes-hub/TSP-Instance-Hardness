from util.load_experiment import load_all_iteration, load_all_hard_instances

# grouped_data = load_all_iteration()
# counts = {k: len(v) for k, v in grouped_data.items()}
# print("Counts of iterations by group:")
# for group, count in counts.items():
#     print(f"{group}: {count}")

hard_instances = load_all_hard_instances()
counts = hard_instances.groupby('mutation_type')['iterations'].count().reset_index(name='hard_instance_count')
print("\nCounts of hard instances by mutation_type type:")
for _, row in counts.iterrows():
    print(f"{row['mutation_type']}: {row['hard_instance_count']} hard instances")