from datasets import concatenate_datasets, load_from_disk

parser.add_argument('--output_path', type=str)
parser.add_argument('--n_gpus', type=int)

def collect_and_save(args):
    paths = [args.output_path + f'_{g}' for g in range(n_gpus)]
    datasets = [load_from_disk(p) for p in paths]
    all_data = concatenate_datasets(datasets)
    all_data.save_to_disk(args.output_path)

if __name__ == "__main__":
    args = parser.parse_args()
    collect_and_save(args)