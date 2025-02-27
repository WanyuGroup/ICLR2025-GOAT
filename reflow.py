# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import argparse
from qm9 import dataset
from models.get_models import get_goat
from utils.utilis_func import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import os
import pickle
from mol_data.get_datasets import get_data_loader
from configs.datasets_config import get_dataset_info
from tqdm import tqdm
from os.path import join
from utils.sampling import reflow_sample
from qm9.analyze import filer_molecules, analyze_stability_for_molecules
from utils import utilis_func as uf
from utils.parse_args import reflow_parse_args

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def generate_reflow_pair(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecule_lis = {'one_hot': [], 'positions': [], 'atom_mask': [], 'noise': [], 'charges': []}
    start_time = time.time()
    for i in tqdm(range(int(n_samples/batch_size))):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask, z = reflow_sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

        molecules = {'one_hot': one_hot.detach().cpu(),
                     'positions': x.detach().cpu(),
                     'atom_mask': node_mask.detach().cpu(),
                     'charges': charges.squeeze(-1).detach().cpu(),
                     'noise': z.detach().cpu()}
        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        molecules = filer_molecules(molecules, dataset_info)
        print('\t %d/%d Molecules generated at %.2f secs/sample (Success: %d)' % (
            current_num_samples, n_samples, secs_per_sample, molecules['noise'].size(0)))
        for key, value in molecule_lis.items():
            molecule_lis[key].append(molecules[key])
    molecule_lis = {key: torch.cat(molecule_lis[key], dim=0) for key in molecule_lis}
    with open(join(eval_args.reflow_path, f'{args.dataset}_reflow_data_{eval_args.reflow_index}.pickle'),
              "wb") as file:
        pickle.dump(molecule_lis, file)
    return molecule_lis

def check_files_exist(file_paths):
    while True:
        missing_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if not missing_files:
            break

        print("Waiting for the following files to appear:")
        for file_path in missing_files:
            print(file_path)

        time.sleep(1)

def process_charges(one_hot, node_size):
    print(one_hot[0])
    charges = torch.argmax(one_hot, dim=2)
    for i, size in enumerate(node_size):
        for j in range(int(size[0])):
            if charges[i, j] == 0:
                charges[i, j] = 1
            else:
                charges[i, j] = charges[i, j] + 5
    print(charges[0])
    return charges

def merge_molecules(args, eval_args):
    merge_range = eval_args.reflow_index + 1
    file_list = [join(eval_args.reflow_path, f'{args.dataset}_reflow_data_{i}.pickle') for i in range(merge_range)]
    check_files_exist(file_list)
    for i, file in enumerate(file_list):
        with open(file, 'rb') as file:
            data_pair = pickle.load(file)
            if i == 0:
                data_pair_full = data_pair
            else:
                for key in data_pair_full:
                    data_pair_full[key] = torch.cat([data_pair_full[key], data_pair[key]], dim=0)
    data_pair_full['charges'] = process_charges(data_pair_full['one_hot'], data_pair_full['atom_mask'].sum(1))
    reflow_data_size = data_pair_full['noise'].size(0)
    split_text = ['train', 'test', 'valid']
    split_size = [int(reflow_data_size * 0.76), int(reflow_data_size * 0.88), reflow_data_size]
    begin = 0
    for text, size in zip(split_text, split_size):
        data_pair_part = {key: data_pair_full[key][begin:size] for key in data_pair_full}
        begin = size
        with open(f'reflow/qm9_reflow_{eval_args.merge}_{text}.pickle', 'wb') as file:
            pickle.dump(data_pair_part, file)

def main():
    parser = reflow_parse_args()
    eval_args, unparsed_args = parser.parse_known_args()
    
    ckpt_path = eval_args.model_path + 'checkpoints/smg.pt'
    assert os.path.isfile(ckpt_path), f'Could not find SMG checkpoint at {ckpt_path}'
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    args = checkpoint["args"]
    model_check_point = checkpoint["model_ema"] if args.ema_decay > 0 else checkpoint["model"]

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    uf.create_folders(args)

    try:
        if eval_args.dataset != 'None':
            print('Load target dataset from ', eval_args.dataset)
            args.is_da_mg = True
            args.dataset = eval_args.dataset
    except:
        print('Load dataset from ', args.dataset)

    args.node_classifier_model_ckpt = eval_args.node_classifier_model_ckpt
    args.without_cat_loss = True

    # Retrieve QM9 dataloaders
    args.batch_size = eval_args.batch_size_gen
    dataloaders, sampler = get_data_loader(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    # Load model
    generative_model, nodes_dist, prop_dist = get_goat(args, device, dataset_info, dataloaders['train'])
    generative_model.load_state_dict(model_check_point)
    generative_model.to(device)

    # Analyze stability, validity, uniqueness and novelty
    if not os.path.exists(eval_args.reflow_path):
        os.makedirs(eval_args.reflow_path)
    molecules = generate_reflow_pair(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
    if eval_args.merge > 0:
        merge_molecules(args, eval_args)

if __name__ == "__main__":
    main()
