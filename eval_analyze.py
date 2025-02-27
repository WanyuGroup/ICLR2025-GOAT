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
import random
from mol_data.get_datasets import get_data_loader
from configs.datasets_config import get_dataset_info
from tqdm import tqdm
from os.path import join
from utils.sampling import sample
from qm9.analyze import analyze_stability_for_molecules
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
from utils import utilis_func as uf
import train_epoch
from utils.parse_args import eval_parse_args

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    for i in tqdm(range(int(n_samples/batch_size))):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)

    return stability_dict, rdkit_metrics


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test', num_passes=1):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            loader_tqdm = tqdm(loader, ncols=80)
            for i, data in enumerate(loader_tqdm):
                # Get mol_data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                if len(args.conditioning) > 0:
                    context = prepare_context(args.conditioning, data).to(device, dtype)
                    assert_correctly_masked(context, node_mask)
                else:
                    context = None
                try:
                    scaffold_mask = data['scaffold_mask'].to(device, dtype)
                except:
                    bs, ns, _ = node_mask.size()
                    scaffold_mask = torch.zeros([bs, ns]).to(device, dtype)
                    masked_length = data['num_atoms']
                    for i in range(bs):
                        ones_indices = random.sample(range(masked_length[i]),
                                                     int((1 - args.mask_ratio) * masked_length[i]))
                        scaffold_mask[i, ones_indices] = 1
                try:
                    noise = data['noise'].to(device, dtype)
                except:
                    noise = None
                # transform batch through flow
                nll, reg_term, mean_abs_z = train_epoch.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask, edge_mask, context, noise)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    test_des = (f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                                 f"NLL: {nll_epoch/n_samples:.2f}")
                    loader_tqdm.set_description(test_des)
    return nll_epoch/n_samples


def eval_data(args, eval_args, dataset_info, dataloaders, generative_model, nodes_dist, prop_dist, device, dtype):
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)

    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, dataloaders, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
    print(stability_dict)

    if rdkit_metrics is not None:
        # rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
        with open(join(eval_args.model_path, f'eval_{eval_args.target_domain}/scaffolds.txt'), "w") as file:
            # Dump the data into the file
            for scaffold in rdkit_metrics[3]:
                print(scaffold, file=file)
        with open(join(eval_args.model_path, f'eval_{eval_args.target_domain}/rings.txt'), "w") as file:
            # Dump the data into the file
            for scaffold in rdkit_metrics[4]:
                print(scaffold, file=file)
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")

    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'valid'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5

    # Evaluate negative log-likelihood for the validation and test partitions
    if args.dataset == 'geom':
        val_nll = 0
        test_nll = 0
    else:
        val_nll = test(args, generative_model, nodes_dist, device, dtype,
                       dataloaders[val_name],
                       partition='Val')
        print(f'Final val nll {val_nll}')
        test_nll = test(args, generative_model, nodes_dist, device, dtype,
                        dataloaders['test'],
                        partition='Test', num_passes=num_passes)
        print(f'Final test nll {test_nll}')

    print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    with open(join(eval_args.model_path, f'eval_log_{eval_args.target_domain}.txt'), 'w') as f:
        print(f'use {eval_args.target_domain} data as condition, dataset = {args.dataset}', file=f)
        print(f'Overview: val nll {val_nll} test nll {test_nll}',
              stability_dict,
              file=f)
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]),
              file=f)

def main():
    parser = eval_parse_args()
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
    stability_dict, rdkit_metrics = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz)
    print(stability_dict)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")

    # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
    if args.dataset == 'geom':
        val_name = 'val'
        num_passes = 1
    else:
        val_name = 'valid'
        num_passes = 5

    # Evaluate negative log-likelihood for the validation and test partitions
    val_nll = test(args, generative_model, nodes_dist, device, dtype,
                   dataloaders[val_name],
                   partition='Val')
    print(f'Final val nll {val_nll}')
    test_nll = test(args, generative_model, nodes_dist, device, dtype,
                    dataloaders['test'],
                    partition='Test', num_passes=num_passes)
    print(f'Final test nll {test_nll}')

    print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict)
    with open(join(eval_args.model_path, 'eval_log.txt'), 'w') as f:
        print(f'Overview: val nll {val_nll} test nll {test_nll}',
              stability_dict,
              file=f)
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]), file=f)



if __name__ == "__main__":
    main()
