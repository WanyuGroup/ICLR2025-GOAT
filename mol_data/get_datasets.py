import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# QM9
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
from qm9.dataset import filter_atoms

# drug
from mol_data import build_geom_dataset
from utils.datasets_config import geom_with_h

def get_data_loader(args, dist=None, rank=None):
    if args.dataset == 'geom':
        data_file = '../../Data/geom/geom_drugs_30.npy'

        if args.remove_h:
            raise NotImplementedError()
        else:
            dataset_info = geom_with_h

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if args.cuda else "cpu")
        dtype = torch.float32

        split_data = build_geom_dataset.load_split_data(data_file, val_proportion=0.1, test_proportion=0.1,
                                                        filter_size=args.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info, args.include_charges, device, args.sequential)
        datasets = {}
        for key, data_list in zip(['train', 'valid', 'test'], split_data):
            datasets[key] = build_geom_dataset.GeomDrugsDataset(data_list, transform=transform)
        if dist is not None:
            samplers = {split: DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                                  rank=rank, shuffle=True, seed=args.global_seed)
                        for split, dataset in datasets.items()}

            dataloaders = {split: build_geom_dataset.GeomDrugsDataLoader(
                sequential=args.sequential, dataset=dataset, batch_size=int(args.global_batch_size // dist.get_world_size()),
                shuffle=(split == 'train') and not args.sequential, num_workers=args.num_workers, sampler=samplers[split], pin_memory=False)
                           for split, dataset in datasets.items()}
        else:
            samplers = None
            dataloaders = {split: build_geom_dataset.GeomDrugsDataLoader(sequential=args.sequential, dataset=dataset,
                                             batch_size=args.batch_size,
                                             shuffle=(split == 'train') and not args.sequential,
                                             num_workers=args.num_workers, sampler=samplers, pin_memory=False)
                           for split, dataset in datasets.items()}

        del split_data
#     if args.dataset == 'geom':
#         data_dir = '../../Data/geom/processed_geom/'
#         dataloaders = {}
#         if args.remove_h:
#             raise NotImplementedError()
#         else:
#             dataset_info = geom_with_h

#         args.cuda = not args.no_cuda and torch.cuda.is_available()
#         device = torch.device("cuda" if args.cuda else "cpu")
#         train_index = int(args.geom_size - args.geom_size * 0.25)
#         test_index = int(args.geom_size)

#         train_dataset = []
#         for save_index in tqdm(range(train_index)):
#             with open(data_dir + f'drug_data_{save_index}.pickle', 'rb') as file:
#                 train_dataset += pickle.load(file)

#         test_dataset = []
#         for save_index in tqdm(range(train_index, test_index)):
#             with open(data_dir + f'drug_data_{save_index}.pickle', 'rb') as file:
#                 test_dataset += pickle.load(file)

#         # save_index = 69
#         # with open(data_dir + f'drug_data_{save_index}.pickle', 'rb') as file:
#         #     train_dataset = pickle.load(file)
#         # with open(data_dir + f'drug_data_{save_index}.pickle', 'rb') as file:
#         #     test_dataset = pickle.load(file)

#         datasets = {'train': train_dataset,
#                     'test': test_dataset,
#                     'valid': test_dataset}
#         if dist is not None:
#             print(args.sequential)
#             samplers = {split: DistributedSampler(dataset, num_replicas=dist.get_world_size(),
#                                                   rank=rank, shuffle=True, seed=args.global_seed)
#                         for split, dataset in datasets.items()}

#             dataloaders = {split: build_geom_dataset.GeomDrugsDataLoader(
#                 sequential=args.sequential, dataset=dataset,
#                 batch_size=int(args.global_batch_size // dist.get_world_size()),
#                 shuffle=(split == 'train') and not args.sequential, num_workers=args.num_workers,
#                 sampler=samplers[split], pin_memory=False)
#                 for split, dataset in datasets.items()}

#         else:
#             samplers = None
#             dataloaders = {split: build_geom_dataset.GeomDrugsDataLoader(sequential=args.sequential, dataset=dataset,
#                                                                          batch_size=args.batch_size,
#                                                                          shuffle=(split == 'train') and not args.sequential,
#                                                                          num_workers=args.num_workers, sampler=samplers,
#                                                                          pin_memory=False)
#                            for split, dataset in datasets.items()}
#         del datasets
    elif 'qm9' in args.dataset:
        # Retrieve QM9 dataloaders
        batch_size = args.batch_size
        num_workers = args.num_workers
        filter_n_atoms = args.filter_n_atoms
        # Initialize dataloader
        qm_args = init_argparse('qm9')
        if len(args.conditioning) > 0:
            property = args.conditioning[0]
        else:
            property = None
        qm_args, datasets, num_species, charge_scale = initialize_datasets(qm_args, args.datadir, args.dataset,
                                                                        subtract_thermo=qm_args.subtract_thermo,
                                                                        force_download=qm_args.force_download,
                                                                        remove_h=args.remove_h, property=property)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114,
                     'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=args.include_charges)
        if dist is not None:
            samplers = {split: DistributedSampler(dataset, num_replicas=dist.get_world_size(),
                                                  rank=rank, shuffle=True, seed=args.global_seed)
                        for split, dataset in datasets.items()}

            dataloaders = {split: DataLoader(dataset,
                                             batch_size=int(args.global_batch_size // dist.get_world_size()),
                                             num_workers=args.num_workers,
                                             sampler=samplers[split],
                                             pin_memory=True,
                                             collate_fn=preprocess.collate_fn)
                           for split, dataset in datasets.items()}
        else:
            dataloaders = {split: DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=qm_args.shuffle if (split == 'train') else False,
                                             num_workers=num_workers,
                                             collate_fn=preprocess.collate_fn)
                           for split, dataset in datasets.items()}
            samplers = None
    else:
        raise NotImplementedError(args.dataset)
    return dataloaders, samplers
