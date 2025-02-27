import argparse


def eval_sample_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--target_domain', type=str, default='train',
                        help='From where to condition')
    parser.add_argument('--dataset', type=str, default='None')
    parser.add_argument("--mask_ratio", default=0, type=float)
    return parser

def reflow_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--saved_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--reflow_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--reflow_index', type=int, default=0,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')

    parser.add_argument('--target_domain', type=str, default='train',
                        help='From where to condition')
    parser.add_argument('--dataset', type=str, default='None')
    parser.add_argument("--mask_ratio", default=0, type=float)
    parser.add_argument("--node_classifier_model_ckpt", type=str, default=None,
                        help="Optional path to a SMG chekcpoint")
    parser.add_argument('--merge', type=int, default=0,
                        help='Merge')
    return parser


def eval_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')

    parser.add_argument('--target_domain', type=str, default='train',
                        help='From where to condition')
    parser.add_argument('--dataset', type=str, default='None')
    parser.add_argument("--mask_ratio", default=0, type=float)
    parser.add_argument("--node_classifier_model_ckpt", type=str, default=None,
                        help="Optional path to a SMG chekcpoint")
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description='E3Diffusion')
    parser.add_argument('--exp_name', type=str, default='debug_10')
    #
    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--distill", type=bool, default=False)


    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='weight of KL term in ELBO')
    parser.add_argument('--model', type=str, default='egnn_dynamics',
                        help='our_dynamics | schnet | simple_dynamics | '
                             'kernel_dynamics | egnn_dynamics |gnn_dynamics')
    parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                        help='diffusion')

    # Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
    parser.add_argument('--diffusion_steps', type=int, default=500)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                        )
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        help='vlb, l2')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--brute_force', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--actnorm', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--break_train_epoch', type=eval, default=False,
                        help='True | False')
    parser.add_argument('--dp', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--condition_time', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--clip_grad', type=eval, default=True,
                        help='True | False')
    parser.add_argument('--trace', type=str, default='hutch',
                        help='hutch | exact')
    parser.add_argument('--analyze_during_train', type=bool, default=False,
                        help='True | False')
    # EGNN args -->
    parser.add_argument('--n_layers', type=int, default=6,
                        help='number of layers')
    parser.add_argument('--inv_sublayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--nf', type=int, default=128,
                        help='number of layers')
    parser.add_argument('--tanh', type=eval, default=True,
                        help='use tanh in the coord_mlp')
    parser.add_argument('--attention', type=eval, default=True,
                        help='use attention in the EGNN')
    parser.add_argument('--norm_constant', type=float, default=1,
                        help='diff/(|diff| + norm_constant)')
    parser.add_argument('--sin_embedding', type=eval, default=False,
                        help='whether using or not the sin embedding')
    # <-- EGNN args

    # Rectified flow
    parser.add_argument('--sampling_init_type', type=str, default='gaussian')
    parser.add_argument('--sampling_init_noise_scale', type=float, default=1)
    parser.add_argument('--sampling_use_ode_sampler', type=str, default='rk45')

    parser.add_argument('--trainable_ae', action='store_true',
                        help='Train first stage AutoEncoder model')
    parser.add_argument(
        "--discrete_path", type=str, default="OT_path", help="OT_path, HB_path, VP_path"
    )
    parser.add_argument("--cat_loss_step", type=float, default=-1)

    parser.add_argument("--cat_loss", type=str, default="l2", help='"l2" or "cse"')

    parser.add_argument("--on_hold_batch", type=int, default=-1)

    parser.add_argument("--sampling_method", type=str, default="vanilla")
    parser.add_argument("--weighted_methods", type=str, default="jump")
    parser.add_argument("--ode_method", type=str, default="dopri5")

    parser.add_argument(
        "--without_cat_loss", action="store_true", help="train without categorical loss"
    )

    parser.add_argument('--latent_nf', type=int, default=2,
                        help='number of latent features')

    parser.add_argument("--node_classifier_model_ckpt", type=str)

    parser.add_argument(
        "--angle_penalty", action="store_true", help="train with angle penalty"
    )
    parser.add_argument("--extend_feature_dim", type=int, default=0)
    parser.add_argument("--minimize_type_entropy", action="store_true", default=False)
    parser.add_argument("--minimize_entropy_grad_coeff", type=float, default=0.0)

    # parser.add_argument("--num_e", default=4000, type=int)

    parser.add_argument('--ode_regularization', type=float, default=1e-3)
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset) | qm9_few_shot')
    parser.add_argument('--datadir', type=str, default='qm9/temp',
                        help='qm9 directory')
    parser.add_argument('--filter_n_atoms', type=int, default=None,
                        help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    parser.add_argument('--dequantization', type=str, default='argmax_variational',
                        help='uniform | variational | argmax_variational | deterministic')
    parser.add_argument('--n_report_steps', type=int, default=1)
    parser.add_argument('--wandb_usr', type=str)
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    parser.add_argument('--shuffle_self_condition', type=bool, default=False, help='Shuffle self condition features or not')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--save_model', type=eval, default=True,
                        help='save model')
    parser.add_argument('--generate_epochs', type=int, default=1,
                        help='save model')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
    parser.add_argument('--test_epochs', type=int, default=10)
    parser.add_argument('--sample_epochs', type=int, default=100)
    parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    parser.add_argument("--conditioning", nargs='+', default=[],
                        help='arguments : homo | lumo | alpha | gap | mu | Cv')
    parser.add_argument('--resume', type=str, default=None,
                        help='')
    parser.add_argument('--vae_path', type=str, default=None,
                        help='')
    parser.add_argument('--reflow_model_path', type=str, default=None,
                        help='')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Amount of EMA decay, 0 means off. A reasonable value'
                             ' is 0.999.')
    parser.add_argument('--augment_noise', type=float, default=0)
    parser.add_argument('--n_stability_samples', type=int, default=500,
                        help='Number of samples to compute the stability')
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                        help='normalize factors for [x, categorical, integer]')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=True,
                        help='include atom charge or not')
    parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                        help="Can be used to visualize multiple times per epoch")
    parser.add_argument('--normalization_factor', type=float, default=1,
                        help="Normalize the sum aggregation of EGNN")
    parser.add_argument('--aggregation_method', type=str, default='sum',
                        help='"sum" or "mean"')
    parser.add_argument('--filter_molecule_size', type=int, default=None,
                        help="Only use molecules below this size.")
    parser.add_argument('--sequential', action='store_true',
                        help='Organize mol_data by size to reduce average memory usage.')
    args = parser.parse_args()
    return args
