import sys

from main_dmt import build_parser as build_dmt_parser


RUN_NAME = 'genesis_experiment'
LOCAL_CONFIG_OVERRIDES = {
    'dataset': 'Genesis',
    'data_path': 'dataset/Genesis',
    'v2_first_kernel_size': 3,
    'active_view': 'dual',
    'state_dim': 64,
    'num_prototypes': 5,
    'lambda_state_consistency': 1.0,
    'batch_size': 64,
    'cache_windows': True,
    'device': 'cuda',
    'stage1_use_injected_triplet': True,
    'stage1_triplet_margin': 0.3,
    'visualization_method': 'pca',
}


def run_dmt_preset(argv):
    parser = build_dmt_parser()
    parser.set_defaults(
        dataset='Genesis',
        data_path='dataset/Genesis',
        input_c=18,
    )
    args = parser.parse_args(argv)
    from solver_dmt import DMTSolver

    DMTSolver(args).run()


def main():
    if '--dmt' in sys.argv[1:]:
        argv = [item for item in sys.argv[1:] if item != '--dmt']
        run_dmt_preset(argv)
        return

    from train import parse_common_preset_args, run_dataset_preset

    args = parse_common_preset_args("Run the Genesis preset experiment.")
    run_dataset_preset('Genesis', RUN_NAME, LOCAL_CONFIG_OVERRIDES, args)


if __name__ == "__main__":
    main()
