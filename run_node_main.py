from node_code.configs.config import args_parser
from node_code.helpers.metrics_utils import log_test_results
import numpy as np
import wandb
from node_code.helpers.helpers import set_random_seed
args = args_parser()
project_name = [args.proj_name, args.proj_name+ "debug"]
proj_name = project_name[0]

def main(args):
    model_name = args.model
    Alg_name = "Alg-" + args.agg_method
    file_name = Alg_name + 'D-{}_M-{}_IID-{}_NW-{}_NM-{}_EB-{}_TS-{}_TPye-{}_TPo-{}_PI-{}_OR-{}'.format(
        args.dataset,
        model_name,
        args.is_iid,
        args.num_workers,
        args.num_mali,
        args.epoch_backdoor,
        args.trigger_size,
        args.trigger_type,
        args.trigger_position,
        args.poisoning_intensity,
        args.overlapping_rate,
    )

    average_overall_performance_list, average_ASR_list, average_Flip_ASR_list, average_transfer_attack_success_rate_list = [], [], [], []
    a_list, r_list, v_list = [], [], []
    results_table = []
    metric_list = []

    from node_main import main as backdoor_main
    set_random_seed(args.seed)
    rs = np.random.RandomState(args.seed)
    for i in range(args.round):
        set_random_seed(args.seed)

        logger = wandb.init(
            project=proj_name,
            group=file_name,
            name=f"round_{i}",
            config=args,
        )

        average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate, a, r, v = backdoor_main(
            args, logger)
        results_table.append(
            [average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate])
        a_list.append(a)
        r_list.append(r)
        v_list.append(v)
        logger.log({"average_overall_performance": average_overall_performance,
                    "average_ASR": average_ASR,
                    "average_Flip_ASR": average_Flip_ASR,
                    "average_transfer_attack_success_rate": average_transfer_attack_success_rate})

        average_overall_performance_list.append(average_overall_performance)
        average_ASR_list.append(average_ASR)
        average_Flip_ASR_list.append(average_Flip_ASR)
        average_transfer_attack_success_rate_list.append(average_transfer_attack_success_rate)
        wandb.finish()

    columns = ["average_overall_performance", "average_ASR", "average_Flip_ASR", "average_transfer_attack_success_rate"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger = wandb.init(
        project=proj_name,
        group=file_name,
        name=f"exp_results",
        config=args,
    )
    table_logger.log({"results": logger_table})
    wandb.finish()

    mean_average_overall_performance = np.mean(np.array(average_overall_performance_list))
    mean_average_ASR = np.mean(np.array(average_ASR_list))
    mean_average_Flip_ASR = np.mean(np.array(average_Flip_ASR_list))
    mean_average_transfer_attack_success_rate = np.mean(np.array(average_transfer_attack_success_rate_list))

    std_average_overall_performance = np.std(np.array(average_overall_performance_list))
    std_average_ASR = np.std(np.array(average_ASR_list))
    std_average_Flip_ASR = np.std(np.array(average_Flip_ASR_list))
    std_average_transfer_attack_success_rate = np.std(np.array(average_transfer_attack_success_rate_list))

    header = ['dataset', 'model', "mean_average_overall_performance",
              "std_average_overall_performance", "mean_average_ASR", "std_average_ASR",
              "mean_average_Flip_ASR", "std_average_Flip_ASR",
              "mean_average_local_unchanged_acc", "std_average_transfer_attack_success_rate"]
    paths = "./checkpoints/Node/"

    metric_list.append(args.dataset)
    metric_list.append(model_name)
    metric_list.append(mean_average_overall_performance)
    metric_list.append(std_average_overall_performance)
    metric_list.append(mean_average_ASR)
    metric_list.append(std_average_ASR)
    metric_list.append(mean_average_Flip_ASR)
    metric_list.append(std_average_Flip_ASR)
    metric_list.append(mean_average_transfer_attack_success_rate)
    metric_list.append(std_average_transfer_attack_success_rate)

    paths = paths + "data-{}/".format(args.dataset) + "model-{}/".format(model_name) + file_name
    log_test_results(paths, header, file_name)
    log_test_results(paths, metric_list, file_name)


if __name__ == '__main__':
    args = args_parser()
    main(args)
