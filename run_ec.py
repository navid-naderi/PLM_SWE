import os
import sys
import math
import pprint
import random

import numpy as np

import torch
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ec import util, dataset, task


def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 500)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        metric = solver.evaluate("valid")
        print(metric)
        
        for k, v in metric.items():
            if k.startswith(cfg.eval_metric):
                result = v
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
            solver.save("best_models/ec_best_model.pt")
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)

    solver.load("best_models/ec_best_model.pt")
    return best_result, solver


if __name__ == "__main__":

    args, vars = util.parse_args()

    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg, args)

    if os.path.isfile(working_dir + '/results.pt'):
        print('Exiting since this specific run has already been completed!')

    else:
        seed = args.seed
        torch.manual_seed(seed + comm.get_rank())
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger = util.get_root_logger()
        if comm.get_rank() == 0:
            logger.warning("Config file: %s" % args.config)
            logger.warning(pprint.pformat(cfg))

        cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
        train_set = core.Configurable.load_config_dict(cfg.dataset)
        cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
        valid_set = core.Configurable.load_config_dict(cfg.dataset)
        cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
        test_set = core.Configurable.load_config_dict(cfg.dataset)
        dataset = (train_set, valid_set, test_set)

        solver, scheduler = util.build_downstream_solver(cfg, dataset)

        best_val_result, solver = train_and_validate(cfg, solver, scheduler)

        logger.warning("Testing on the test set with sequence identity 95%")
        test_95_metric = solver.evaluate("test")

        cfg.dataset.split = "test"
        cfg.dataset.percent = 50
        test_set50 = core.Configurable.load_config_dict(cfg.dataset)
        solver.test_set = test_set50
        logger.warning("Testing on the test set with sequence identity 50%")
        test_50_metric = solver.evaluate("test")

        cfg.dataset.percent = 30
        test_set30 = core.Configurable.load_config_dict(cfg.dataset)
        solver.test_set = test_set30
        logger.warning("Testing on the test set with sequence identity 30%")
        test_30_metric = solver.evaluate("test")

        torch.save([best_val_result, test_95_metric, test_50_metric, test_30_metric], working_dir + '/results.pt')
        
    print('Done!')