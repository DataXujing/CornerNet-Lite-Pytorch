#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib

from core.dbs import datasets
from core.test import test_func
from core.config import SystemConfig
from core.nnet.py_factory import NetworkFactory

torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--testiter", dest="testiter",
                        help="test at iteration i",
                        default=None, type=int)  ## 测试的模型参数： 'cfg_file' + '_' + 'testiter' + '.pkl'
    parser.add_argument("--split", dest="split",
                        help="which split to use",
                        default="validation", type=str)  #valid or test
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    return args


# 创建文件夹的函数
def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


# 测试函数
def test(db, system_config, model, args):
    split = args.split
    testiter = args.testiter
    debug = args.debug
    suffix = args.suffix

    # 输出的文件夹result_dir+testiter+split
    result_dir = system_config.result_dir
    result_dir = os.path.join(result_dir, str(testiter), split)

    # 后缀的添加
    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)

    # 创建文件夹
    make_dirs([result_dir])

    # 赋值test_iter 如果没有传入就用预设值
    test_iter = system_config.max_iter if testiter is None else testiter
    print("loading parameters at iteration: {}".format(test_iter))

    # 构建神经网络
    print("building neural network...")
    nnet = NetworkFactory(system_config, model)
    print("loading parameters...")
    nnet.load_params(test_iter)

    nnet.cuda()
    nnet.eval_mode()
    test_func(system_config, db, nnet, result_dir, debug=debug)


def main(args):
    # 后缀的与否，以及整个配置文件在config文件夹下
    if args.suffix is None:
        cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    else:
        cfg_file = os.path.join("./configs", args.cfg_file + "-{}.json".format(args.suffix))
    print("cfg_file: {}".format(cfg_file))

    # 使用json.load读取json配置文件
    with open(cfg_file, "r") as f:
        config = json.load(f)

    # 添加快照的配置，并在完成后生成系统配置类的对象
    config["system"]["snapshot_name"] = args.cfg_file
    system_config = SystemConfig().update_config(config["system"])

    # 模型文件名生成 导入模型 初始化模型
    model_file = "core.models.{}".format(args.cfg_file)
    model_file = importlib.import_module(model_file)
    model = model_file.model()

    # 考虑训练步长、验证步长和测试步长
    train_split = system_config.train_split
    val_split = system_config.val_split
    test_split = system_config.test_split

    # 默认使用的是validation的split分割
    split = {
        "train": train_split,
        "valid": val_split,
        "test": test_split
    }[args.split]

    print("loading all datasets...")
    dataset = system_config.dataset
    print("split: {}".format(split))
    testing_db = datasets[dataset](config["db"], split=split, sys_config=system_config)

    print("system config...")
    pprint.pprint(system_config.full)

    print("db config...")
    pprint.pprint(testing_db.configs)

    test(testing_db, system_config, model, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
