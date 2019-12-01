#!/usr/bin/env python
import os
import sys
import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Pool

from core.dbs import datasets
from core.utils import stdout_to_tqdm
from core.config import SystemConfig
from core.sample import data_sampling_func
from core.nnet.py_factory import NetworkFactory

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--initialize", action="store_true")

    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--world-size", default=-1, type=int,
                        help="number of nodes of distributed training")
    parser.add_argument("--rank", default=0, type=int,
                        help="node rank for distributed training")
    parser.add_argument("--dist-url", default=None, type=str,
                        help="url used to set up distributed training")
    parser.add_argument("--dist-backend", default="nccl", type=str)

    args = parser.parse_args()
    return args


def prefetch_data(system_config, db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    # 用pid[进程id]做随机数种子
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(system_config, db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e


def _pin_memory(ts):
    if type(ts) is list:
        return [t.pin_memory() for t in ts]
    return ts.pin_memory()


def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [_pin_memory(x) for x in data["xs"]]
        data["ys"] = [_pin_memory(y) for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return


def init_parallel_jobs(system_config, dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(system_config, db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks


def terminate_tasks(tasks):
    for task in tasks:
        task.terminate()


# 训练函数（训练数据，验证数据，系统参数，模型，参数）
def train(training_dbs, validation_db, system_config, model, args):
    print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                  sys._getframe().f_code.co_name) + "\033[0m")

    # reading arguments from command
    start_iter = args.start_iter
    distributed = args.distributed
    world_size = args.world_size
    initialize = args.initialize
    gpu = args.gpu
    rank = args.rank

    # reading arguments from json file
    batch_size = system_config.batch_size
    learning_rate = system_config.learning_rate
    max_iteration = system_config.max_iter
    pretrained_model = system_config.pretrain
    stepsize = system_config.stepsize
    snapshot = system_config.snapshot
    val_iter = system_config.val_iter
    display = system_config.display
    decay_rate = system_config.decay_rate
    stepsize = system_config.stepsize

    print("\033[1;36m " + "Process {}: building model(生成模型中)...".format(rank) + "\033[0m")
    nnet = NetworkFactory(system_config, model, distributed=distributed, gpu=gpu)

    if initialize:
        nnet.save_params(0)
        exit(0)

    # 开4个队列去存数据
    # queues storing data for training
    training_queue = Queue(system_config.prefetch_size)
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue = queue.Queue(system_config.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # allocating resources for parallel reading
    training_tasks = init_parallel_jobs(system_config, training_dbs, training_queue, data_sampling_func, True)
    if val_iter:
        validation_tasks = init_parallel_jobs(system_config, [validation_db], validation_queue, data_sampling_func,
                                              False)

    training_pin_semaphore = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    # 看是否有先训练的模型
    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("Process {}: loading from pretrained model".format(rank))
        nnet.load_pretrained_params(pretrained_model)

    # 看有没有开始的迭代器
    if start_iter:
        nnet.load_params(start_iter)
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.set_lr(learning_rate)
        print("Process {}: training starts from iteration {} with learning_rate {}".format(rank, start_iter + 1,
                                                                                           learning_rate))
    else:
        nnet.set_lr(learning_rate)

    # 训练模型
    if rank == 0:
        print("\033[1;36m " + "training start(训练开始)...".format(rank) + "\033[0m")

    nnet.cuda()
    nnet.train_mode()

    #
    with stdout_to_tqdm() as save_stdout:
        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss = nnet.train(**training)

            # 如果设置了display的步长，我们在步长整数倍时展示损失函数的值
            if display and iteration % display == 0:
                print(
                    "\033[1;36m " + "Process(进程){}: iteration(迭代数) [{}]时的training loss(损失函数值):".format(
                        rank, iteration) + "\033[0m" + "{}".format(training_loss.item()))
            del training_loss

            # 如果设置了变量迭代器步长[这边是验证集了]
            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("\033[1;33m " + "Process {}:".format(
                    rank) + "\033[0m" + "\033[1;36m " + "validation loss at iteration {}:".format(
                    iteration) + "\033[0m" + "{}".format(validation_loss.item()))
                nnet.train_mode()

            # 快照步长
            if iteration % snapshot == 0 and rank == 0:
                nnet.save_params(iteration)

            # 学习率更新步长
            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                print("\033[1;35m " + "此时学习率更新为:" + "\033[0m" + "{}".format(learning_rate))
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    # 杀掉进程的消息
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    # 结束任务的消息
    terminate_tasks(training_tasks)
    terminate_tasks(validation_tasks)


# 主函数main
def main(gpu, ngpus_per_node, args):
    # 将gpu复制到args里
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    rank = args.rank

    # 读取配置文件
    # cfg_file是必须传入的要训练模型名字
    cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)

    # 上面的读取的单独模型配置的"system"里添加一项
    config["system"]["snapshot_name"] = args.cfg_file

    # 更新参数配置
    system_config = SystemConfig().update_config(config["system"])

    # 模型的名字作为模型导入
    model_file = "core.models.{}".format(args.cfg_file)
    model_file = importlib.import_module(model_file)
    model = model_file.model()

    # 从i系统配置里取出的参数
    train_split = system_config.train_split
    val_split = system_config.val_split

    print("Process {}: loading all datasets...".format(rank))

    # 用了几个worker
    dataset = system_config.dataset
    workers = args.workers
    print("Process {}: using {} workers".format(rank, workers))

    training_dbs = [datasets[dataset](config["db"], split=train_split, sys_config=system_config) for _ in
                    range(workers)]
    validation_db = datasets[dataset](config["db"], split=val_split, sys_config=system_config)

    if rank == 0:
        print("system config...")
        pprint.pprint(system_config.full)

        print("db config...")
        pprint.pprint(training_dbs[0].configs)

        print("len of db: {}".format(len(training_dbs[0].db_inds)))
        print("distributed: {}".format(args.distributed))

    # 调用train训练函数
    train(training_dbs, validation_db, system_config, model, args)


if __name__ == "__main__":
    # 先parse参数
    args = parse_args()

    # 分布式的选项以及节点数量
    distributed = args.distributed
    world_size = args.world_size

    # 分布式节点必须大于0
    if distributed and world_size < 0:
        raise ValueError("world size must be greater than 0 in distributed training")

    # 每个节点上的gpu
    ngpus_per_node = torch.cuda.device_count()

    # torch 下分布式运行
    if distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print(ngpus_per_node)
        main(None, ngpus_per_node, args)
