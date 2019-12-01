from .cornernet import cornernet
from .cornernet_saccade import cornernet_saccade

import os
import sys


def test_func(sys_config, db, nnet, result_dir, debug=False):
    print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                  sys._getframe().f_code.co_name) + "\033[0m")
    print("\033[0;36m " + "{}".format("这个地方根据我们的配置文件可能分别进入两个模型(CornerNet或者CornerNet_Saccade)") + "\033[0m")
    return globals()[sys_config.sampling_function](db, nnet, result_dir, debug=debug)
