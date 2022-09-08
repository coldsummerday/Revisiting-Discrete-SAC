import os.path
from typing import Any, Callable, Optional, Tuple,List

from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger

import  csv
class CSVFileLogger(BaseLogger):
    """

    """
    def __init__(
        self,
        log_path,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
        write_flush: bool = True,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.save_interval = save_interval
        self.write_flush = write_flush
        self.last_save_step = -1
        self.eval_file_handler = open(os.path.join(log_path,"progress_eval.csv"),'w')
        self.train_file_handler = open(os.path.join(log_path,"progress_train.csv"),'w')

        self.eval_csv_log_writer = csv.DictWriter(self.eval_file_handler,
                                                  fieldnames=[ "test/reward","test/reward_std", "test/length","test/length_std",
                                                              "test/env_step"])

        self.train_csv_log_writer = csv.DictWriter(self.train_file_handler, fieldnames=['train/reward','train/length',"train/episode","train/env_step"])

        self.eval_csv_log_writer.writeheader()
        self.train_csv_log_writer.writeheader()




    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        "write do nothing"
        pass

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if collect_result["n/ep"] > 0:
            if step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    "train/episode": collect_result["n/ep"],
                    "train/reward": collect_result["rew"],
                    "train/length": collect_result["len"],
                    "train/env_step":step
                }
                self.train_csv_log_writer.writerow(log_data)
                self.train_file_handler.flush()
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],

            }
            self.eval_csv_log_writer.writerow(log_data)
            self.eval_file_handler.flush()
            # self.write("test/env_step", step, log_data)
            self.last_log_test_step = step

class LoggerMerge(BaseLogger):
    def __init__(self,loggers:List[BaseLogger]):
        self.loggers = loggers

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        pass

    def log_test_data(self, collect_result: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_test_data(collect_result, step)

    def log_train_data(self, collect_result: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_train_data(collect_result, step)
    def log_update_data(self, update_result: dict, step: int) -> None:
        for logger in self.loggers:
            logger.log_update_data(update_result, step)


