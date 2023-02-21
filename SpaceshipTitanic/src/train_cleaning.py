from src.logger_setup import Setup
import os
import logging
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import missingno as msno


class TrainCleaning:

    def __init__(self) -> None:
        self.setup = Setup()
        self.config = self.setup.config
        self.logger = logging.getLogger("TrainCleaning")
        self.raw_path = self.setup.ROOT_PATH + self.config["raw"]["RawDFDir"]

    def to_pickle(self, compression=None) -> int:
        raw_dir_abs_path = self.setup.ROOT_PATH + self.config["raw"]["RawDFDir"]
        raw_files = os.listdir(raw_dir_abs_path)
        csvs = [raw_dir_abs_path + f for f in raw_files if f.endswith(".csv")]
        for path in csvs:
            self.logger.info(path)
            df = pd.read_csv(path)
            self.logger.info(f"{self.config['raw']['TrainPkl']}")
            df.to_pickle(path=self.config['raw']['TrainPkl'], compression=compression)

        return 0

    def get_missing_data_statistics(self, plot_name="RawDataNABar") -> int:
        with open(f"{self.config['raw']['TrainPkl']}", "rb") as handle:
            df_train_raw = pickle.load(handle)
        fig = plt.figure()
        msno.bar(df_train_raw, figsize=(10, 4), fontsize=8)
        if not os.path.exists("reports/"):
            os.mkdir("reports/")
        fig.savefig(f"reports/{plot_name}.pdf", bbox_inches="tight")
        self.logger.info(f"reports/{plot_name}.pdf")

        return 0
