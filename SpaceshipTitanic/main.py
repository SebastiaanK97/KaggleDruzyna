from src.train_cleaning import *
import sys


def main():

    tc = TrainCleaning()
    tc.to_pickle()
    tc.get_missing_data_statistics()

    sys.exit("Aborting.")


if __name__ == "__main__":
    main()
