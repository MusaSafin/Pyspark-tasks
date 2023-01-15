"""task3"""
import random

import numpy as np
import pandas as pd
from abhelper.eval.tools import get_spark
from matplotlib import pyplot as plt
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from utils import save_pickle

TXN = "HIVE_SSA_MAIN.FCT_RTL_TXN"
STORE = "HIVE_SSA_BDSD.DIM_STORE"
PATH = (
    "hdfs://bigdata/share/products/nonproduct/ad-hoc/newbie_tasks/musa.safin/task3.pkl"
)
START_DATE = "2020-02-24"
END_DATE = "2020-05-03"
UNIT_FORMAT = "D"
SUBJECT = 77
WEEKS_NUM = 10
RANDOM_SEED = 328
ITER_NUM = 1000
GROUP_SIZES = [5, 20, 100]
METHODS = ["random", "nearest"]


def filter_headers(headers: DataFrame, start_date: str, end_date: str) -> DataFrame:
    """Filtering the check header table in the specified time period

    input:
    headers -- check headers table
    start_date -- start date of the time period
    end_date -- end date of the time period

    output:
    filtered_headers -- filtered table
    """
    filtered_headers = (
        headers.filter(
            (F.col("rtl_txn_dt") >= start_date) & (F.col("rtl_txn_dt") < end_date)
        )
        .withColumn("week", F.weekofyear("rtl_txn_dt"))
        .select("store_id", "rtl_txn_dt", "week", "turnover_no_vat_amt")
    )
    return filtered_headers


def filter_stores(
    stores: DataFrame,
    headers: DataFrame,
    unit_format: str,
    subject: int,
    weeks_num: int,
) -> DataFrame:
    """Filtering the store table

    input:
    stores -- stores table
    headers -- check headers table
    unit_format -- store format
    subject -- subject code
    weeks_num -- number of weeks in the time period

    output:
    filtered_stores -- filtered store table with only active stores
    """
    moscow_D_stores = stores.filter(
        (F.col("financial_unit_format_dk") == unit_format)
        & (F.col("federal_subject_dk") == subject)
    ).select("store_id")

    filtered_stores = (
        headers.join(moscow_D_stores, "store_id")
        .groupBy("store_id")
        .agg(F.countDistinct("week").alias("active_weeks_cnt"))
        .filter(F.col("active_weeks_cnt") == weeks_num)
        .select("store_id")
        .join(headers, "store_id")
    )

    return filtered_stores


def aggregate_stores(filtered_stores: DataFrame) -> pd.DataFrame:
    """Getting rto for each store and week pair

    input:
    filtered_stores -- filtered store table

    output:
    aggregated_stores -- rto value for each store and week pair
    """
    rto_store_week_df = (
        filtered_stores.groupBy("store_id", "week")
        .agg(F.sum("turnover_no_vat_amt").alias("total_rto"))
        .select("store_id", "total_rto", "week")
        .toPandas()
    )
    rto_store_week_df["total_rto"] = rto_store_week_df["total_rto"].astype("float")
    aggregated_stores = rto_store_week_df.pivot_table(
        index="store_id", columns="week", values="total_rto"
    )
    return aggregated_stores


def get_statistic_dicts(
    df: pd.DataFrame, group_sizes: list, iter_num: int, methods: list
) -> tuple:
    """Calculation of resulting dictionaries with averages and variances for each of the specified methods

    input:
    df -- rto value for each store and week pair
    group_sizes -- list of sample sizes
    iter_num -- number of iterations
    methods -- methods list

    output:
    method_size_2_mean_diff -- iter_num of mean difference values for each method and each sample size dict
    method_2_statistics -- mean and std for each method and each sample size dict
    """
    store_set = df.index.unique()
    method_size_2_mean_diff = {}
    method_2_statistics = {}

    for method in methods:
        method_size_2_mean_diff[method] = {}
        method_2_statistics[method] = {"mean": [], "std": []}
        group_size_2_diffs = {}
        for group_size in group_sizes:
            mean_diff_array = []
            for _ in tqdm(range(iter_num)):
                if method == "random":
                    group1, group2 = get_random_groups(store_set, group_size)
                elif method == "nearest":
                    group1, group2 = get_nearest_groups(store_set, group_size, df)

                mean_diff_array.append(
                    np.mean(aggregated_stores.query("index.isin(@group1)").mean())
                    - np.mean(aggregated_stores.query("index.isin(@group2)").mean())
                )
                
            method_size_2_mean_diff[method][group_size] = np.array(mean_diff_array)
                
            method_2_statistics[method]["mean"].append(
                    np.mean(np.array(mean_diff_array))
                )
            method_2_statistics[method]["std"].append(np.std(np.array(mean_diff_array)))

    return method_size_2_mean_diff, method_2_statistics


def get_random_groups(store_set: np.array, group_size: int) -> tuple:
    """Getting two non-overlapping random groups of stores

    input:
    store_set -- store id's
    group_size -- sample size

    output:
    group1, group2 -- numpy arrays of store id's
    """
    random_stores = np.random.choice(store_set, 2 * group_size, replace=False)
    group1, group2 = random_stores[:group_size], random_stores[group_size:]
    return group1, group2


def get_nearest_groups(
    store_set: np.array, group_size: int, df: pd.DataFrame
) -> tuple:
    """The 1st group is selected randomly. The 2nd group gets closest to the 1st

    input:
    store_set -- store id's
    group_size -- sample size
    df -- rto value for each store and week pair

    output:
    group1, group2 -- numpy arrays of store id's
    """
    group1 = np.random.choice(store_set, group_size, replace=False)
    is_used = df.index.isin(group1)
    group1_df = df[is_used]
    not_used_df = df[~is_used]

    dists = pairwise_distances(group1_df, not_used_df)
    group2 = []
    not_used_list = not_used_df.index
    for i in range(len(dists)):
        min_index = dists[i].argmin()
        min_id = not_used_list[min_index]
        group2.append(min_id)
        dists = np.delete(dists, min_index, 1)
        not_used_list = np.delete(not_used_list, min_index)
    group2 = np.array(group2)
    return group1, group2


def draw_and_save_fig(method_size_2_mean_diff: dict) -> None:
    """Building histograms for each method and each sample size

    input:
    method_size_2_mean_diff -- mean difference values for each method and each sample size dict
    """
    fig = plt.figure(figsize=(15, 20))
    for i in range(1, 3):
        method = "random" if i == 1 else "nearest"
        plt.subplot(2, 1, i)
        for group_size in GROUP_SIZES:
            plt.hist(
                method_size_2_mean_diff[method][group_size],
                bins=50,
                density=True,
                alpha=0.5,
                label="k={}".format(group_size),
            )
        plt.ylabel("density", size=14)
        plt.title("{} method".format(method))
        plt.legend(loc="upper right")
    plt.savefig("hist")


if __name__ == "__main__":
    spark = get_spark()

    headers = spark.table(TXN)
    stores = spark.table(STORE)

    filtered_headers = filter_headers(headers, START_DATE, END_DATE)
    filtered_stores = filter_stores(
        stores, filtered_headers, UNIT_FORMAT, SUBJECT, WEEKS_NUM
    )
    aggregated_stores = aggregate_stores(filtered_stores)

    random.seed(RANDOM_SEED)
    method_size_2_mean_diff, method_2_statistics = get_statistic_dicts(
        aggregated_stores, GROUP_SIZES, ITER_NUM, METHODS
    )
    draw_and_save_fig(method_size_2_mean_diff)
    save_pickle(method_2_statistics, PATH)

"""
Чем больше размер выборки, тем ближе статистики посчитанные по ней к статистиками генеральной совокупности и меньше дисперсия. 
При жадном подборе похожих магазинов дисперсия меньше, чем при рандомном.
"""
