"""task2"""
from abhelper.eval.tools import get_spark
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F

from utils import save_parquet

TXN = "HIVE_SSA_MAIN.FCT_RTL_TXN"
STORE = "HIVE_SSA_BDSD.DIM_STORE"
START_DATE = "2019-01-01"
END_DATE = "2020-01-01"
MONTH_NUM = 12
WEEKDAY = "weekday"
PLANT = "plant"
PATH = "hdfs://bigdata/share/products/nonproduct/ad-hoc/newbie_tasks/musa.safin/task2.parquet"


def filter_and_join_tables(
    headers: DataFrame, stores: DataFrame, start_date: str, end_date: str
) -> DataFrame:
    """Filtering and joining a tables

    input:
    headers -- headers table
    stores -- stores table

    output:
    joined_df -- joined tables
    """
    headers_filtered = headers.filter(
        (F.col("rtl_txn_dt") >= start_date)
        & (F.col("rtl_txn_dt") < end_date)
        & (F.col("loyalty_card_no").isNotNull())
        & (F.col("loyalty_card_no") != "")
    ).select(
        "store_id", "loyalty_card_no", "rtl_txn_dt", "rtl_txn_id", "turnover_no_vat_amt"
    )

    stores_filtered = stores.filter(F.col("financial_unit_format_dk") == "D").select(
        "store_id", "region_nm"
    )

    joined_df = (
        headers_filtered.join(stores_filtered, "store_id")
        .withColumn("month", F.month("rtl_txn_dt"))
        .withColumn("weekday", F.dayofweek("rtl_txn_dt"))
        .withColumnRenamed("store_id", "plant")
        .select(
            "loyalty_card_no",
            "rtl_txn_dt",
            "rtl_txn_id",
            "month",
            "weekday",
            "turnover_no_vat_amt",
            "region_nm",
            "plant",
        )
    )
    return joined_df


def select_active_clients(df: DataFrame, month_num: int) -> DataFrame:
    """Filtering clients been active every month

    input:
    df -- table
    month_num -- number of months in a year

    output:
    filtered_df -- fultered table
    """
    active_clients = (
        df.groupby("loyalty_card_no")
        .agg(F.countDistinct("month").alias("active_month_num"))
        .filter(F.col("active_month_num") == month_num)
        .select("loyalty_card_no")
    )
    filtered_df = df.join(active_clients, "loyalty_card_no")
    return filtered_df


def get_most_freq_val_for_client(df: DataFrame, col_name: str) -> DataFrame:
    """Finding the most popular value in column for each client

    input:
    df -- table
    col_name -- column name

    output:
    most_freq_val_for_client -- the most frequent value for each client
    """
    most_freq_val_for_client = (
        df.groupby("loyalty_card_no", col_name)
        .agg(F.count("rtl_txn_id").alias("visit_cnt"))
        .withColumn(
            "feature_rank",
            F.row_number().over(
                W.partitionBy("loyalty_card_no").orderBy(F.desc("visit_cnt"))
            ),
        )
        .filter(F.col("feature_rank") == 1)
        .select(
            F.col("loyalty_card_no").alias("zloyid"),
            F.col(col_name).alias("most_freq_" + col_name),
        )
    )
    return most_freq_val_for_client


def aggregate_values(active_clients_df: DataFrame, month_num: int) -> DataFrame:
    """Counting statistics

    input:
    df -- table
    month_num -- number of months in a year

    output:
    aggregated_df -- table ith aggregated values
    """
    aggregated_df = (
        active_clients_df.groupby("loyalty_card_no")
        .agg(
            F.sum("turnover_no_vat_amt").alias("total_rto"),
            F.count("rtl_txn_id").alias("total_traffic"),
            (F.sum("turnover_no_vat_amt") / F.count("rtl_txn_id")).alias("avg_check"),
            F.countDistinct("plant").alias("n_plants"),
            (F.count("rtl_txn_id") / month_num).alias("avg_monthly_traffic"),
            (F.sum("turnover_no_vat_amt") / month_num).alias("avg_monthly_rto"),
            F.countDistinct("region_nm").alias("n_regions"),
        )
        .select(
            F.col("loyalty_card_no").alias("zloyid"),
            "total_rto",
            "total_traffic",
            "n_plants",
            "avg_check",
            "avg_monthly_rto",
            "avg_monthly_traffic",
            "n_regions",
        )
    )
    return aggregated_df


if __name__ == "__main__":
    spark = get_spark()

    headers = spark.table(TXN)
    stores = spark.table(STORE)

    joined_df = filter_and_join_tables(headers, stores, START_DATE, END_DATE)
    active_clients_df = select_active_clients(joined_df, MONTH_NUM)
    most_freq_plant = get_most_freq_val_for_client(active_clients_df, PLANT)
    most_freq_weekday = get_most_freq_val_for_client(active_clients_df, WEEKDAY)
    most_freq_plant_and_weekday = most_freq_plant.join(most_freq_weekday, "zloyid")
    aggregated_df = aggregate_values(active_clients_df, MONTH_NUM)
    result_df = aggregated_df.join(most_freq_plant_and_weekday, "zloyid")

    save_parquet(result_df, PATH)
