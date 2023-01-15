"""task1"""
from abhelper.eval.tools import get_spark
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from utils import save_parquet

START_DATE = "2019-11-04"
END_DATE = "2019-12-08"
PATH = "hdfs://bigdata/share/products/nonproduct/ad-hoc/newbie_tasks/musa.safin/task1.parquet"
TXN_ITEM = "HIVE_SSA_MAIN.FCT_RTL_TXN_ITEM"
TXN = "HIVE_SSA_MAIN.FCT_RTL_TXN"
STORE = "HIVE_SSA_BDSD.DIM_STORE"
PLU = "HIVE_SSA_BDSD.DIM_PLU"
WORD = "попкорн"


def filter_positions(df: DataFrame, start_date: str, end_date: str) -> DataFrame:
    """Filtering a table with check positions in a specified time period

    input:
    df -- table with check positions
    start_date -- start date of the time period
    end_date -- the end date of the time period

    output:
    filtered_df -- table containing rtl_txn_id, store_id, plu_id, turnover_no_vat_amt, prime_cost_no_vat_amt
    """
    filtered_df = df.filter(
        (F.col("rtl_txn_dt") >= start_date) & (F.col("rtl_txn_dt") <= end_date)
    ).select(
        "rtl_txn_id",
        "store_id",
        "plu_id",
        "turnover_no_vat_amt",
        "prime_cost_no_vat_amt",
    )
    return filtered_df


def filter_headers(df: DataFrame, start_date: str, end_date: str) -> DataFrame:
    """Filtering the table with check headers in the specified time period and by customers who used a loyalty card

    input:
    df -- table with check headers
    start_date -- start date of the time period
    end_date -- the end date of the time period

    output:
    filtered_df -- table containing rtl_txn_id, rtl_txn_dt
    """
    filtered_df = df.filter(
        (F.col("loyalty_card_no") != "")
        & (F.col("loyalty_card_no").isNotNull())
        & (F.col("rtl_txn_dt") >= start_date)
        & (F.col("rtl_txn_dt") <= end_date)
    ).select("rtl_txn_id", "rtl_txn_dt")
    return filtered_df


def filter_shops(df: DataFrame) -> DataFrame:
    """Filtering store table

    input:
    df -- table of stores

    output:
    filtered_df -- table containing store_id, financial_unit_format_dk
    """
    filtered_df = df.filter(
        (F.col("financial_unit_format_dk") == "O")
        | (F.col("financial_unit_format_dk") == "S")
    ).select("store_id", "financial_unit_format_dk")
    return filtered_df


def filter_plues(df: DataFrame, word: str) -> DataFrame:
    """Filtering the table with products by the presence of the specified word in the UI4

    input:
    df -- products table
    word -- the word that must be present in the UI 4

    output:
    filtered_df -- table containing plu_id, plu_hierarchy_lvl_2_desc
    """
    filtered_df = df.filter(
        F.lower(F.col("plu_hierarchy_lvl_4_desc")).contains(word)
    ).select("plu_id", "plu_hierarchy_lvl_2_desc")
    return filtered_df


def join_tables(
    positions: DataFrame, headers: DataFrame, shops: DataFrame, plues: DataFrame
) -> DataFrame:
    """Connecting tables of positions, cheques, stores and goods

    input:
    positions -- check positions table
    headers -- check headers table
    shops -- stores table
    plues -- goods table

    output:
    joined_df -- joined table
    """
    joined_df = (
        positions.join(headers, on="rtl_txn_id")
        .join(shops, on="store_id")
        .join(plues, on="plu_id")
    )
    return joined_df


def rename_cols_and_values(df: DataFrame) -> DataFrame:
    """Renaming column names and store format values

    input:
    df -- table

    output:
    renamed_df -- table with renamed columns and values
    """
    renamed_df = (
        df.withColumn(
            "financial_unit_format_dk",
            F.when(F.col("financial_unit_format_dk") == "S", "Offline").when(
                F.col("financial_unit_format_dk") == "O", "Online"
            ),
        )
        .withColumn("week", F.weekofyear(F.col("rtl_txn_dt")))
        .withColumn(
            "marg", F.col("turnover_no_vat_amt") - F.col("prime_cost_no_vat_amt")
        )
        .withColumnRenamed("financial_unit_format_dk", "shop_type")
        .withColumnRenamed("store_id", "plant")
        .withColumnRenamed("plu_hierarchy_lvl_2_desc", "ui2_name")
    )
    return renamed_df


def aggregate(df: DataFrame) -> DataFrame:
    """Creating the resulting showcase

    input:
    df -- table

    output:
    aggregated_df -- table with aggregated values
    """
    aggregated_df = df.groupBy(["shop_type", "plant", "ui2_name", "week"]).agg(
        F.sum(df.turnover_no_vat_amt).alias("total_rto"),
        F.countDistinct(df.plu_id).alias("total_traffic"),
        F.sum(df.marg).alias("front_marginality"),
    )
    return aggregated_df


if __name__ == "__main__":
    spark = get_spark()

    fct_rtl_txn_item = spark.table(TXN_ITEM)
    fct_rtl_txn = spark.table(TXN)
    dim_store = spark.table(STORE)
    dim_plu = spark.table(PLU)

    positions = filter_positions(fct_rtl_txn_item, START_DATE, END_DATE)
    headers = filter_headers(fct_rtl_txn, START_DATE, END_DATE)
    shops = filter_shops(dim_store)
    plues = filter_plues(dim_plu, WORD)

    joined_df = join_tables(positions, headers, shops, plues)
    renamed_df = rename_cols_and_values(joined_df)
    aggregated_df = aggregate(renamed_df)
    save_parquet(aggregated_df, PATH)
