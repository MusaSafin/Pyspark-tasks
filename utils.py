"""utils"""
import pydoop.hdfs as hd
import pickle

def save_parquet(df, path):
    """Save dataframe in parquet format"""
    
    return df.repartition(1).write.mode("overwrite").parquet(path)

def save_pickle(values_dict, path):
    """Save pickle file"""
    with hd.open(path, 'w') as file:
        pickle.dump(values_dict, file)