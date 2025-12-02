from pyAFL.utils import snowflake_utils
import pandas as pd

if __name__ == "__main__":

    rpf_sql = """
    select * from dwh.ref.ref_product_fee where event_type = 'Australian Rules' and date_to > '2025-01-01'
    """

    poc_sql = """
    select * from dwh.ref.ref_poc_tax where date_to > '2025-01-01'
    """

    config = snowflake_utils.get_snowflake_connection_dict()
    rpf_df = snowflake_utils.read_snowflake_query_to_df(config, rpf_sql).rename(
        str.lower, axis=1
    )
    rpf_df.to_parquet("data/fees/rpf_costs.parquet")

    poc_df = snowflake_utils.read_snowflake_query_to_df(config, poc_sql).rename(
        str.lower, axis=1
    )
    poc_df.to_parquet("data/fees/poc_costs.parquet")
