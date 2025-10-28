from contextlib import contextmanager
from typing import Tuple
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
import os
import pandas as pd


def get_snowflake_connection_dict(running_in_azure: bool = False) -> dict:
    """Load a dictionary of snowflake connection properties.

    Note that the account passwords (and service username) are stored in environment variables or an Azure keyvault.

    Args:
        running_in_azure (bool, optional): Whether to use the service account (if we're running in Azure) or not. Defaults to False.

    Returns:
        dict: _description_
    """

    if running_in_azure:
        raise NotImplementedError("Need to implement Azure access")
    else:
        SNOWFLAKE_CONFIG = {
            "user": "sam.vaughan@betr.com.au",
            "password": os.getenv("SNOWFLAKE_PASS"),
            "account": "ci36650.australia-east.azure",
            "warehouse": "MACHINE_LEARNING_WH",
            "database": "ANALYTICS_SANDBOX",
            "schema": "SANDBOX",
            "role": "ANALYST_ADMIN_FR",
            "client_session_keep_alive": True,
        }
    return SNOWFLAKE_CONFIG


@contextmanager
def snowflake_connection(SNOWFLAKE_CONFIG: dict):
    """
    Context manager to establish and close a Snowflake connection.

    Yields:
        snowflake.connector.connection: An active Snowflake connection.
    """

    conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
    try:
        yield conn
    finally:
        if conn:
            conn.close()
            print("Snowflake connection closed.")


def read_snowflake_query_to_df(snowflake_config: dict, sql: str) -> pd.DataFrame:
    """
    Executes a SQL query and returns the result as a pandas DataFrame.

    Args:
        sql (str): The SQL query to execute.

    Returns:
        pd.DataFrame: The query result.
    """
    with snowflake_connection(snowflake_config) as conn:
        cur = conn.cursor()
        try:
            cur.execute(sql)
            df = cur.fetch_pandas_all()
            return df
        finally:
            cur.close()


def write_df_to_snowflake_table(
    snowflake_config: dict, df: pd.DataFrame, table_name: str, **kwargs
) -> Tuple[bool, int, int]:
    """Write a pandas dataframe to a snowflake table

    Args:
        snowflake_config (dict): A snowflake connection dictionary
        df (pd.DataFrame): A pandas dataframe we want to write to snowflake
        table_name (str): Name of a table in snowflake
        kwargs (dict): Extra kwargs to pass to the write_pandas function from snowflake.connector

    Returns:
        Tuple: A tuple fo success, n_chunks, nrows
    """

    with snowflake_connection(snowflake_config) as conn:
        try:
            success, nchunks, nrows, _ = write_pandas(
                conn, df, table_name, auto_create_table=True, **kwargs
            )
            return success, nchunks, nrows
        finally:
            conn.close()


def run_snowflake_procedure(snowflake_config: dict, proc_sql: str) -> None:
    """
    Calls a Snowflake stored procedure and waits for completion.
    Args:
        snowflake_config (dict): A snowflake connection dictionary

    Returns:
        None
    Raises:
        RuntimeError: If procedure execution fails.
    """

    with snowflake_connection(snowflake_config) as conn:
        try:
            print(f"Running Snowflake stored procedure: {proc_sql}")
            cur = conn.cursor()
            cur.execute(proc_sql)
            print("Procedure completed successfully.")
        finally:
            cur.close()
            conn.close()
