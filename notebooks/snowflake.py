import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from cryptography.hazmat.primitives import serialization
    from snowflake.sqlalchemy import URL
    from sqlalchemy import create_engine

    with open(os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"], "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    engine = create_engine(
        URL(
            account=os.environ["SNOWFLAKE_ACCOUNT"],  # use privatelink suffix: acct.region.privatelink
            user=os.environ["SNOWFLAKE_USER"],
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.environ["SNOWFLAKE_SCHEMA"],
            role=os.environ.get("SNOWFLAKE_ROLE", ""),
        ),
        connect_args={"private_key": private_key_bytes},
    )
    return (engine,)


@app.cell
def _(mo):
    _df = mo.sql(
        f"""
        SELECT * FROM
        """
    )
    return


if __name__ == "__main__":
    app.run()
