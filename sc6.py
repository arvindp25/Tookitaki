
def optimize_shuffle_partitions(spark, data_size_gb: float = 60.0):
    """
    Optimization A: Heavy shuffles.

    Strategy:
      - Set shuffle partition count based on data size.
        Rule of thumb: data_size / 256 MB per partition.
      - Enable Adaptive Query Execution (AQE) to auto-tune at runtime.
      - Broadcast small lookup tables to eliminate join shuffles entirely.
    """
    target = max(200, int(data_size_gb * 1024 / 256))  # MB → partition count

    spark.conf.set("spark.sql.shuffle.partitions",                    str(target))
    spark.conf.set("spark.sql.adaptive.enabled",                      "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled",   "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionNum", "10")

    print(f"[Optimization] Shuffle partitions set to {target}")


def join_with_broadcast(large_df, small_df, join_key: str):
    """
    Optimization A (continued): Broadcast small table to avoid shuffle join.
    Use when the small table is < 10 MB (default spark.sql.autoBroadcastJoinThreshold).
    """
    return large_df.join(F.broadcast(small_df), join_key)


def handle_data_skew_salting(skewed_df, small_df, join_key: str, salt_factor: int = 20):
    """
    Optimization B: Data skew.

    Salting technique:
      - Append a random integer (0..salt_factor) to the join key in the large table.
      - Explode the small table to have salt_factor copies of each row.
      - Join on the salted key → distributes one hot-key across salt_factor partitions.
    """
    # Salt the large (skewed) table
    salted_large = skewed_df.withColumn(
        "_salted_key",
        F.concat(
            F.col(join_key),
            F.lit("_"),
            (F.rand() * salt_factor).cast("int").cast("string"),
        ),
    )

    # Explode the small table so every salt value is represented
    salt_array  = F.array([F.lit(i) for i in range(salt_factor)])
    salted_small = (
        small_df
        .withColumn("_salt", F.explode(salt_array))
        .withColumn(
            "_salted_key",
            F.concat(F.col(join_key), F.lit("_"), F.col("_salt").cast("string")),
        )
        .drop("_salt")
    )

    result = salted_large.join(salted_small, "_salted_key").drop("_salted_key")
    return result



def fix_small_files(df, output_path: str, n_partitions: int = 120):
    """
    Optimization C: Too many small output files.

    Strategies applied:
      1. coalesce()          — reduces partitions without a full shuffle.
      2. maxRecordsPerFile   — caps file size within each partition directory.
      3. AQE coalescePartitions — merges small post-shuffle partitions automatically.

    Use repartition() (full shuffle) instead of coalesce() only if data is heavily skewed.
    """
    (
        df
        .coalesce(n_partitions)           # merge small partitions — no shuffle
        .write
        .mode("overwrite")
        .option("maxRecordsPerFile", MAX_RECORDS_PER_FILE)
        .option("compression", "snappy")
        .parquet(output_path)
    )


# ── SQL Queries ────────────────────────────────────────────────────────────────

SQL_FIND_DUPLICATE_PARTY_KEYS = """
-- Query 1: Identify duplicate party_key records in the raw layer
SELECT
    party_key,
    COUNT(*)                        AS record_count,
    MIN(source_updated_at)          AS earliest_update,
    MAX(source_updated_at)          AS latest_update,
    COUNT(DISTINCT ingested_at)     AS distinct_ingested_at_values
FROM raw.customer_deltas
WHERE batch_date = '{batch_date}'
GROUP BY party_key
HAVING COUNT(*) > 1
ORDER BY record_count DESC;
"""

SQL_RAW_VS_CURATED_COUNTS = """
-- Query 2: Compare record counts between raw staging and curated tables
WITH raw_counts AS (
    SELECT
        batch_date,
        COUNT(*)                    AS raw_total,
        COUNT(DISTINCT party_key)   AS raw_unique_keys,
        SUM(CASE WHEN is_deleted THEN 1 ELSE 0 END) AS raw_soft_deletes
    FROM raw.customer_deltas
    WHERE batch_date = '{batch_date}'
    GROUP BY batch_date
),
curated_counts AS (
    SELECT
        COUNT(*)                    AS curated_total,
        COUNT(DISTINCT party_key)   AS curated_unique_keys,
        SUM(CASE WHEN is_deleted THEN 1 ELSE 0 END) AS curated_soft_deletes
    FROM curated.customer
)
SELECT
    r.batch_date,
    r.raw_total,
    r.raw_unique_keys,
    c.curated_total,
    c.curated_unique_keys,
    -- Metrics
    r.raw_total - r.raw_unique_keys          AS raw_duplicates_eliminated,
    c.curated_total - c.curated_unique_keys  AS curated_anomalies,     -- should be 0
    r.raw_unique_keys - c.curated_unique_keys AS net_new_or_updated_keys
FROM raw_counts r
CROSS JOIN curated_counts c;
"""

SQL_FIND_REJECTED_RECORDS = """
-- Query 3: Find records rejected during ingestion (from quarantine table)
SELECT
    q.party_key,
    q.source_updated_at,
    q.reject_reason,
    q.ingested_at,
    q.batch_date,
    -- Check if the party_key was eventually accepted into curated
    CASE WHEN c.party_key IS NOT NULL THEN 'YES' ELSE 'NO' END
        AS exists_in_curated
FROM quarantine.customer_rejects q
LEFT JOIN curated.customer c
    ON q.party_key = c.party_key
WHERE q.batch_date = '{batch_date}'
ORDER BY q.reject_reason, q.party_key;

-- Summary: rejection counts by reason
SELECT
    reject_reason,
    COUNT(*)            AS rejected_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2
    )                   AS pct_of_total_rejections
FROM quarantine.customer_rejects
WHERE batch_date = '{batch_date}'
GROUP BY reject_reason
ORDER BY rejected_count DESC;
"""


def run_sql_queries(spark, batch_date: str):
    """
    Execute the three SQL queries and return DataFrames.
    Requires 'raw.customer_deltas', 'curated.customer', and
    'quarantine.customer_rejects' tables to be registered in the Spark catalog.
    """
    duplicates_df = spark.sql(SQL_FIND_DUPLICATE_PARTY_KEYS.format(batch_date=batch_date))
    recon_df      = spark.sql(SQL_RAW_VS_CURATED_COUNTS.format(batch_date=batch_date))
    rejected_df   = spark.sql(SQL_FIND_REJECTED_RECORDS.format(batch_date=batch_date))

    print("=== Duplicate party_keys ===")
    duplicates_df.show(20, truncate=False)

    print("=== Raw vs Curated Reconciliation ===")
    recon_df.show(5, truncate=False)

    print("=== Rejected Records ===")
    rejected_df.show(20, truncate=False)

    return duplicates_df, recon_df, rejected_df