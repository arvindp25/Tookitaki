# =============================================================================
# CUSTOMER DATA INGESTION — DE-DUPLICATION, LARGE FILE HANDLING, VALIDATION
#
# Scenario 1 — Idempotent ingestion with de-duplication
# Scenario 2 — Efficient 60 GB CSV ingestion with checkpointing
# Scenario 3 — Hard and soft data validation before ingestion
# =============================================================================

import datetime
from typing import List, Tuple

from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, TimestampType, BooleanType, DateType,
)


# =============================================================================
# SHARED CONSTANTS
# =============================================================================

ALLOWED_COUNTRIES = [
    "SG", "MY", "TH", "PH", "ID", "VN", "MM", "KH", "LA", "BN",
]

TARGET_PARTITIONS    = 240          # ~256 MB per partition for a 60 GB file
MAX_RECORDS_PER_FILE = 500_000      # prevents giant files in skewed country partitions
CHECKPOINT_PATH_TPL  = "s3://bucket/checkpoints/customer/{batch_date}.done"

HARD_REJECT_THRESHOLD    = 0.05     # block batch if >5% rows are bad
SOFT_COUNTRY_THRESHOLD   = 0.01     # alert if >1% rows have an unrecognised country
SOFT_NAME_NULL_THRESHOLD = 0.10     # alert if >10% of name values are null


# =============================================================================
# SHARED HELPERS
# =============================================================================

def get_customer_schema() -> StructType:
    return StructType([
        StructField("party_key",         StringType(),    nullable=False),
        StructField("source_updated_at", TimestampType(), nullable=False),
        StructField("name",              StringType(),    nullable=True),
        StructField("dob",               DateType(),      nullable=True),
        StructField("country",           StringType(),    nullable=True),
        StructField("is_deleted",        BooleanType(),   nullable=False),
        StructField("ingested_at",       TimestampType(), nullable=True),
    ])


def read_csv(spark, path: str):
    # PERMISSIVE keeps bad rows alive — they land in _corrupt_record for quarantine
    return (
        spark.read
        .schema(get_customer_schema())
        .option("header",                    "true")
        .option("mode",                      "PERMISSIVE")
        .option("columnNameOfCorruptRecord", "_corrupt_record")
        .option("dateFormat",                "yyyy-MM-dd")
        .option("timestampFormat",           "yyyy-MM-dd'T'HH:mm:ss")
        .csv(path)
    )


def stamp_ingested_at(df):
    # Keep source value if present; fall back to now()
    return df.withColumn(
        "ingested_at",
        F.coalesce(F.col("ingested_at"), F.current_timestamp()),
    )


def deduplicate(df):
    # Latest record per party_key: source_updated_at first, ingested_at as tie-breaker
    window = (
        Window
        .partitionBy("party_key")
        .orderBy(
            F.col("source_updated_at").desc(),
            F.col("ingested_at").desc(),
        )
    )
    return (
        df
        .withColumn("_rank", F.dense_rank().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


# =============================================================================
# SCENARIO 1 — IDEMPOTENT INGESTION
# =============================================================================

def run_idempotent_ingestion(spark, raw_path: str, curated_path: str, batch_date: str):
    # 1. Raw deltas
    raw_df = stamp_ingested_at(read_csv(spark, raw_path))

    # 2. Existing curated records (empty DataFrame on first run)
    try:
        existing_df = spark.read.parquet(curated_path)
    except Exception:
        existing_df = spark.createDataFrame([], get_customer_schema())

    # 3. Union + deduplicate across all history — latest record always wins
    final_df = deduplicate(existing_df.unionByName(raw_df, allowMissingColumns=True))

    # 4. overwrite — safe to re-run, partial writes from failed runs are replaced
    (
        final_df
        .write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("country")
        .parquet(curated_path)
    )

    active_df  = final_df.filter(F.col("is_deleted") == False)   # noqa: E712
    deleted_df = final_df.filter(F.col("is_deleted") == True)    # noqa: E712

    print(f"[Ingestion] Total curated  : {final_df.count()}")
    print(f"[Ingestion] Active records : {active_df.count()}")
    print(f"[Ingestion] Soft-deleted   : {deleted_df.count()}")

    return final_df, active_df, deleted_df


# =============================================================================
# SCENARIO 2 — LARGE FILE INGESTION (60 GB CSV)
# =============================================================================

def ingest_large_csv(spark, input_path: str, staging_path: str, batch_date: str):
    enriched_df = (
        stamp_ingested_at(read_csv(spark, input_path))
        .withColumn("batch_date",  F.lit(batch_date))
        .withColumn("source_file", F.input_file_name())
    )

    (
        enriched_df
        .repartition(TARGET_PARTITIONS, "country")   # even distribution, ~256 MB each
        .write
        .mode("overwrite")
        .partitionBy("country", "batch_date")
        .option("maxRecordsPerFile", MAX_RECORDS_PER_FILE)  # caps skewed country files
        .option("compression", "snappy")
        .parquet(staging_path)
    )

    return enriched_df


def _checkpoint_exists(spark, path: str) -> bool:
    try:
        spark.read.text(path)
        return True
    except Exception:
        return False


def _write_checkpoint(spark, path: str):
    marker_df = spark.createDataFrame(
        [(datetime.datetime.utcnow().isoformat(),)], ["completed_at"]
    )
    marker_df.write.mode("overwrite").text(path)


def ingest_with_checkpoint(spark, input_path: str, staging_path: str, batch_date: str):
    # Marker is written only after a successful write, so re-runs are safe
    marker = CHECKPOINT_PATH_TPL.format(batch_date=batch_date)

    if _checkpoint_exists(spark, marker):
        print(f"[Checkpoint] Batch {batch_date} already ingested. Skipping.")
        return

    print(f"[Checkpoint] Processing batch {batch_date} ...")
    ingest_large_csv(spark, input_path, staging_path, batch_date)
    _write_checkpoint(spark, marker)
    print(f"[Checkpoint] Batch {batch_date} complete.")


# =============================================================================
# SCENARIO 3 — VALIDATION
# =============================================================================

def _emit_alert(code: str, message: str):
    # Stub — swap in PagerDuty / SNS in production
    print(f"[ALERT][{code}] {message}")


def run_hard_validations(df) -> Tuple:
    # Raises RuntimeError if reject rate exceeds HARD_REJECT_THRESHOLD (5%)
    # Hard rules: non-null party_key, valid non-future source_updated_at, non-null is_deleted
    total = df.count()

    null_key_df    = df.filter(F.col("party_key").isNull())
    invalid_ts_df  = df.filter(
        F.col("source_updated_at").isNull()
        | (F.col("source_updated_at") > F.current_timestamp())
        | (F.col("source_updated_at") < F.lit("2000-01-01").cast("timestamp"))
    )
    invalid_del_df = df.filter(F.col("is_deleted").isNull())

    quarantine_df = (
        null_key_df.withColumn("reject_reason", F.lit("NULL_PARTY_KEY"))
        .unionByName(invalid_ts_df.withColumn("reject_reason", F.lit("INVALID_SOURCE_UPDATED_AT")))
        .unionByName(invalid_del_df.withColumn("reject_reason", F.lit("NULL_IS_DELETED")))
        .distinct()
    )

    bad_count = quarantine_df.count()
    bad_pct   = bad_count / total if total > 0 else 0.0

    if bad_pct > HARD_REJECT_THRESHOLD:
        raise RuntimeError(
            f"[Hard Validation] BATCH BLOCKED — {bad_pct:.1%} of records rejected "
            f"(threshold {HARD_REJECT_THRESHOLD:.0%}). "
            f"Bad rows: {bad_count} / {total}."
        )

    bad_keys = quarantine_df.select("party_key").distinct()
    clean_df = df.join(bad_keys, on="party_key", how="left_anti")

    print(f"[Hard Validation] Passed: {clean_df.count()}, Rejected: {bad_count}")
    return clean_df, quarantine_df


def run_soft_validations(df, allowed_countries: List[str] = ALLOWED_COUNTRIES):
    # Non-blocking — flags rows and emits alerts, never stops ingestion
    total = df.count()

    invalid_country_count = df.filter(
        ~F.col("country").isin(allowed_countries) | F.col("country").isNull()
    ).count()
    invalid_country_pct = invalid_country_count / total if total > 0 else 0.0

    if invalid_country_pct > SOFT_COUNTRY_THRESHOLD:
        _emit_alert(
            "SOFT_COUNTRY_ANOMALY",
            f"{invalid_country_pct:.2%} of records have an unrecognised country "
            f"({invalid_country_count} rows). Threshold: {SOFT_COUNTRY_THRESHOLD:.0%}.",
        )

    name_null_count = df.filter(F.col("name").isNull()).count()
    name_null_pct   = name_null_count / total if total > 0 else 0.0

    if name_null_pct > SOFT_NAME_NULL_THRESHOLD:
        _emit_alert(
            "SOFT_NAME_NULL_HIGH",
            f"name is null in {name_null_pct:.2%} of records "
            f"({name_null_count} rows). Threshold: {SOFT_NAME_NULL_THRESHOLD:.0%}.",
        )

    flagged_df = df.withColumn(
        "quality_flag",
        F.when(
            ~F.col("country").isin(allowed_countries) | F.col("country").isNull(),
            F.lit("UNKNOWN_COUNTRY"),
        ).when(
            F.col("name").isNull(),
            F.lit("MISSING_NAME"),
        ).otherwise(F.lit(None).cast("string")),
    )

    return flagged_df


def run_all_validations(df, quarantine_path: str):
    # Hard validation may raise RuntimeError — caller handles pipeline failure
    clean_df, quarantine_df = run_hard_validations(df)

    if quarantine_df.count() > 0:
        (
            quarantine_df
            .write
            .mode("append")
            .partitionBy("reject_reason")
            .parquet(quarantine_path)
        )

    flagged_df = run_soft_validations(clean_df)
    return flagged_df, quarantine_df


# =============================================================================
# DESIGN NOTES
# =============================================================================
#
# Shared helpers:
#   get_customer_schema()  — single schema definition; one place to update field types.
#   read_csv()             — central home for Spark read options (mode, formats, etc.).
#   stamp_ingested_at()    — was copy-pasted across S1 and S2; extracted for consistency.
#   deduplicate()          — dense_rank window was duplicated; now one implementation.
#
# Scenario 1 — Idempotent ingestion:
#   Union raw deltas with full curated history, then re-deduplicate — the latest
#   record always wins regardless of run order or retries. Atomic overwrite means
#   no partial-write window is visible to readers.
#
# Scenario 2 — 60 GB CSV:
#   Explicit schema skips type-inference cost. repartition(240, country) avoids
#   the small-files problem. maxRecordsPerFile handles skewed countries.
#   Parquet + Snappy: splittable, columnar, 5-7x compression.
#
# Scenario 3 — Validation:
#   Hard rules protect integrity (bad primary key, future timestamp = untrustworthy).
#   Soft rules protect quality (unknown country, missing name = worth investigating).
#   The 5% circuit breaker catches upstream feed failures before they corrupt the
#   curated table.  table retains all rejected rows for audit.


