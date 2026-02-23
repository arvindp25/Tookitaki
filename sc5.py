from datetime import timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
    from airflow.providers.slack.operators.slack import SlackAPIPostOperator
    from airflow.utils.dates import days_ago
    _AIRFLOW_AVAILABLE = True
except ImportError:
    _AIRFLOW_AVAILABLE = False


DEFAULT_ARGS = {
    "owner":                     "data-engineering",
    "depends_on_past":           False,
    "email_on_failure":          True,
    "email":                     ["dxyz@company.com"],
    "retries":                   3,
    "retry_delay":               timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay":           timedelta(minutes=30),
}


def _submit_spark_job(**context):
    batch_date   = context["ds"]
    input_path   = f"s3://raw-bucket/customer/{batch_date}/"
    staging_path = "s3://staging/customer/"
    curated_path = "s3://curated/customer/"

    spark = SparkSession.builder.appName("CustomerIngestion").getOrCreate()
    ingest_with_checkpoint(spark, input_path, staging_path, batch_date)
    run_idempotent_ingestion(spark, staging_path, curated_path, batch_date)


def _generate_recon_metrics(**context):
    batch_date = context["ds"]
    spark = SparkSession.builder.appName("Reconciliation").getOrCreate()

    raw_count     = spark.read.parquet(f"s3://staging/customer/batch_date={batch_date}").count()
    curated_count = spark.read.parquet("s3://curated/customer/").count()

    print(f"[Recon] Raw records   : {raw_count}")
    print(f"[Recon] Curated total : {curated_count}")

    spark.createDataFrame(
        [(batch_date, raw_count, curated_count)],
        ["batch_date", "raw_count", "curated_count"],
    ).write.mode("append").parquet("s3://metrics/customer_recon/")


if _AIRFLOW_AVAILABLE:
    with DAG(
        dag_id            = "customer_ingestion_pipeline",
        default_args      = DEFAULT_ARGS,
        schedule_interval = "0 6 * * *",
        start_date        = days_ago(1),
        catchup           = False,
        tags              = ["ingestion", "customer"],
        max_active_runs   = 1,
        doc_md            = __doc__,
    ) as dag:

        detect_file = S3KeySensor(
            task_id        = "detect_new_files",
            bucket_name    = "{{ var.value.raw_bucket }}",
            bucket_key     = "customer/{{ ds }}/part-*.csv",
            wildcard_match = True,
            timeout        = 3600,
            poke_interval  = 300,
            mode           = "reschedule",
        )

        validate = PythonOperator(
            task_id         = "run_ge_validation",
            python_callable = _run_ge_validation,
            retries         = 1,
        )

        ingest = PythonOperator(
            task_id         = "run_spark_ingestion",
            python_callable = _submit_spark_job,
        )

        recon = PythonOperator(
            task_id         = "generate_recon_metrics",
            python_callable = _generate_recon_metrics,
            retries         = 2,
        )

        notify_success = SlackAPIPostOperator(
            task_id      = "notify_success",
            channel      = "#data-pipelines",
            text         = ":white_check_mark: Customer ingestion {{ ds }} succeeded.",
            trigger_rule = "all_success",
        )

        notify_failure = SlackAPIPostOperator(
            task_id      = "notify_failure",
            channel      = "#data-alerts",
            text         = ":red_circle: Customer ingestion {{ ds }} FAILED. Check Airflow.",
            trigger_rule = "one_failed",
        )

        detect_file >> validate >> ingest >> recon >> [notify_success, notify_failure]