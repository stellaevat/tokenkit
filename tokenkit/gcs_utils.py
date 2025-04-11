import os
from concurrent.futures import ThreadPoolExecutor

from google.cloud import storage


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(
        f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}"
    )


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(
        f"Uploaded {source_file_name} to bucket {bucket_name} as {destination_blob_name}"
    )


def is_gcs_path(path):
    return path.startswith("gs://")


def parse_gcs_path(gcs_path):
    path_parts = gcs_path[len("gs://") :].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1].rstrip("/") if len(path_parts) > 1 else ""
    return bucket_name, blob_name


def upload_file(bucket_name, source_file_path, destination_blob_name):
    """
    Uploads a single file to the specified GCS bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_path)
    print(f"Uploaded {source_file_path} to gs://{bucket_name}/{destination_blob_name}")


def upload_directory_to_gcs(bucket_name, source_directory, target_directory=""):
    """
    Uploads all files from a local directory to the specified GCS bucket,
    optionally under a target directory.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_directory (str): Path to the local directory to upload.
        target_directory (str): Optional target directory in the GCS bucket.
    """
    executor = ThreadPoolExecutor()

    for root, _, files in os.walk(source_directory):
        for file in files:
            local_path = os.path.join(root, file)
            # Define the destination path in the bucket
            relative_path = os.path.relpath(local_path, source_directory)
            destination_path = os.path.join(target_directory, relative_path).replace(
                "\\", "/"
            )  # Ensure GCS path format
            # upload_file(bucket_name, local_path, destination_path)
            executor.submit(upload_file, bucket_name, local_path, destination_path)

    return executor
