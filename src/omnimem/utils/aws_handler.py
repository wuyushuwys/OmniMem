import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import boto3
from huggingface_hub import HfApi
from omnimem.utils.logging_tool import get_logger


class S3:

    def __init__(self, bucket, subdir, ignored_path=None, enable=True):
        self.bucket = bucket
        self.subdir = subdir
        self.ignored_path = ignored_path

        # Determine if S3 operations should be active
        self._enabled = enable
        if self.bucket is not None and self.subdir is not None:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None

    def _normalize_target_path(self, target_path):
        if self.ignored_path is not None:
            target_path = target_path.replace(self.ignored_path, "").lstrip("/")
        return target_path

    def upload_folder(self, folder_path, global_step=None):
        """Upload a directory to S3, or make a local backup copy when S3 is not configured.

        folder_path: local directory to upload / back up.
        global_step: step number appended to the folder name.
        """
        if not self._enabled:
            return
        assert os.path.isdir(folder_path), f"{folder_path} is not a directory"
        logger = get_logger()

        # no S3 configured -> local backup copy suffixed with the step
        if self.s3_client is None:
            if global_step is None:
                logger.info("S3 disabled and no global_step provided; skip local folder copy.")
                return
            target_path = f"{folder_path}_{global_step}"
            try:
                shutil.copytree(folder_path, target_path, dirs_exist_ok=True)
                logger.info(f"Copied folder locally to {target_path}")
            except Exception as e:
                logger.error(f"Failed local folder copy {folder_path} -> {target_path}: {e}")
            return

        # s3 upload path
        target_path = f"{folder_path}_{global_step}" if global_step is not None else folder_path
        target_path = self._normalize_target_path(target_path)
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                local_path = os.path.join(root, filename)
                # maintain relative path structure for S3 keys
                relative_path = os.path.relpath(local_path, folder_path)
                s3_path = os.path.join(self.subdir, target_path, relative_path)
                try:
                    self.s3_client.upload_file(local_path, self.bucket, s3_path)
                    logger.info(f"Upload file to {s3_path}")
                except Exception as e:
                    logger.error(f"Failed to upload {local_path}: {e}")

    def upload(self, fpath, global_step=None, step_folder=False):
        if not self._enabled:
            return
        logger = get_logger()

        if global_step is not None:
            if step_folder:
                folder_name, fname = os.path.split(fpath)
                target_path = os.path.join(folder_name + f"_{global_step}", fname)
            else:
                fname, ext = os.path.splitext(fpath)
                target_path = fname + f"_{global_step}" + ext
        else:
            target_path = fpath

        target_path = self._normalize_target_path(target_path)

        if self.s3_client is None:
            # If no global_step and target_path == fpath, this would copy onto itself.
            if os.path.abspath(target_path) == os.path.abspath(fpath):
                logger.info("S3 disabled and no global_step provided; skip local file copy.")
                return

            try:
                os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
                shutil.copy2(fpath, target_path)
                logger.info(f"Copied file locally to {target_path}")
            except Exception as e:
                logger.error(f"Failed local file copy {fpath} -> {target_path}: {e}")
            return

        # s3 upload path
        try:
            self.s3_client.upload_file(fpath, self.bucket, os.path.join(self.subdir, target_path))
            logger.info(f"Upload file to {os.path.join(self.subdir, target_path)}")
        except Exception as e:
            logger.error(e)

    def close(self):
        """Closes the underlying boto3 client connection pool."""
        if self._enabled and self.s3_client is not None:
            self.s3_client.close()

    def __enter__(self):
        """Allows the S3 class to be used in a 'with' block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncS3UploadAndDeleteHook:
    def __init__(self, bucket_name: str, s3_prefix: str, max_concurrent_uploads: int = 2):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_uploads)
        self.logger = get_logger()

    def __call__(self, local_tar_path: str):
        self.executor.submit(self._upload_and_delete, local_tar_path)

    def _upload_and_delete(self, local_tar_path: str):
        file_name = os.path.basename(local_tar_path)
        s3_key = os.path.join(self.s3_prefix, file_name).replace("\\", "/")
        try:
            self.s3_client.upload_file(local_tar_path, self.bucket_name, s3_key)
            os.remove(local_tar_path)
        except Exception as e:
            self.logger.error(e, exc_info=True)

    def wait_and_close(self):
        self.executor.shutdown(wait=True)


class AsyncHFUploadAndDeleteHook:
    def __init__(self, repo_id: str, token: str, path_in_repo: str = "", max_concurrent_uploads: int = 2,
                 repo_type: str = "dataset"):
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo
        self.repo_type = repo_type
        self.api = HfApi(token=token)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_uploads)
        self.logger = get_logger()

    def __call__(self, local_tar_path: str):
        self.executor.submit(self._upload_and_delete, local_tar_path)

    def _upload_and_delete(self, local_tar_path: str):
        file_name = os.path.basename(local_tar_path)
        target_path = os.path.join(self.path_in_repo, file_name).replace("\\", "/") if self.path_in_repo else file_name
        try:
            self.api.upload_file(
                path_or_fileobj=local_tar_path,
                path_in_repo=target_path,
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            )
            os.remove(local_tar_path)
            self.logger.info(f"Uploaded and deleted: {local_tar_path}")
        except Exception as e:
            self.logger.error(f"Failed to upload {local_tar_path}: {e}", exc_info=True)

    def wait_and_close(self):
        self.executor.shutdown(wait=True)