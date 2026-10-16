"""Immutable, verified ODE checkpoints; only rank zero publishes and ``root`` must be shared."""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Callable
import uuid


logger = logging.getLogger(__name__)
COMPLETED = "COMPLETED.json"
WEIGHTS = "diffusion_pytorch_model.safetensors"
WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
_STEP = re.compile(r"step_(\d{8,})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _positive_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def _relative(name: str) -> PurePosixPath:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError(f"Unsafe checkpoint relative path: {name!r}")
    return PurePosixPath(name)


def config_differences(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    """Keys whose values differ between two model configurations, ignoring private ``_`` keys."""
    keys = {key for key in first.keys() | second.keys() if not key.startswith("_")}
    return sorted(key for key in keys if key not in first or key not in second or first[key] != second[key])


def check_shard_index(directory: Path, has_shard: Callable[[str], bool]) -> None:
    """Require the weight index in ``directory`` to name only safe shards that ``has_shard`` accepts."""
    index = json.loads((directory / WEIGHTS_INDEX).read_text())
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"Shard index has no weight_map: {directory}")
    for shard in weights.values():
        try:
            name = str(_relative(shard))
        except ValueError:
            name = None
        if name is None or not name.endswith(".safetensors") or not has_shard(name):
            raise ValueError(f"Shard index in {directory} references missing/invalid shard: {shard!r}")


def check_model_dir(path: Path) -> dict[str, Any]:
    """Return a saved model's configuration after checking that all of its weight files exist."""
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Invalid model configuration: {path}")
    if (path / WEIGHTS_INDEX).exists():
        check_shard_index(path, lambda name: (path / name).is_file())
    elif not (path / WEIGHTS).is_file():
        raise ValueError(f"Missing model weights: {path}")
    return config


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _files(path: Path) -> dict[str, Path]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Checkpoint is not a regular directory: {path}")
    files = {}
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Checkpoint contains a symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"Checkpoint contains a nonregular file: {item}")
        name = item.relative_to(path).as_posix()
        _relative(name)
        if name != COMPLETED:
            files[name] = item
    return files


def _manifest(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("Unsupported or missing checkpoint manifest version")
    _positive_int(data.get("global_step"), "global_step")
    world_size = _positive_int(data.get("world_size"), "world_size", 1)
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Checkpoint manifest has no payload files")
    for name, meta in files.items():
        _relative(name)
        if name == COMPLETED or not isinstance(meta, dict):
            raise ValueError(f"Invalid manifest entry: {name}")
        _positive_int(meta.get("size"), f"size of {name}")
        if not isinstance(meta.get("sha256"), str) or not _SHA256.fullmatch(meta["sha256"]):
            raise ValueError(f"Missing or invalid SHA256 for {name}")
    required = {"model/config.json", "optimizer/.metadata", "training_state.pt"}
    if not required.issubset(files):
        raise ValueError(f"Missing required checkpoint files: {sorted(required - files.keys())}")
    if not any(name.startswith("optimizer/") and name != "optimizer/.metadata" for name in files):
        raise ValueError("Checkpoint has optimizer metadata but no optimizer shards")
    if not ({f"model/{WEIGHTS}", f"model/{WEIGHTS_INDEX}"} & files.keys()):
        raise ValueError("Checkpoint has no model weights or model shard index")
    ranks = [int(match.group(1)) for name in files
             if (match := re.fullmatch(r"rng/rank_(\d+)\.pt", name))]
    if len(ranks) != world_size or set(ranks) != set(range(world_size)):
        raise ValueError(f"Checkpoint RNG files must cover all {world_size} ranks exactly once")
    return data


class ODECheckpointStore:
    def __init__(self, root: Path, s3_client: Any = None, bucket: str | None = None,
                 s3_prefix: str | None = None):
        self.root = Path(root)
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = (s3_prefix or "").strip("/")
        self._remote_legacy = False
        if self.s3 is not None and not self.bucket:
            raise ValueError("An S3 bucket is required with s3_client")
        if self.s3 is None and (bucket is not None or s3_prefix is not None):
            raise ValueError("An injected s3_client is required for remote checkpoints")

    @staticmethod
    def _name(step: int) -> str:
        return f"step_{_positive_int(step, 'global_step'):08d}"

    def _key(self, name: str, relative: str = "") -> str:
        return "/".join(part for part in (self.prefix, name, relative) if part)

    def begin(self, step: int) -> Path:
        name = self._name(step)
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".incomplete-{name}-{uuid.uuid4().hex}"
        staging.mkdir()
        return staging

    @staticmethod
    def _check_payload(path: Path, step: int, world_size: int) -> None:
        """Hook for trainer-specific payload checks on commit and validation."""

    def commit(self, staging: Path, step: int, world_size: int) -> Path:
        staging = Path(staging)
        self._check_payload(staging, step, world_size)
        name = self._name(step)
        if (staging.parent.resolve() != self.root.resolve()
                or not staging.name.startswith(f".incomplete-{name}-") or staging.is_symlink()):
            raise ValueError("Staging directory must come from begin(step) in this store")
        destination = self.root / name
        if destination.exists():
            raise FileExistsError(f"Checkpoint is immutable: {destination}")
        if (staging / COMPLETED).exists():
            raise ValueError("Staging directory already has a completion marker")
        files = {name: _fingerprint(path) for name, path in sorted(_files(staging).items())}
        manifest = _manifest({"version": 1, "global_step": step,
                              "world_size": world_size, "files": files})
        self._check_model_index(staging, manifest)
        self._write_manifest(staging, manifest)
        staging.rename(destination)
        return destination

    @staticmethod
    def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
        with (path / COMPLETED).open("x") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _check_model_index(path: Path, manifest: dict[str, Any]) -> None:
        if f"model/{WEIGHTS_INDEX}" in manifest["files"]:
            check_shard_index(path / "model", lambda name: f"model/{name}" in manifest["files"])

    def validate(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        files = _files(path)
        manifest = _manifest(json.loads((path / COMPLETED).read_text()))
        if _STEP.fullmatch(path.name) and path.name != self._name(manifest["global_step"]):
            raise ValueError("Checkpoint directory step differs from its manifest")
        if files.keys() != manifest["files"].keys():
            raise ValueError("Checkpoint payload files differ from completion manifest")
        for name, payload in files.items():
            if _fingerprint(payload) != manifest["files"][name]:
                raise ValueError(f"Checkpoint checksum/size mismatch: {name}")
        self._check_model_index(path, manifest)
        self._check_payload(path, manifest["global_step"], manifest["world_size"])
        return manifest

    def _remote_names(self) -> set[str]:
        names = set()
        self._remote_legacy = False
        prefix = self._key("") + ("/" if self.prefix else "")
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.startswith(prefix):
                    continue
                parts = key[len(prefix):].split("/")
                if parts == ["checkpoint.pth"]:
                    self._remote_legacy = True
                if len(parts) == 2 and _STEP.fullmatch(parts[0]) and parts[1] == COMPLETED:
                    names.add(parts[0])
        return names

    def _remote_manifest(self, name: str) -> dict[str, Any]:
        body = self.s3.get_object(Bucket=self.bucket, Key=self._key(name, COMPLETED))["Body"]
        try:
            data = body.read(16 * 1024 * 1024 + 1)
        finally:
            body.close()
        if len(data) > 16 * 1024 * 1024:
            raise ValueError("Checkpoint completion manifest is too large")
        manifest = _manifest(json.loads(data))
        if name != self._name(manifest["global_step"]):
            raise ValueError("Remote checkpoint name and manifest step differ")
        return manifest

    def _download(self, name: str) -> Path:
        manifest = self._remote_manifest(name)
        staging = self.begin(manifest["global_step"])
        for relative in manifest["files"]:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, self._key(name, relative), str(target))
        self._write_manifest(staging, manifest)
        self.validate(staging)
        destination = self.root / name
        if destination.exists():
            # Preserve the invalid local copy when restoring a verified remote copy.
            destination.rename(self.root / f".corrupt-{name}-{uuid.uuid4().hex}")
        staging.rename(destination)
        return destination

    def latest(self) -> Path | None:
        local = {p.name for p in self.root.iterdir() if _STEP.fullmatch(p.name)} if self.root.exists() else set()
        remote = set()
        listing_error = None
        if self.s3 is not None:
            try:
                remote = self._remote_names()
            except Exception as error:
                listing_error = error
                logger.warning("Cannot list remote ODE checkpoints; newest remote step is unknown: %s", error)
        candidates = sorted(local | remote, key=lambda n: int(_STEP.fullmatch(n).group(1)), reverse=True)
        for name in candidates:
            if name in local:
                try:
                    self.validate(self.root / name)
                    return self.root / name
                except (OSError, ValueError) as error:
                    logger.warning("Ignoring invalid local ODE checkpoint %s: %s", name, error)
            if name in remote:
                try:
                    return self._download(name)
                except Exception as error:
                    code = getattr(error, "response", {}).get("Error", {}).get("Code")
                    if code not in ("404", "NoSuchKey", "NotFound") and not isinstance(error, (OSError, ValueError)):
                        raise
                    logger.warning("Ignoring invalid/unavailable remote ODE checkpoint %s: %s", name, error)
        if listing_error is not None:
            raise RuntimeError("Remote checkpoint discovery failed and no valid local checkpoint exists") from listing_error
        if candidates:
            raise ValueError("Completed checkpoint candidates exist, but none passed validation")
        if self._remote_legacy:
            raise ValueError("Legacy remote checkpoint.pth exists without a verified resume checkpoint")
        return None

    def upload(self, path: Path) -> None:
        if self.s3 is None:
            raise ValueError("No S3 client configured")
        path = Path(path)
        manifest = self.validate(path)
        name = self._name(manifest["global_step"])
        try:
            existing = self._remote_manifest(name)
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if code not in ("404", "NoSuchKey", "NotFound"):
                raise
        else:
            if existing != manifest:
                raise FileExistsError(f"Remote checkpoint is immutable: {self._key(name)}")
            return
        for relative in sorted(manifest["files"]):
            self.s3.upload_file(
                str(path / relative), self.bucket, self._key(name, relative),
                ExtraArgs={"Metadata": {"sha256": manifest["files"][relative]["sha256"]}})
        self.s3.put_object(Bucket=self.bucket, Key=self._key(name, COMPLETED),
                           Body=json.dumps(manifest, sort_keys=True).encode(), ContentType="application/json")

    def _verify_remote_payload(self, name: str, manifest: dict[str, Any]) -> None:
        """Check every object before deleting the only local checkpoint copy."""
        for relative, expected in manifest["files"].items():
            head = self.s3.head_object(Bucket=self.bucket, Key=self._key(name, relative),
                                       ChecksumMode="ENABLED")
            if head.get("ContentLength") != expected["size"]:
                raise ValueError(f"S3 checkpoint size mismatch: {relative}")
            sha256 = head.get("Metadata", {}).get("sha256")
            if sha256 is not None and sha256 != expected["sha256"]:
                raise ValueError(f"S3 checkpoint SHA256 metadata mismatch: {relative}")
            # Multipart composite checksums hash part checksums, not the file.
            checksum = head.get("ChecksumSHA256")
            if checksum is not None and head.get("ChecksumType") != "COMPOSITE":
                expected_checksum = base64.b64encode(bytes.fromhex(expected["sha256"])).decode("ascii")
                if checksum != expected_checksum:
                    raise ValueError(f"S3 checkpoint SHA256 mismatch: {relative}")

    def prune_local(self, keep: int) -> list[Path]:
        """Drop local copies beyond the newest ``keep`` once S3 holds a verified identical copy."""
        if self.s3 is None:
            raise ValueError("Pruning local checkpoints requires an S3 store")
        _positive_int(keep, "keep")
        if not self.root.exists():
            return []
        published = sorted((p for p in self.root.iterdir() if _STEP.fullmatch(p.name)),
                           key=lambda p: int(_STEP.fullmatch(p.name).group(1)), reverse=True)
        removed = []
        for path in published[keep:]:
            try:
                manifest = _manifest(json.loads((path / COMPLETED).read_text()))
                if self._remote_manifest(path.name) != manifest:
                    raise ValueError("S3 copy differs from the local manifest")
                self._verify_remote_payload(path.name, manifest)
            except Exception as error:
                logger.warning("Keeping local checkpoint %s: cannot confirm its S3 copy: %s", path, error)
                continue
            shutil.rmtree(path)
            removed.append(path)
        return removed
