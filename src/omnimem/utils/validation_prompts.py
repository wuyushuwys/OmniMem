"""Explicit multi-segment prompts for causal, distributed video evaluation."""
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path


def validate_prompt_schedules(records, *, num_frames, frame_per_block):
    """Normalize schedules; every record must share the same latent-frame switch positions."""
    if type(num_frames) is not int or num_frames <= 0:
        raise ValueError("Scheduled validation needs a positive integer num_frames")
    if type(frame_per_block) is not int or frame_per_block <= 0:
        raise ValueError("Scheduled validation needs a positive integer frame_per_block")
    latent_frames = (num_frames - 1) // 4 + 1
    if latent_frames % frame_per_block:
        raise ValueError("Scheduled validation must generate complete latent AR blocks")
    normalized = []
    seen_ids = set()
    shared_positions = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Prompt schedule {index} must be a JSON object")
        name = record.get("id", f"prompt-{index:03d}")
        if not isinstance(name, str) or not name.strip() or name in seen_ids:
            raise ValueError(f"Prompt schedule {index} needs a unique nonempty id")
        seen_ids.add(name)
        texts = record.get("prompts")
        positions = record.get("switch_frame_indices")
        if not isinstance(texts, Sequence) or isinstance(texts, str) or not len(texts):
            raise ValueError(f"Prompt schedule {name} needs a nonempty prompts list")
        if any(not isinstance(p, str) or not p.strip() for p in texts):
            raise ValueError(f"Prompt schedule {name} contains an empty/non-string prompt")
        if not isinstance(positions, Sequence) or isinstance(positions, str):
            raise ValueError(f"Prompt schedule {name} needs explicit switch_frame_indices")
        positions = list(positions)
        if len(positions) != len(texts) - 1:
            raise ValueError(f"Prompt schedule {name}: one switch index is required per later prompt")
        if any(type(p) is not int or p <= 0 or p >= latent_frames for p in positions):
            raise ValueError(f"Prompt schedule {name}: switch indices must be inside the latent video")
        if positions != sorted(set(positions)) or any(p % frame_per_block for p in positions):
            raise ValueError(f"Prompt schedule {name}: switch indices must increase and align to AR blocks")
        if shared_positions is not None and positions != shared_positions:
            raise ValueError("All validation prompts must share the same switch schedule for FSDP")
        shared_positions = positions
        normalized.append({"id": name, "prompts": [p.strip() for p in texts],
                           "switch_frame_indices": positions})
    if not normalized:
        raise ValueError("Prompt schedule file contains no prompts")
    return normalized


def prepare_validation_prompts(validation_data, *, frame_per_block=None):
    """Load configured prompt files into the trainer's serializable config."""
    validation_export_frames(validation_data)
    schedule_path = validation_data.get("prompt_schedule_file")
    if schedule_path:
        if validation_data.get("prompt_file"):
            raise ValueError("Set prompt_schedule_file or prompt_file, not both")
        source = Path(schedule_path)
        raw = source.read_bytes()
        records = []
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid prompt JSON") from exc
        records = validate_prompt_schedules(
            records, num_frames=validation_data.get("num_frames"),
            frame_per_block=frame_per_block)
        validation_data["prompt_schedules"] = records
        validation_data["prompts"] = [record["prompts"][0] for record in records]
        validation_data["prompt_schedule_source"] = {
            "path": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
            "record_ids": [record["id"] for record in records],
        }
    elif "prompt_file" in validation_data:
        files = validation_data["prompt_file"]
        if isinstance(files, (str, Path)):
            files = [files]
        prompts = []
        for filename in files:
            if Path(filename).suffix.lower() in {".json", ".jsonl"}:
                raise ValueError("Structured prompts require prompt_schedule_file, not prompt_file")
            prompts.extend(p.strip() for p in Path(filename).read_text(encoding="utf-8").splitlines()
                           if p.strip())
        validation_data["prompts"] = prompts


def evaluation_prompt_records(validation_data, *, frame_per_block=None):
    """Return normalized inference records (a plain TXT prompt is a one-prompt schedule)."""
    if validation_data.get("prompt_schedule_file") and not validation_data.get("prompt_schedules"):
        prepare_validation_prompts(validation_data, frame_per_block=frame_per_block)
    if validation_data.get("prompt_schedules"):
        return validate_prompt_schedules(
            validation_data["prompt_schedules"], num_frames=validation_data.get("num_frames"),
            frame_per_block=frame_per_block)
    return [{"id": f"prompt-{i:03d}", "prompts": [p], "switch_frame_indices": []}
            for i, p in enumerate(validation_data["prompts"])]


def validation_export_frames(validation_data):
    """Optional tail-only crop, e.g. 969 generated RGB frames -> 960 at 16 fps."""
    count = validation_data.get("export_num_frames")
    if count is None:
        return None
    frames = validation_data.get("num_frames")
    if type(frames) is not int or frames <= 0:
        raise ValueError("export_num_frames requires positive integer num_frames")
    decoded_frames = 4 * ((frames - 1) // 4) + 1
    if type(count) is not int or not 0 < count <= decoded_frames:
        raise ValueError("export_num_frames must be positive and no greater than the decoded frame count")
    return count
