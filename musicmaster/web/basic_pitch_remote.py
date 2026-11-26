"""Remote Basic Pitch CPU transcription through the Colab worker job API.

The website never loads TensorFlow. It submits audio to the lightweight broker,
waits for the Colab CPU worker, then converts returned polyphonic events into
MusicMaster's local score artifacts.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


def _api_config() -> tuple[str, dict[str, str]]:
    base = os.getenv("BASIC_PITCH_API_URL", "").strip().rstrip("/")
    api_key = os.getenv("BASIC_PITCH_API_KEY", "").strip()
    if not base:
        raise RuntimeError("尚未配置 BASIC_PITCH_API_URL，远程 Basic Pitch CPU 服务不可用。")
    if not api_key:
        raise RuntimeError("尚未配置 BASIC_PITCH_API_KEY，远程 Basic Pitch 服务鉴权未完成。")
    return base, {"X-API-Key": api_key}


def transcribe_remote(job: Any, audio_path: str, output_dir: Path) -> dict:
    """Submit audio, wait for the Colab worker, and save canonical target + MIDI."""
    base, headers = _api_config()
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"上传音频不存在：{source}")

    max_wait = max(60, int(os.getenv("BASIC_PITCH_TIMEOUT_SECONDS", "3600")))
    poll_seconds = max(2, int(os.getenv("BASIC_PITCH_POLL_SECONDS", "5")))

    job.set_stage("正在把音频安全提交到 Basic Pitch 云端 CPU…", 0.34)
    with source.open("rb") as audio:
        response = requests.post(
            f"{base}/v1/jobs",
            headers=headers,
            files={"file": (source.name, audio, "application/octet-stream")},
            timeout=(20, 180),
        )
    response.raise_for_status()
    created = response.json()
    remote_id = created.get("job_id")
    if not remote_id:
        raise RuntimeError(f"Basic Pitch 服务没有返回任务编号：{created}")

    started = time.monotonic()
    last_status = None
    while True:
        status_response = requests.get(
            f"{base}/v1/jobs/{remote_id}",
            headers=headers,
            timeout=(10, 30),
        )
        status_response.raise_for_status()
        state = status_response.json()
        status = state.get("status")

        if status != last_status:
            if status == "queued":
                job.set_stage("音频已提交，等待 Google Colab CPU Worker 领取…", 0.40)
            elif status == "processing":
                job.set_stage("Google Colab CPU 正在运行 Basic Pitch 复音识别…", 0.58)
            last_status = status

        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError(f"Basic Pitch CPU 推理失败：{state.get('error') or '未知错误'}")
        if status not in ("queued", "processing"):
            raise RuntimeError(f"Basic Pitch 服务返回未知任务状态：{status!r}")
        if time.monotonic() - started > max_wait:
            raise TimeoutError(
                f"等待 Colab CPU 计算超过 {max_wait} 秒；任务编号：{remote_id}"
            )
        time.sleep(poll_seconds)

    job.set_stage("CPU 识别完成，正在接收音符与 MIDI…", 0.78)
    result_response = requests.get(
        f"{base}/v1/jobs/{remote_id}/result",
        headers=headers,
        timeout=(10, 120),
    )
    result_response.raise_for_status()
    raw = result_response.json()
    raw_notes = raw.get("notes")
    if not isinstance(raw_notes, list):
        raise RuntimeError("Basic Pitch 结果缺少 notes 音符数组。")

    notes = []
    for item in raw_notes:
        try:
            pitch = int(item["midi"])
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= pitch <= 127 and start_s >= 0 and end_s > start_s:
            notes.append({
                "midi": pitch,
                "start_s": round(start_s, 6),
                "end_s": round(end_s, 6),
            })
    notes.sort(key=lambda item: (item["start_s"], item["midi"], item["end_s"]))
    if not notes:
        raise RuntimeError("Basic Pitch 已完成计算，但没有识别到有效音符。")

    midi_response = requests.get(
        f"{base}/v1/jobs/{remote_id}/midi",
        headers=headers,
        timeout=(10, 120),
    )
    midi_response.raise_for_status()
    if not midi_response.content:
        raise RuntimeError("Basic Pitch 返回的 MIDI 文件为空。")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "out.mid").write_bytes(midi_response.content)

    tempo_bpm = raw.get("tempo_bpm")
    try:
        tempo_bpm = float(tempo_bpm) if tempo_bpm else 120.0
    except (TypeError, ValueError):
        tempo_bpm = 120.0
    if tempo_bpm <= 0:
        tempo_bpm = 120.0

    target = {"tempo": tempo_bpm, "notes": notes}
    (output_dir / "notes.json").write_text(
        json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def target_to_polyphonic_musicxml(
    target: dict, out_path: Path, *, key_name: str = ""
) -> Path:
    """Write overlapping Basic Pitch notes as simultaneous chords/voices.

    Notes with the same quantized onset and duration become a chord. Other
    overlapping events are assigned to separate voices on the same staff.
    Exact seconds remain unchanged in notes.json and MIDI.
    """
    from music21 import chord, duration, key as m21key, meter, note, stream, tempo

    bpm = float(target.get("tempo") or 120.0)
    if bpm <= 0:
        bpm = 120.0
    seconds_per_quarter = 60.0 / bpm
    quantum = 0.5  # eighth-note notation grid

    grouped: dict[tuple[float, float], set[int]] = defaultdict(set)
    for item in target.get("notes", []):
        pitch = int(item["midi"])
        start_s = max(0.0, float(item["start_s"]))
        end_s = float(item["end_s"])
        if end_s <= start_s:
            continue
        start_ql = round((start_s / seconds_per_quarter) / quantum) * quantum
        duration_ql = round(
            ((end_s - start_s) / seconds_per_quarter) / quantum
        ) * quantum
        duration_ql = max(quantum, duration_ql)
        grouped[(start_ql, duration_ql)].add(pitch)

    events = [
        (start, dur, sorted(pitches))
        for (start, dur), pitches in grouped.items()
        if pitches
    ]
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    if not events:
        raise RuntimeError("没有可用于生成 MusicXML 的音符。")

    # Greedy interval partitioning: each voice contains no overlapping events.
    voices: list[dict[str, Any]] = []
    for start, dur, pitches in events:
        end = start + dur
        destination = next(
            (voice for voice in voices if start >= voice["end"] - 1e-9),
            None,
        )
        if destination is None:
            destination = {"end": 0.0, "events": []}
            voices.append(destination)
        destination["events"].append((start, dur, pitches))
        destination["end"] = end

    part = stream.Part()
    part.insert(0, tempo.MetronomeMark(number=int(round(bpm))))
    part.insert(0, meter.TimeSignature("4/4"))

    resolved_key = key_name.strip() if key_name else ""
    if not resolved_key:
        try:
            from musicmaster.core.transcribe import _detect_key_name
            detected = _detect_key_name(target.get("notes", []))
            if detected:
                part.insert(0, m21key.Key(detected[0], detected[1]))
        except Exception:
            pass
    else:
        try:
            part.insert(0, m21key.Key(resolved_key))
        except Exception as exc:
            raise ValueError(f"无法识别调号 {resolved_key!r}：{exc}") from exc

    for index, voice_data in enumerate(voices, start=1):
        voice = stream.Voice(id=f"voice-{index}")
        cursor = 0.0
        for start, dur, pitches in voice_data["events"]:
            if start > cursor + 1e-9:
                rest = note.Rest()
                rest.duration = duration.Duration(quarterLength=start - cursor)
                voice.insert(cursor, rest)
            item = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
            item.duration = duration.Duration(quarterLength=dur)
            voice.insert(start, item)
            cursor = start + dur
        part.insert(0, voice)

    score = stream.Score()
    score.append(part.makeNotation(inPlace=False))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out_path))
    return out_path
