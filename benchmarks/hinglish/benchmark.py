"""Reproducible accuracy and resource benchmark on ujs/hinglish's test split."""

import argparse
import csv
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import tarfile
import tempfile
import time
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin"
RECOGNIZER = ROOT / "nemotron_asr"
DATA = HERE / "data"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(
        char if char.isalnum() or unicodedata.category(char).startswith("M") else " "
        for char in text
    ).split())


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref_item in enumerate(reference, 1):
        current = [index]
        for position, hyp_item in enumerate(hypothesis, 1):
            current.append(min(
                current[-1] + 1,
                previous[position] + 1,
                previous[position - 1] + (ref_item != hyp_item),
            ))
        previous = current
    return previous[-1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def recognize(wav_path: Path, threads: int) -> tuple[str, float, float, float]:
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    command = [str(RECOGNIZER), "-m", str(MODEL), "-i", str(wav_path),
               "-l", "auto", "--strip-tags", "--att-right", "6", "-t", str(threads)]
    worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    process = psutil.Process(worker.pid)
    peak_ram_bytes = 0
    while True:
        try:
            peak_ram_bytes = max(peak_ram_bytes, process.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        try:
            stdout, stderr = worker.communicate(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            continue
    wall_seconds = time.perf_counter() - start
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = ((usage_after.ru_utime + usage_after.ru_stime)
                   - (usage_before.ru_utime + usage_before.ru_stime))
    if worker.returncode != 0:
        raise RuntimeError(f"Recognizer failed for {wav_path.name}: {stderr[-500:]}")
    return stdout.strip(), wall_seconds, cpu_seconds, peak_ram_bytes / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1 or not 1 <= args.threads <= 16:
        parser.error("--samples must be positive and --threads must be from 1 to 16")
    if not MODEL.is_file() or not RECOGNIZER.is_file():
        parser.error("Build nemotron_asr and download the model first")

    with (DATA / "metadata-test.csv").open(newline="", encoding="utf-8") as source:
        rows = [(row[0], ",".join(row[1:])) for row in csv.reader(source) if len(row) >= 2]
    if args.samples > len(rows):
        parser.error(f"Only {len(rows)} test rows are available")
    selected = random.Random(args.seed).sample(rows, args.samples)
    results = []
    start_all = time.perf_counter()

    with tarfile.open(DATA / "test.tar.gz", "r:gz") as archive, tempfile.TemporaryDirectory() as temp:
        temp_path = Path(temp)
        for index, (audio_name, reference) in enumerate(selected, 1):
            member = archive.extractfile(audio_name)
            if member is None:
                raise RuntimeError(f"Missing audio in archive: {audio_name}")
            source_wav = temp_path / "source.wav"
            normalized_wav = temp_path / "normalized.wav"
            source_wav.write_bytes(member.read())
            clip_start = time.perf_counter()
            converted = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(source_wav), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(normalized_wav)],
                capture_output=True, text=True, check=False,
            )
            if converted.returncode != 0:
                raise RuntimeError(f"ffmpeg failed for {audio_name}: {converted.stderr[-500:]}")
            with wave.open(str(normalized_wav), "rb") as audio:
                duration = audio.getnframes() / audio.getframerate()
            hypothesis, inference_seconds, cpu_seconds, peak_ram_mb = recognize(normalized_wav, args.threads)
            total_seconds = time.perf_counter() - clip_start

            ref_normal = normalize(reference)
            hyp_normal = normalize(hypothesis)
            ref_words = ref_normal.split()
            hyp_words = hyp_normal.split()
            word_errors = edit_distance(ref_words, hyp_words)
            char_errors = edit_distance(ref_normal.replace(" ", ""), hyp_normal.replace(" ", ""))
            results.append({
                "path": audio_name, "duration_s": round(duration, 3),
                "latency_s": round(total_seconds, 3),
                "inference_s": round(inference_seconds, 3),
                "cpu_seconds": round(cpu_seconds, 3),
                "mean_cpu_percent": round(100 * cpu_seconds / inference_seconds, 1),
                "peak_ram_mb": round(peak_ram_mb, 1),
                "reference_words": len(ref_words), "word_errors": word_errors,
                "reference_chars": len(ref_normal.replace(" ", "")), "char_errors": char_errors,
                "reference": reference, "hypothesis": hypothesis,
            })
            print(f"{index:3}/{len(selected)}  {duration:5.1f}s audio  {total_seconds:5.2f}s latency  "
                  f"{100 * cpu_seconds / inference_seconds:5.0f}% CPU  {peak_ram_mb:6.1f} MB RAM  "
                  f"WER {word_errors / max(len(ref_words), 1):.2f}", flush=True)

    latencies = [row["latency_s"] for row in results]
    audio_seconds = sum(row["duration_s"] for row in results)
    total_latency = sum(latencies)
    word_errors = sum(row["word_errors"] for row in results)
    reference_words = sum(row["reference_words"] for row in results)
    char_errors = sum(row["char_errors"] for row in results)
    reference_chars = sum(row["reference_chars"] for row in results)
    wer = word_errors / reference_words
    summary = {
        "dataset": "ujs/hinglish", "split": "test", "population_rows": len(rows),
        "samples": len(results), "seed": args.seed, "threads": args.threads,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.platform(), "logical_cpus": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "audio_seconds": round(audio_seconds, 3),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_median_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(percentile(latencies, 0.95), 3),
        "real_time_factor": round(total_latency / audio_seconds, 3),
        "mean_cpu_percent": round(100 * sum(row["cpu_seconds"] for row in results)
                                  / sum(row["inference_s"] for row in results), 1),
        "peak_ram_mb": max(row["peak_ram_mb"] for row in results),
        "word_error_rate": round(wer, 4),
        "word_accuracy_1_minus_wer": round(1 - wer, 4),
        "character_error_rate_no_spaces": round(char_errors / reference_chars, 4),
        "exact_match_rate": round(sum(normalize(row["reference"]) == normalize(row["hypothesis"])
                                      for row in results) / len(results), 4),
        "elapsed_benchmark_s": round(time.perf_counter() - start_all, 3),
    }
    HERE.joinpath("results.json").write_text(
        json.dumps({"summary": summary, "samples": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with HERE.joinpath("results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
