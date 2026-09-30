"""Local FastAPI upload interface for the Nemotron C recognizer."""

import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import psutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin"
RECOGNIZER = ROOT / "nemotron_asr"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
TIMEOUT_SECONDS = 30 * 60
CHUNK_BYTES = 1024 * 1024
ALLOWED_SUFFIXES = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".webm", ".aac"}
processing_lock = threading.Lock()
state_lock = threading.Lock()
stage = "Idle"
model_process: psutil.Process | None = None

app = FastAPI(title="Nemotron Transcription")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


def set_state(new_stage: str, process: psutil.Process | None = None) -> None:
    global stage, model_process
    with state_lock:
        stage = new_stage
        model_process = process


@app.get("/api/metrics")
def metrics() -> dict:
    with state_lock:
        cpu_percent = 0.0
        ram_mb = 0.0
        if model_process is not None:
            try:
                cpu_percent = model_process.cpu_percent(interval=None)
                ram_mb = model_process.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {
            "active": stage != "Idle",
            "stage": stage,
            "cpu_percent": round(cpu_percent, 1),
            "ram_mb": round(ram_mb, 1),
        }


def transcribe_audio(source: Path, wav: Path) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not MODEL.is_file() or not RECOGNIZER.is_file():
        raise HTTPException(
            status_code=503,
            detail="Server setup is incomplete: ffmpeg, the model, and the built recognizer are required.",
        )

    try:
        threads = int(os.environ.get("NEMO_THREADS", "4"))
    except ValueError:
        raise HTTPException(status_code=503, detail="NEMO_THREADS must be an integer from 1 to 16.")
    if not 1 <= threads <= 16:
        raise HTTPException(status_code=503, detail="NEMO_THREADS must be from 1 to 16.")

    try:
        set_state("Converting audio")
        converted = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False,
        )
        if converted.returncode != 0 or not wav.is_file() or wav.stat().st_size <= 44:
            raise HTTPException(status_code=400, detail="Could not decode this audio file.")

        recognizer = subprocess.Popen(
            [str(RECOGNIZER), "-m", str(MODEL), "-i", str(wav),
             "-l", "auto", "--strip-tags", "--att-right", "6", "-t", str(threads)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            process = psutil.Process(recognizer.pid)
            process.cpu_percent(interval=None)  # Prime the next nonblocking CPU sample.
        except psutil.NoSuchProcess:
            process = None
        set_state("Transcribing", process)
        try:
            stdout, _ = recognizer.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            recognizer.kill()
            recognizer.communicate()
            raise
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Processing timed out.")
    except OSError:
        raise HTTPException(status_code=503, detail="Could not start the audio processor.")

    if recognizer.returncode != 0:
        raise HTTPException(status_code=500, detail="The recognizer failed to process this audio.")
    return stdout.strip()


@app.post("/api/transcribe")
def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Choose an audio file.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")
    if not processing_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another transcription is processing. Try again shortly.")

    try:
        set_state("Uploading audio")
        with tempfile.TemporaryDirectory(prefix="nemotron-upload-") as temp_dir:
            source = Path(temp_dir) / f"upload{suffix}"
            wav = Path(temp_dir) / "audio.wav"
            size = 0
            with source.open("wb") as output:
                while chunk := file.file.read(CHUNK_BYTES):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Audio file exceeds the 200 MB limit.")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="The audio file is empty.")
            return {"text": transcribe_audio(source, wav)}
    finally:
        file.file.close()
        set_state("Idle")
        processing_lock.release()
