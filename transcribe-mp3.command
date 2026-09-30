#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 input.mp3 [output.txt]" >&2
  exit 2
fi

input=$1
case "$input" in
  *.[mM][pP]3) ;;
  *) echo "Expected an MP3 file: $input" >&2; exit 2 ;;
esac
if [[ ! -f "$input" ]]; then
  echo "File not found: $input" >&2
  exit 1
fi

output=${2:-${input%.*}.txt}
if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but was not found." >&2
  exit 1
fi

model="$script_dir/nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin"
recognizer="$script_dir/nemotron_asr"
if [[ ! -f "$model" || ! -x "$recognizer" ]]; then
  echo "Model or recognizer missing from $script_dir" >&2
  exit 1
fi

threads=${NEMO_THREADS:-4}
if ! [[ "$threads" =~ ^([1-9]|1[0-6])$ ]]; then
  echo "NEMO_THREADS must be from 1 to 16." >&2
  exit 2
fi

temp_dir=$(mktemp -d)
cleanup() {
  rm -f -- "$temp_dir/transcript.txt"
  rmdir -- "$temp_dir"
}
trap cleanup EXIT

ffmpeg -nostdin -hide_banner -loglevel error -i "$input" \
  -vn -ac 1 -ar 16000 -c:a pcm_s16le -f s16le - \
  | "$recognizer" -m "$model" --stdin \
      -l auto --strip-tags --att-right 6 -t "$threads" > "$temp_dir/transcript.txt"

mv -- "$temp_dir/transcript.txt" "$output"
cat -- "$output"
echo "Saved transcript: $output" >&2
