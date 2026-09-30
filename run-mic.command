#!/bin/zsh
cd -- "$(dirname -- "$0")" || exit 1

echo "Listening on the MacBook Air microphone. Press Ctrl-C to stop."
ffmpeg -nostdin -hide_banner -loglevel error -nostats \
  -f avfoundation -i ':0' -ac 1 -ar 16000 -f s16le - \
  | ./nemotron_asr \
      -m nemotron-3.5-asr-streaming-0.6b-w8a8-linear.bin \
      --stdin -l auto --strip-tags --att-right 3 -t 4
