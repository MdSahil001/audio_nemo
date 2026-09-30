# Hinglish transcription benchmark

Source: [ujs/hinglish](https://huggingface.co/datasets/ujs/hinglish), a Hindi–English
code switched audio dataset. This benchmark uses the **test** split, which has
3,136 reference transcripts. The downloaded files are retained locally under
`data/` and excluded from Git because the audio archive is large.

## Result

The run on 30 September 2026 used 50 clips selected uniformly with random seed
42 from the test split (319.5 seconds of audio), four recognizer threads,
`-l auto --strip-tags --att-right 6`, on an 8 logical CPU, 8 GB Apple Silicon Mac.
Each request starts a new recognizer process, as the web server does.

| Measure | Result |
| --- | ---: |
| Median end-to-end latency per clip | 1.053 s |
| Mean / 95th percentile latency | 1.171 s / 2.119 s |
| Total processing time / audio duration | 0.183× (5.46× real time) |
| Mean recognizer CPU usage | 197.1% (about 1.97 cores) |
| Highest sampled recognizer RAM (RSS) | 692.1 MB |
| Word error rate (WER) | 57.21% |
| Word accuracy (`1 - WER`) | 42.79% |
| Character error rate, spaces excluded | 62.01% |
| Exact transcript match | 6% |

Latency includes the web server's FFmpeg conversion command, model loading, and
recognition, but excludes HTTP transfer. CPU is recognizer user plus system CPU
time divided by recognition wall time; it can exceed 100% with multiple cores.
RAM is the highest process RSS observed with 50 ms sampling. WER is corpus
Levenshtein distance divided by reference word count. Text is NFKC normalized,
case folded, stripped of punctuation, and whitespace collapsed; scripts and
spellings are preserved. Thus a reference English word and the same spoken word
written in Devanagari count as different. Some clips also produced repeated
phrases, increasing WER. These numbers describe this sample and normalization,
not the full test split or semantic correctness.

Per-clip references, hypotheses, latency, CPU, RAM, and error counts are in
[results.csv](results.csv). [results.json](results.json) contains the same rows
and the aggregate summary.

## Reproduce

```bash
mkdir -p benchmarks/hinglish/data
curl -L --fail -o benchmarks/hinglish/data/metadata-test.csv \
  https://huggingface.co/datasets/ujs/hinglish/resolve/main/data/metadata-test.csv
curl -L --fail -o benchmarks/hinglish/data/test.tar.gz \
  https://huggingface.co/datasets/ujs/hinglish/resolve/main/data/test.tar.gz
.venv/bin/python benchmarks/hinglish/benchmark.py --samples 50 --seed 42 --threads 4
```

The run used these SHA-256 hashes:

```text
metadata-test.csv  ed7ff641ada1bfaad7f20f2acfbc886a90ca7abbfa2d1d9897b0e7345242bf8a
test.tar.gz        35d1603741d364556ae4c9c3582f4a966acf107de89fc9eb829fd1e6326d46c1
```
