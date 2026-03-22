"""Test the API server's streaming and non-streaming speech endpoints using the OpenAI SDK.

Usage:
    pip install openai
    python tests/test_streaming.py --base-url http://localhost:8080/v1 --output-dir test-output/stream

Produces WAV files in the output directory:
  - {voice}_sdk.wav        : non-streaming via SDK with_streaming_response (proves SDK compat)
  - {voice}_stream.wav     : SSE streaming reassembled into WAV
"""

import argparse
import base64
import json
import struct
import sys
from pathlib import Path

from openai import OpenAI


SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
NUM_CHANNELS = 1

STREAM_TEXT = (
    "Hello, this is a streaming test. "
    "The first chunk should arrive quickly. "
    "Then the rest of the sentences follow one by one. "
    "This verifies that SSE streaming works correctly."
)


def pcm_to_wav(pcm_data: bytes) -> bytes:
    """Wrap raw 16-bit LE PCM data in a WAV header (mono, 24 kHz)."""
    data_size = len(pcm_data)
    byte_rate = SAMPLE_RATE * NUM_CHANNELS * BITS_PER_SAMPLE // 8
    block_align = NUM_CHANNELS * BITS_PER_SAMPLE // 8

    header = b"RIFF"
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVE"
    header += b"fmt "
    header += struct.pack(
        "<IHHIIHH",
        16, 1, NUM_CHANNELS, SAMPLE_RATE, byte_rate, block_align, BITS_PER_SAMPLE,
    )
    header += b"data"
    header += struct.pack("<I", data_size)
    return header + pcm_data


def test_sdk_streaming(client: OpenAI, voice: str, output_dir: Path):
    """Test non-streaming endpoint via SDK with_streaming_response (transport-level streaming)."""
    out_path = output_dir / f"{voice}_sdk.wav"
    print(f"[SDK] Generating {voice} via with_streaming_response ...")

    with client.audio.speech.with_streaming_response.create(
        model="kitten-tts",
        voice=voice,
        input=STREAM_TEXT,
        response_format="wav",
    ) as response:
        response.stream_to_file(str(out_path))

    size = out_path.stat().st_size
    assert size > 44, f"WAV file too small ({size} bytes), expected audio data"
    print(f"[SDK] {out_path.name}: {size} bytes")


def test_sse_streaming(client: OpenAI, voice: str, output_dir: Path):
    """Test SSE streaming endpoint: collect audio deltas and reassemble into WAV."""
    out_path = output_dir / f"{voice}_stream.wav"
    print(f"[SSE] Streaming {voice} via stream=true ...")

    # Use extra_body to pass stream=true, and get raw SSE text via iter_text
    with client.audio.speech.with_streaming_response.create(
        model="kitten-tts",
        voice=voice,
        input=STREAM_TEXT,
        response_format="pcm",
        extra_body={"stream": True},
    ) as response:
        raw_text = response.text

    # Parse SSE events from the raw response
    pcm_data = bytearray()
    chunk_count = 0
    got_done = False

    for line in raw_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        event = json.loads(payload)

        if event["type"] == "speech.audio.delta":
            pcm_data.extend(base64.b64decode(event["delta"]))
            chunk_count += 1
        elif event["type"] == "speech.audio.done":
            got_done = True
        elif event["type"] == "error":
            print(f"[SSE] ERROR: {event['error']['message']}", file=sys.stderr)
            sys.exit(1)

    assert chunk_count > 0, "No audio chunks received"
    assert got_done, "Never received speech.audio.done event"
    assert chunk_count > 1, f"Expected multiple chunks for multi-sentence text, got {chunk_count}"

    wav_data = pcm_to_wav(bytes(pcm_data))
    out_path.write_bytes(wav_data)

    print(
        f"[SSE] {out_path.name}: {chunk_count} chunks, "
        f"{len(pcm_data)} PCM bytes, {len(wav_data)} WAV bytes"
    )


def main():
    parser = argparse.ArgumentParser(description="Test KittenTTS streaming endpoints")
    parser.add_argument(
        "--base-url", default="http://localhost:8080/v1",
        help="Base URL for the API server (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--output-dir", default="test-output/stream",
        help="Directory to write output WAV files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    voices = ["alloy", "nova"]
    failures = []

    for voice in voices:
        try:
            test_sdk_streaming(client, voice, output_dir)
        except Exception as e:
            print(f"FAIL [SDK] {voice}: {e}", file=sys.stderr)
            failures.append(f"SDK/{voice}")

        try:
            test_sse_streaming(client, voice, output_dir)
        except Exception as e:
            print(f"FAIL [SSE] {voice}: {e}", file=sys.stderr)
            failures.append(f"SSE/{voice}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All tests passed. Output in {output_dir}/")


if __name__ == "__main__":
    main()
