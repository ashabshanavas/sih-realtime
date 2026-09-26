import sounddevice as sd
import numpy as np


# =========================
# Audio configuration
# =========================

SAMPLE_RATE = 16_000       # 16 kHz
BLOCK_SIZE = 160           # 10 ms = 160 samples
CHANNELS = 1               # Mono

INPUT_DEVICE = 0           # MacBook Air Microphone
OUTPUT_DEVICE = 1          # MacBook Air Speakers


# =========================
# Real-time audio callback
# =========================

def audio_callback(indata, outdata, frames, time, status):

    # Report audio problems, if any.
    if status:
        print(status)

    # Pass microphone audio directly to the speakers.
    outdata[:] = indata


# =========================
# Start real-time stream
# =========================

print("Starting real-time audio...")
print("Microphone -> Python -> Speakers")
print("Press Ctrl+C to stop.\n")

try:
    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=CHANNELS,
        dtype="float32",
        device=(INPUT_DEVICE, OUTPUT_DEVICE),
        callback=audio_callback,
    ):
        while True:
            sd.sleep(1000)

except KeyboardInterrupt:
    print("\nStopped.")
