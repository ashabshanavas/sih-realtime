import sounddevice as sd
import numpy as np
import time


# =========================
# Configuration
# =========================

SAMPLE_RATE = 16_000
BLOCK_SIZE = 160
CHANNELS = 1

INPUT_DEVICE = 0
OUTPUT_DEVICE = 1


# =========================
# Statistics
# =========================

processing_times = []
callback_count = 0


# =========================
# Real-time callback
# =========================

def audio_callback(indata, outdata, frames, time_info, status):

    global callback_count

    if status:
        print("Audio status:", status)

    start_time = time.perf_counter()

    # --------------------------------
    # Our current "processing"
    # --------------------------------

    outdata[:] = indata

    # --------------------------------
    # End processing
    # --------------------------------

    end_time = time.perf_counter()

    processing_time = end_time - start_time

    processing_times.append(processing_time)

    callback_count += 1


# =========================
# Start test
# =========================

print("Starting latency test...")
print()
print("Speak into the microphone.")
print("The system will run for 10 seconds.")
print()


with sd.Stream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    channels=CHANNELS,
    dtype="float32",
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    callback=audio_callback,
):

    time.sleep(10)


# =========================
# Calculate results
# =========================

times_ms = np.array(processing_times) * 1000

audio_duration_ms = (BLOCK_SIZE / SAMPLE_RATE) * 1000

average_ms = np.mean(times_ms)
maximum_ms = np.max(times_ms)
minimum_ms = np.min(times_ms)

rtf = average_ms / audio_duration_ms


print()
print("==============================")
print(" REAL-TIME LATENCY RESULTS")
print("==============================")

print(f"Callbacks processed : {callback_count}")
print(f"Audio per block     : {audio_duration_ms:.2f} ms")
print(f"Average processing  : {average_ms:.4f} ms")
print(f"Minimum processing  : {minimum_ms:.4f} ms")
print(f"Maximum processing  : {maximum_ms:.4f} ms")
print(f"Real-Time Factor    : {rtf:.4f}")

print("==============================")
