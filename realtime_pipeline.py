import time

import numpy as np
import sounddevice as sd

from enhancer import enhance_frame


SAMPLE_RATE = 16_000
BLOCK_SIZE = 160
CHANNELS = 1

INPUT_DEVICE = 0
OUTPUT_DEVICE = 1


# Model state.
# Later this will contain things like recurrent/conv states.
state = None

processing_times = []
callback_count = 0


def audio_callback(indata, outdata, frames, time_info, status):
    global state
    global callback_count

    if status:
        print("Audio status:", status)

    start_time = time.perf_counter()

    # ----------------------------------------
    # 1. Get the microphone frame
    # ----------------------------------------

    frame = indata[:, 0].copy()

    # ----------------------------------------
    # 2. Send frame to the enhancer
    # ----------------------------------------

    enhanced_frame, state = enhance_frame(
        frame,
        state
    )

    # ----------------------------------------
    # 3. Send enhanced audio to headphones
    # ----------------------------------------

    outdata[:, 0] = enhanced_frame

    # ----------------------------------------
    # 4. Measure processing time
    # ----------------------------------------

    end_time = time.perf_counter()

    processing_time = end_time - start_time

    processing_times.append(processing_time)

    callback_count += 1


print("==============================")
print(" REAL-TIME AUDIO PIPELINE")
print("==============================")
print()
print("Sample rate :", SAMPLE_RATE, "Hz")
print("Block size  :", BLOCK_SIZE, "samples")
print("Block time  :", BLOCK_SIZE / SAMPLE_RATE * 1000, "ms")
print()
print("Speak into the microphone.")
print("Press Ctrl+C to stop.")
print()


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
            time.sleep(1)


except KeyboardInterrupt:

    print()
    print("Stopping...")


# ----------------------------------------
# Calculate statistics
# ----------------------------------------

if processing_times:

    times_ms = np.array(processing_times) * 1000

    audio_duration_ms = (
        BLOCK_SIZE / SAMPLE_RATE
    ) * 1000

    average_ms = np.mean(times_ms)
    minimum_ms = np.min(times_ms)
    maximum_ms = np.max(times_ms)

    rtf = average_ms / audio_duration_ms

    print()
    print("==============================")
    print(" PIPELINE RESULTS")
    print("==============================")

    print(
        f"Callbacks processed : {callback_count}"
    )

    print(
        f"Audio per block     : "
        f"{audio_duration_ms:.2f} ms"
    )

    print(
        f"Average processing  : "
        f"{average_ms:.4f} ms"
    )

    print(
        f"Minimum processing  : "
        f"{minimum_ms:.4f} ms"
    )

    print(
        f"Maximum processing  : "
        f"{maximum_ms:.4f} ms"
    )

    print(
        f"Real-Time Factor    : "
        f"{rtf:.4f}"
    )

    print("==============================")
