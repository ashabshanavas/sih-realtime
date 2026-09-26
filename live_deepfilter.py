import time
import threading

import numpy as np
import sounddevice as sd

from deepfilter_adapter import DeepFilterAdapter


# ============================================================
# AUDIO SETTINGS
# ============================================================

SAMPLE_RATE = 16_000
BLOCK_SIZE = 160
CHANNELS = 1

INPUT_DEVICE = 0
OUTPUT_DEVICE = 1


# ============================================================
# GLOBAL STATE
# ============================================================

running = True
enhancer = None

processing_times = []
callback_count = 0


# ============================================================
# AUDIO CALLBACK
# ============================================================

def audio_callback(indata, outdata, frames, time_info, status):

    global callback_count

    if status:
        print("Audio status:", status)

    start_time = time.perf_counter()

    # Get mono microphone signal.
    frame = indata[:, 0].copy()

    # Process through DeepFilterNet.
    enhanced_frame = enhancer.process(frame)

    # Send enhanced audio to output.
    outdata[:, 0] = enhanced_frame

    # Measure processing time.
    processing_time = time.perf_counter() - start_time

    processing_times.append(processing_time)

    callback_count += 1


# ============================================================
# KEYBOARD CONTROL
# ============================================================

def keyboard_thread():

    global running

    print()
    print("Press Q then ENTER to stop.")
    print()

    while running:

        command = input().strip().lower()

        if command == "q":
            running = False
            break


# ============================================================
# MAIN
# ============================================================

def main():

    global enhancer

    print("======================================")
    print(" LIVE DEEPFILTERNET TEST")
    print("======================================")
    print()

    print("Sample rate :", SAMPLE_RATE)
    print("Block size  :", BLOCK_SIZE)
    print("Block time  :", BLOCK_SIZE / SAMPLE_RATE * 1000, "ms")
    print("Input       :", INPUT_DEVICE)
    print("Output      :", OUTPUT_DEVICE)
    print()

    # Load DeepFilterNet.
    enhancer = DeepFilterAdapter()

    # Start keyboard thread.
    thread = threading.Thread(
        target=keyboard_thread,
        daemon=True
    )

    thread.start()

    print()
    print("Starting live audio...")
    print("Speak into the microphone.")
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

            while running:
                time.sleep(0.1)

    except KeyboardInterrupt:

        print()
        print("Stopped by keyboard interrupt.")

    finally:

        enhancer.close()

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("======================================")
    print(" LIVE TEST RESULTS")
    print("======================================")

    print()
    print("Callbacks processed :", callback_count)

    if processing_times:

        times_ms = np.array(processing_times) * 1000

        average = np.mean(times_ms)
        minimum = np.min(times_ms)
        maximum = np.max(times_ms)

        block_time_ms = (
            BLOCK_SIZE / SAMPLE_RATE
        ) * 1000

        rtf = average / block_time_ms

        print(
            f"Audio per block     : "
            f"{block_time_ms:.2f} ms"
        )

        print(
            f"Average processing  : "
            f"{average:.3f} ms"
        )

        print(
            f"Minimum processing  : "
            f"{minimum:.3f} ms"
        )

        print(
            f"Maximum processing  : "
            f"{maximum:.3f} ms"
        )

        print(
            f"Real-Time Factor    : "
            f"{rtf:.4f}"
        )

    print()
    print("======================================")


if __name__ == "__main__":
    main()