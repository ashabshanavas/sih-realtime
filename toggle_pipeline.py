import time
import threading

import numpy as np
import sounddevice as sd

from enhancer import enhance_frame


SAMPLE_RATE = 16_000
BLOCK_SIZE = 160
CHANNELS = 1

INPUT_DEVICE = 0
OUTPUT_DEVICE = 1


# ----------------------------------------
# Global state
# ----------------------------------------

model_state = None

# True  = enhanced/model output
# False = raw microphone output
enhanced_mode = False

running = True

processing_times = []
callback_count = 0


# ----------------------------------------
# Keyboard control
# ----------------------------------------

def keyboard_control():
    global enhanced_mode
    global running

    print()
    print("Controls:")
    print("  R = Raw microphone")
    print("  E = Enhanced/model output")
    print("  Q = Quit")
    print()

    while running:

        command = input().strip().lower()

        if command == "r":
            enhanced_mode = False
            print("Mode: RAW microphone")

        elif command == "e":
            enhanced_mode = True
            print("Mode: ENHANCED/model")

        elif command == "q":
            running = False
            print("Stopping...")

        else:
            print("Unknown command. Use R, E, or Q.")


# ----------------------------------------
# Audio callback
# ----------------------------------------

def audio_callback(indata, outdata, frames, time_info, status):
    global model_state
    global callback_count

    if status:
        print("Audio status:", status)

    start_time = time.perf_counter()

    # Get the microphone frame.
    frame = indata[:, 0].copy()

    # Always run the enhancer.
    enhanced_frame, model_state = enhance_frame(
        frame,
        model_state
    )

    # Choose what goes to headphones.
    if enhanced_mode:
        outdata[:, 0] = enhanced_frame
    else:
        outdata[:, 0] = frame

    # Measure processing time.
    end_time = time.perf_counter()

    processing_time = end_time - start_time

    processing_times.append(processing_time)

    callback_count += 1


# ----------------------------------------
# Start
# ----------------------------------------

print("==============================")
print(" RAW / ENHANCED AUDIO TEST")
print("==============================")
print()
print("Sample rate :", SAMPLE_RATE, "Hz")
print("Block size  :", BLOCK_SIZE, "samples")
print("Block time  :", BLOCK_SIZE / SAMPLE_RATE * 1000, "ms")
print()
print("Starting audio...")
print()


# Start keyboard input in another thread.
keyboard_thread = threading.Thread(
    target=keyboard_control,
    daemon=True
)

keyboard_thread.start()


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

    running = False
    print()
    print("Interrupted.")


# ----------------------------------------
# Results
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
    print(" RESULTS")
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
