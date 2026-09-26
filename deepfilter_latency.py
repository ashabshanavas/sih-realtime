import numpy as np
import time

from deepfilter_adapter import DeepFilterAdapter


SAMPLE_RATE = 16_000
BLOCK_SIZE = 160

TOTAL_SAMPLES = 32_000

# Give DeepFilterNet time to warm up.
WARMUP_SAMPLES = 4_000

# Length of our test signal.
TEST_LENGTH = 800


print("======================================")
print(" DEEPFILTERNET STREAMING DELAY TEST")
print("======================================")
print()

print("Sample rate :", SAMPLE_RATE)
print("Block size  :", BLOCK_SIZE)
print("Total audio :", TOTAL_SAMPLES, "samples")
print("Warm-up     :", WARMUP_SAMPLES, "samples")
print()


# ------------------------------------------------------------
# Create input signal
# ------------------------------------------------------------

input_signal = np.zeros(
    TOTAL_SAMPLES,
    dtype=np.float32
)


# Create a short broadband test signal.
#
# A single impulse can be heavily suppressed by a
# speech denoiser, so we use a short noise burst instead.

rng = np.random.default_rng(12345)

test_signal = (
    rng.standard_normal(TEST_LENGTH)
    * 0.3
).astype(np.float32)


TEST_START = WARMUP_SAMPLES

input_signal[
    TEST_START:TEST_START + TEST_LENGTH
] = test_signal


print("Test signal starts at:", TEST_START)
print("Test signal length    :", TEST_LENGTH)
print()


# ------------------------------------------------------------
# Create DeepFilterNet
# ------------------------------------------------------------

adapter = DeepFilterAdapter()

output_signal = np.zeros(
    TOTAL_SAMPLES,
    dtype=np.float32
)

processing_times = []


# ------------------------------------------------------------
# Process streaming audio
# ------------------------------------------------------------

print("Processing...")
print()

output_position = 0

try:

    for start in range(
        0,
        TOTAL_SAMPLES,
        BLOCK_SIZE
    ):

        frame = input_signal[
            start:start + BLOCK_SIZE
        ]

        if len(frame) < BLOCK_SIZE:

            padded = np.zeros(
                BLOCK_SIZE,
                dtype=np.float32
            )

            padded[:len(frame)] = frame

            frame = padded


        processing_start = time.perf_counter()

        output = adapter.process(frame)

        processing_end = time.perf_counter()


        processing_times.append(
            (processing_end - processing_start)
            * 1000
        )


        end = output_position + BLOCK_SIZE

        if end <= TOTAL_SAMPLES:

            output_signal[
                output_position:end
            ] = output


        output_position += BLOCK_SIZE


finally:

    adapter.close()


# ------------------------------------------------------------
# Cross-correlation
# ------------------------------------------------------------

print("Finding signal alignment...")
print()


# Search only after the original test signal.
#
# We allow a large search window because we don't yet know
# the actual delay.

search_start = TEST_START

search_end = min(
    TOTAL_SAMPLES - TEST_LENGTH,
    TEST_START + 8_000
)


best_correlation = -np.inf
best_position = None


for position in range(
    search_start,
    search_end
):

    candidate = output_signal[
        position:position + TEST_LENGTH
    ]


    # Normalize both signals.

    candidate_energy = np.sqrt(
        np.sum(candidate ** 2)
    )

    test_energy = np.sqrt(
        np.sum(test_signal ** 2)
    )


    if candidate_energy == 0:

        continue


    correlation = np.sum(
        candidate * test_signal
    ) / (
        candidate_energy * test_energy
    )


    if correlation > best_correlation:

        best_correlation = correlation

        best_position = position


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

processing_times = np.array(
    processing_times
)


print()
print("======================================")
print(" RESULTS")
print("======================================")
print()


print(
    f"Input signal position : "
    f"{TEST_START} samples"
)


print(
    f"Best output position  : "
    f"{best_position} samples"
)


delay_samples = (
    best_position - TEST_START
)


delay_ms = (
    delay_samples
    / SAMPLE_RATE
    * 1000
)


print(
    f"Delay                 : "
    f"{delay_samples} samples"
)


print(
    f"Estimated delay       : "
    f"{delay_ms:.2f} ms"
)


print()


print(
    f"Correlation           : "
    f"{best_correlation:.4f}"
)


print()


print(
    f"Average processing    : "
    f"{np.mean(processing_times):.3f} ms"
)


print(
    f"Maximum processing    : "
    f"{np.max(processing_times):.3f} ms"
)


print()


# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

if best_correlation > 0.3:

    print(
        "Signal alignment detected successfully."
    )

else:

    print(
        "Correlation was weak."
    )

    print(
        "The denoiser may be strongly modifying "
        "the broadband test signal."
    )


print()
print("======================================")