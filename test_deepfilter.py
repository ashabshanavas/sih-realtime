import numpy as np

from dfnstream_py import DeepFilterNetStreaming


print("Creating DeepFilterNet...")

processor = DeepFilterNetStreaming()

print("Model loaded.")
print()


# Generate 10 ms of fake 16 kHz audio.
frame = np.zeros(
    160,
    dtype=np.float32
)

print("Input:")
print("  samples:", len(frame))
print("  dtype  :", frame.dtype)


output = processor.process_chunk(frame)


print()
print("Output:")
print("  samples:", len(output))
print("  dtype  :", output.dtype)


processor.close()

print()
print("Test completed successfully.")
