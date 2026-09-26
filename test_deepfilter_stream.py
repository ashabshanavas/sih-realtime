import numpy as np

from dfnstream_py import DeepFilterNetStreaming


print("Creating DeepFilterNet...")
processor = DeepFilterNetStreaming()

print("Model loaded.")
print()

for i in range(30):

    frame = np.zeros(
        160,
        dtype=np.float32
    )

    output = processor.process_chunk(frame)

    print(
        f"Frame {i + 1:02d}: "
        f"input={len(frame):3d} samples  "
        f"output={len(output):3d} samples"
    )


processor.close()

print()
print("Test completed.")