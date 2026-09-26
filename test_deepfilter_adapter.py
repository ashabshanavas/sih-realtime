import numpy as np

from deepfilter_adapter import DeepFilterAdapter


print("Creating adapter...")
adapter = DeepFilterAdapter()

print()
print("Testing 30 input frames...")
print()

total_input = 0
total_output = 0

try:
    for i in range(30):

        frame = np.zeros(
            160,
            dtype=np.float32
        )

        output = adapter.process(frame)

        total_input += len(frame)
        total_output += len(output)

        print(
            f"Frame {i + 1:02d}: "
            f"input={len(frame)} "
            f"output={len(output)}"
        )

finally:
    adapter.close()


print()
print("==============================")
print(" ADAPTER TEST RESULTS")
print("==============================")
print()
print(f"Total input  : {total_input} samples")
print(f"Total output : {total_output} samples")
print()
print("==============================")