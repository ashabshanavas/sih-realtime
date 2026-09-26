import numpy as np


BLOCK_SIZE = 160


def enhance_frame(frame_160, state):
    """
    Process one 10 ms audio frame.

    Input:
        frame_160 : numpy array containing exactly 160 samples
                    at 16 kHz, represented as float32.

        state     : model's internal state.
                    Currently unused because this is a dummy enhancer.

    Output:
        out_160   : enhanced 160-sample frame
        state     : updated model state
    """

    # Make sure we received exactly one 10 ms frame.
    if len(frame_160) != BLOCK_SIZE:
        raise ValueError(
            f"Expected {BLOCK_SIZE} samples, "
            f"but received {len(frame_160)} samples."
        )

    # Dummy enhancer:
    # For now, simply pass the audio through unchanged.
    out_160 = frame_160.copy()

    # No internal state is required yet.
    return out_160, state
