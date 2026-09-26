import numpy as np

from dfnstream_py import DeepFilterNetStreaming


BLOCK_SIZE = 160


class DeepFilterEnhancer:

    def __init__(self):
        print("Loading DeepFilterNet...")

        self.processor = DeepFilterNetStreaming()

        print("DeepFilterNet loaded.")

    def process(self, frame):
        """
        Process one audio frame.

        Input:
            frame: float32 numpy array

        Output:
            enhanced audio frame
        """

        frame = np.asarray(
            frame,
            dtype=np.float32
        )

        output = self.processor.process_chunk(frame)

        return np.asarray(
            output,
            dtype=np.float32
        )

    def close(self):
        self.processor.close()


# Create the model once.
enhancer = DeepFilterEnhancer()


def enhance_frame(frame_160, state):
    """
    Standard interface used by the real-time pipeline.

    frame_160:
        One incoming audio block.

    state:
        Reserved for model state.

    Returns:
        enhanced_frame, state
    """

    frame_160 = np.asarray(
        frame_160,
        dtype=np.float32
    )

    enhanced = enhancer.process(frame_160)

    return enhanced, state
