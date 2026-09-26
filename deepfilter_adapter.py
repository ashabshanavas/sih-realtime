import numpy as np
from dfnstream_py import DeepFilterNetStreaming


INPUT_BLOCK_SIZE = 160
MODEL_BLOCK_SIZE = 480


class DeepFilterAdapter:

    def __init__(self):

        print("Loading DeepFilterNet...")

        self.model = DeepFilterNetStreaming()

        self.input_buffer = np.zeros(
            0,
            dtype=np.float32
        )

        self.output_buffer = np.zeros(
            0,
            dtype=np.float32
        )

        print("DeepFilterNet loaded.")

    def process(self, frame_160):

        frame_160 = np.asarray(
            frame_160,
            dtype=np.float32
        )

        if len(frame_160) != INPUT_BLOCK_SIZE:

            raise ValueError(
                f"Expected {INPUT_BLOCK_SIZE} samples, "
                f"got {len(frame_160)}"
            )

        # Add this 160-sample block to the model input buffer.
        self.input_buffer = np.concatenate(
            (
                self.input_buffer,
                frame_160
            )
        )

        # DeepFilterNet processes 480 samples.
        while len(self.input_buffer) >= MODEL_BLOCK_SIZE:

            model_input = self.input_buffer[
                :MODEL_BLOCK_SIZE
            ]

            self.input_buffer = self.input_buffer[
                MODEL_BLOCK_SIZE:
            ]

            model_output = self.model.process_chunk(
                model_input
            )

            if len(model_output) > 0:

                self.output_buffer = np.concatenate(
                    (
                        self.output_buffer,
                        np.asarray(
                            model_output,
                            dtype=np.float32
                        )
                    )
                )

        # Our application always needs exactly 160 samples.
        if len(self.output_buffer) >= INPUT_BLOCK_SIZE:

            output = self.output_buffer[
                :INPUT_BLOCK_SIZE
            ]

            self.output_buffer = self.output_buffer[
                INPUT_BLOCK_SIZE:
            ]

        else:

            output = np.zeros(
                INPUT_BLOCK_SIZE,
                dtype=np.float32
            )

        return output

    def close(self):

        self.model.close()


# ------------------------------------------------------------
# Compatibility interface for the rest of the project
# ------------------------------------------------------------

_default_enhancer = None


def enhance_frame(frame_160, state):

    global _default_enhancer

    if state is None:

        if _default_enhancer is None:

            _default_enhancer = DeepFilterAdapter()

        state = _default_enhancer

    output = state.process(frame_160)

    return output, state


def close_default_enhancer():

    global _default_enhancer

    if _default_enhancer is not None:

        _default_enhancer.close()

        _default_enhancer = None