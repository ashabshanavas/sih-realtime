# SIH 2026 — Real-Time Adaptive Noise Cancellation

AI/ML-enabled adaptive noise cancellation system for defence communication environments.

The system is designed to suppress **stationary, non-stationary, and impulsive noise** while preserving speech intelligibility in real time on embedded hardware.

> **Current status:** Real-time laptop simulation and DeepFilterNet integration are implemented. Hardware/Jetson deployment and the final custom neural model are future stages.

---

## 1. System Architecture

The intended system is:

```text
                    PRIMARY MICROPHONE
                           │
                           ▼
                  ┌──────────────────┐
                  │ Transient Limiter│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Neural Denoiser  │
                  │  (DeepFilterNet  │
                  │   / Own Model)   │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
REFERENCE MIC ───►│   NLMS Adaptive  │
                  │      Filter      │
                  └────────┬─────────┘
                           │
                           ▼
                    ENHANCED SPEECH
                           │
                           ▼
                    HEADSET / OUTPUT
```

The current laptop simulation primarily tests:

```text
Microphone
    ↓
Streaming audio blocks
    ↓
DeepFilterNet
    ↓
Speaker / Headphones
```

The transient limiter and two-microphone NLMS stage will be integrated into the complete pipeline in the next development stage.

---

# 2. Current Simulation

The current simulation runs on a laptop and provides:

* Real-time microphone capture
* Real-time audio output
* 16 kHz application audio pipeline
* 160-sample blocks
* 10 ms application blocks
* Streaming state handling
* DeepFilterNet integration
* Processing-time measurement
* Real-Time Factor (RTF) measurement
* Raw/enhanced pipeline testing

The current simulation does **not** yet represent the final embedded hardware system.

---

# 3. Requirements

## Hardware

For the current simulation:

* macOS, Linux, or Windows
* Microphone
* Speakers or headphones
* Python-capable computer

For the final system:

* Jetson Orin Nano or selected edge board
* Two microphones
* USB audio interface with at least two input channels
* Headset/headphones

---

# 4. Software Requirements

Recommended:

* Python 3.11–3.13
* Git
* pip
* virtual environment

Current development environment was tested with:

```text
Python 3.13.5
NumPy 2.5.3
sounddevice 0.5.6
```

---

# 5. Clone the Repository

Clone the project:

```bash
git clone <REPOSITORY_URL>
cd sih
```

Replace `<REPOSITORY_URL>` with the team's actual Git repository URL.

---

# 6. Create the Python Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

After activation, the terminal should show:

```text
(.venv)
```

---

# 7. Install Dependencies

Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install numpy sounddevice
python -m pip install dfnstream-py
```

The main packages are:

| Package        | Purpose                              |
| -------------- | ------------------------------------ |
| `numpy`        | Numerical/audio processing           |
| `sounddevice`  | Real-time microphone and speaker I/O |
| `dfnstream-py` | DeepFilterNet streaming denoiser     |

---

# 8. Check Audio Devices

Run:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Example:

```text
0 MacBook Air Microphone, Core Audio (1 in, 0 out)
1 MacBook Air Speakers, Core Audio (0 in, 2 out)
```

The device numbers may be different on another computer.

If they are different, update the device numbers in the relevant real-time script.

---

# 9. Run the Basic Pass-Through Simulation

The simplest real-time test sends the microphone directly to the output.

Run:

```bash
python realtime/pass_through.py
```

This tests:

```text
Microphone
    ↓
sounddevice
    ↓
Output
```

No denoising is performed.

Use this to verify that the real-time audio path works before testing the neural model.

---

# 10. Run the DeepFilterNet Simulation

The current main neural-denoising simulation is:

```bash
python realtime/live_deepfilter.py
```

You should see:

```text
LIVE DEEPFILTERNET TEST

Sample rate : 16000
Block size  : 160
Block time  : 10.0 ms

Loading DeepFilterNet...
DeepFilterNet loaded.

Press Q then ENTER to stop.

Starting live audio...
Speak into the microphone.
```

Speak into the microphone.

When finished, press:

```text
q
```

and press Enter.

The program will report processing statistics such as:

```text
LIVE TEST RESULTS

Callbacks processed : ...
Audio per block     : 10.00 ms
Average processing  : ...
Minimum processing  : ...
Maximum processing  : ...
Real-Time Factor    : ...
```

---

# 11. Understanding the Performance Numbers

### Block size

The current application uses:

```text
160 samples
```

at:

```text
16,000 samples/second
```

Therefore:

```text
160 / 16000 = 0.01 seconds
```

or:

```text
10 ms
```

of audio per application block.

---

## Processing Time

Suppose the program reports:

```text
Average processing : 0.45 ms
```

This means the computer takes approximately 0.45 ms of processing time to process one application block.

It does **not** mean that the complete microphone-to-ear latency is 0.45 ms.

---

## Real-Time Factor

RTF is approximately:

```text
RTF = processing time / audio duration
```

For example:

```text
0.45 ms / 10 ms = 0.045
```

An RTF below 1 means the processing computation is faster than the amount of audio being processed.

---

# 12. Important: End-to-End Latency

The current laptop tests measure **processing performance**, not true physical end-to-end latency.

True end-to-end latency is approximately:

```text
Microphone
    ↓
Audio input buffer
    ↓
DSP
    ↓
Neural model
    ↓
Output buffer
    ↓
DAC
    ↓
Headphones
```

To measure this properly, the system needs a suitable hardware loopback setup or audio interface.

Do not report processing time as end-to-end latency.

---

# 13. DeepFilterNet Sample Rate Note

DeepFilterNet's streaming implementation operates at 48 kHz, while the current SIH application interface is designed around 16 kHz.

Therefore, the current DeepFilterNet integration is an **experimental integration**, not the final production audio architecture.

The final system must explicitly handle:

```text
16 kHz application audio
        ↓
proper streaming resampling
        ↓
48 kHz DeepFilterNet
        ↓
proper streaming resampling
        ↓
16 kHz application audio
```

Alternatively, the final custom SIH model can be designed directly for the project's target sample rate.

Sample rate must always be considered when integrating the final model.

---

# 14. Real-Time Model Interface

The project uses a model-independent interface:

```python
enhance_frame(frame_160, state)
```

The expected behavior is:

```text
Input:
    frame_160
        ↓
    160 audio samples

State:
    model's persistent streaming state

Output:
    out_160
        ↓
    160 enhanced audio samples

    state
        ↓
    updated model state
```

Conceptually:

```python
output, state = enhance_frame(frame, state)
```

This allows DeepFilterNet to be replaced later by the team's own neural model without rewriting the real-time audio pipeline.

---

# 15. Current File Structure

```text
realtime/
│
├── pass_through.py
│   Basic microphone → output test
│
├── latency_test.py
│   Measures basic callback processing time
│
├── enhancer.py
│   Model-independent enhancer interface
│
├── realtime_pipeline.py
│   Generic real-time streaming pipeline
│
├── toggle_pipeline.py
│   Raw/enhanced pipeline switching
│
├── deepfilter_adapter.py
│   Adapter between the application interface
│   and DeepFilterNet streaming
│
├── test_deepfilter.py
│   Basic DeepFilterNet loading test
│
├── test_deepfilter_stream.py
│   DeepFilterNet streaming/buffering test
│
├── test_deepfilter_adapter.py
│   Adapter input/output test
│
├── live_deepfilter.py
│   Main live DeepFilterNet simulation
│
└── deepfilter_latency.py
    Experimental streaming delay test
```

---

# 16. Recommended Commands

For someone who has already cloned the repository, the important commands are:

### Check devices

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Test basic audio

```bash
python realtime/pass_through.py
```

### Run neural denoising simulation

```bash
python realtime/live_deepfilter.py
```

---

# 17. Development Roadmap

## Phase 1 — Real-Time Audio

```text
Microphone
    ↓
sounddevice
    ↓
10 ms blocks
    ↓
Output
```

**Status: Completed**

---

## Phase 2 — Neural Denoiser

```text
Microphone
    ↓
Streaming interface
    ↓
DeepFilterNet
    ↓
Output
```

**Status: Experimental integration completed**

Remaining:

* Correct sample-rate handling
* Final model selection
* Streaming latency validation

---

## Phase 3 — Classical DSP

Implement:

### Transient limiter

For impulsive noise such as:

* gunshots
* sharp mechanical impulses
* other high-amplitude transients

### NLMS adaptive filter

For correlated noise using:

```text
Primary microphone
        +
Reference microphone
        ↓
      NLMS
        ↓
Residual signal
```

---

## Phase 4 — Full System

Target architecture:

```text
                 PRIMARY MIC
                     │
                     ▼
              Transient Limiter
                     │
                     ▼
               Neural Denoiser
                     │
                     ▼
                NLMS Filter ◄──── Reference Mic
                     │
                     ▼
                  OUTPUT
                     │
                     ▼
                  HEADSET
```

The pipeline should support switching between:

```text
RAW
LIMITER
NEURAL
NLMS
FULL
```

for demonstrations and evaluation.

---

# 18. Evaluation

The final evaluation system should measure:

* Input SNR
* Output SNR
* SNR improvement
* STOI
* PESQ
* SI-SDR
* Processing time
* Real-Time Factor
* End-to-end latency

Evaluation must be performed separately for different noise categories and input SNR levels.

Example:

```text
Noise Type       Input SNR       Output SNR
------------------------------------------------
Gunshot             0 dB             ...
Helicopter          0 dB             ...
Engine              5 dB             ...
Drone               5 dB             ...
Wind               10 dB             ...
Sirens              0 dB             ...
```

**Never manually enter evaluation numbers.**

All reported metrics must come directly from the evaluation scripts.

---

# 19. Dataset and Test-Set Rule

The test set must remain frozen.

```text
Training Data
     │
     ▼
Training / Validation
     │
     ▼
Model
     │
     ▼
FROZEN TEST SET
     │
     ▼
Final Metrics
```

Never train on the final test set.

Evaluation should cover:

* stationary noise
* non-stationary noise
* impulsive noise
* multiple input SNR conditions

---

# 20. Embedded Deployment — Future

Once the neural model is finalized:

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime
   ↓
TensorRT
   ↓
FP16 / INT8
   ↓
Jetson
```

The embedded system will then use:

```text
2-channel audio input
        ↓
Jetson
        ↓
Real-time processing
        ↓
Headset
```

Performance measurements will include:

* latency
* RTF
* RAM usage
* GPU usage
* power consumption
* thermal behavior

---

# 21. Troubleshooting

## No microphone detected

Run:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Check that the selected input device has at least one input channel.

On macOS, make sure Terminal/Python has microphone permission:

```text
System Settings
→ Privacy & Security
→ Microphone
```

---

## No sound from output

Check:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Make sure the selected output device has output channels.

Also check the system volume and selected audio output.

---

## Python command not found

Try:

```bash
python3 --version
```

and create the environment using:

```bash
python3 -m venv .venv
```

---

## Virtual environment not active

Run:

```bash
source .venv/bin/activate
```

The terminal should show:

```text
(.venv)
```

---

## DeepFilterNet fails to load

Make sure the virtual environment is active and run:

```bash
python -m pip install dfnstream-py
```

Then test:

```bash
python realtime/test_deepfilter.py
```

---

# 22. Important Development Principle

The real-time callback must remain lightweight.

Avoid putting expensive operations such as:

```text
print()
file I/O
model loading
large memory allocations
network requests
```

inside the audio callback.

The final architecture should use persistent model state and preallocated buffers wherever possible.

---

# 23. Current Status

### Completed (by 26th)

* [x] Python virtual environment
* [x] Audio device detection
* [x] Real-time audio capture
* [x] Real-time output
* [x] 16 kHz / 160-sample application interface
* [x] Pass-through pipeline
* [x] Streaming model interface
* [x] DeepFilterNet integration
* [x] Streaming adapter
* [x] Raw/enhanced switching
* [x] Processing-time measurement
* [x] RTF measurement

### In Progress

* [ ] Correct 16 kHz ↔ 48 kHz DeepFilterNet integration
* [ ] Transient limiter
* [ ] NLMS adaptive filter
* [ ] Two-microphone simulation
* [ ] Full pipeline
* [ ] Automated evaluation
* [ ] Frozen test set
* [ ] Custom neural model

### Future

* [ ] ONNX export
* [ ] TensorRT optimization
* [ ] Jetson deployment
* [ ] Two-channel USB audio interface
* [ ] Real microphones
* [ ] Hardware loopback latency measurement
* [ ] Final headset demonstration

---

# 24. Quick Start — TL;DR

For someone who has already cloned the repository:

```bash
cd sih

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy sounddevice dfnstream-py

python -c "import sounddevice as sd; print(sd.query_devices())"

python realtime/pass_through.py
```

If the pass-through test works, stop it and run:

```bash
python realtime/live_deepfilter.py
```

Speak into the microphone.

When finished:

```text
q + Enter
```

This runs the current **real-time neural denoising laptop simulation**.
