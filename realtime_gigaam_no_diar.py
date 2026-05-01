#!/usr/bin/env python3
"""
Real-time Russian Speech Recognition with Sber GigaAM v3_e2e_rnnt

This script provides low-latency speech-to-text transcription from your
microphone using Sber's open-source GigaAM model with punctuation and
text normalization support.

=== Prerequisites ===
1. Python 3.10+
2. Install ffmpeg and a CUDA-enabled PyTorch build if you want GPU inference on Windows
3. Clone and install GigaAM:
   git clone https://github.com/salute-developers/GigaAM.git
   cd GigaAM
   pip install -e .
4. Install other dependencies:
   pip install -r requirements.txt

=== Usage ===
python realtime_gigaam.py
python realtime_gigaam.py --device cuda --block-duration 0.5

Press Ctrl+C to stop.

=== Design Notes ===
- Uses a rolling audio buffer (last N seconds) for context
- Periodically runs full-buffer transcription (pseudo-streaming)
- Compares transcriptions to show only new/changed content
- Uses CUDA automatically on Windows when a CUDA-enabled PyTorch build is available
- Optimized for Apple Silicon with MPS (Metal) acceleration

"""

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform colored output
colorama_init()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for real-time transcription."""
    
    # Audio settings
    sample_rate: int = 16000           # Hz, required by GigaAM
    channels: int = 1                  # Mono audio
    block_duration: float = 0.5        # Seconds per audio callback block
    
    # Buffer settings
    max_buffer_duration: float = 10.0  # Keep last N seconds in buffer
    
    # Transcription settings
    update_interval: float = 0.5       # Re-run transcription every N seconds of new audio
    
    # Device settings
    device: str = "auto"               # "auto", "cuda", "gpu", "mps", "cpu"
    
    @property
    def block_size(self) -> int:
        """Number of samples per audio block."""
        return int(self.sample_rate * self.block_duration)
    
    @property
    def max_buffer_samples(self) -> int:
        """Maximum samples to keep in rolling buffer."""
        return int(self.sample_rate * self.max_buffer_duration)
    
    @property
    def update_samples(self) -> int:
        """Samples threshold to trigger transcription update."""
        return int(self.sample_rate * self.update_interval)


# ============================================================================
# DEVICE DETECTION
# ============================================================================

def print_cuda_diagnostics() -> None:
    """Print enough CUDA state to explain why GPU was or was not selected."""
    print(f"{Fore.CYAN}CUDA diagnostics:{Style.RESET_ALL}")
    print(f"  torch: {torch.__version__}")
    print(f"  torch.version.cuda: {torch.version.cuda}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")


def get_device(requested: str = "auto") -> torch.device:
    """
    Determine the best available device for inference.
    
    Args:
        requested: "auto", "cuda", "gpu", "mps", or "cpu"
    
    Returns:
        torch.device for model and tensor placement
    """
    requested = requested.lower()

    if requested == "cpu":
        return torch.device("cpu")

    if requested in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"{Fore.GREEN}CUDA acceleration available: {device_name}{Style.RESET_ALL}")
            return torch.device("cuda")
        if requested in ("cuda", "gpu"):
            print(f"{Fore.RED}CUDA was requested, but PyTorch cannot use CUDA in this environment.{Style.RESET_ALL}")
            print_cuda_diagnostics()
            print(
                f"{Fore.YELLOW}Install a CUDA-enabled torch build in the venv, then run again.{Style.RESET_ALL}"
            )
            sys.exit(1)
        print(f"{Fore.YELLOW}CUDA is not available to PyTorch; checking other accelerators.{Style.RESET_ALL}")
        print_cuda_diagnostics()
    
    if requested in ("auto", "mps"):
        if torch.backends.mps.is_available():
            print(f"{Fore.GREEN}✓ MPS (Metal) acceleration available{Style.RESET_ALL}")
            return torch.device("mps")
        elif requested == "mps":
            print(f"{Fore.YELLOW}⚠ MPS requested but not available, falling back to CPU{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}ℹ Using CPU for inference{Style.RESET_ALL}")
    return torch.device("cpu")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_gigaam_model(device: torch.device):
    """
    Load the GigaAM v3_e2e_rnnt model.
    
    Args:
        device: Target device for the model
    
    Returns:
        Loaded GigaAM model on the specified device
    
    Note:
        The v3_e2e_rnnt model supports punctuation and text normalization.
        Model weights are automatically downloaded to ~/.cache/gigaam/
    """
    # try:
    #     import gigaam
    # except ImportError:
    #     print(f"{Fore.RED}✗ GigaAM not found. Please install it:{Style.RESET_ALL}")
    #     print("  git clone https://github.com/salute-developers/GigaAM.git")
    #     print("  cd GigaAM && pip install -e .")
    #     sys.exit(1)

    
    import gigaam
    
    
    print(f"{Fore.CYAN}Loading GigaAM v3_e2e_rnnt model...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}(First run will download ~1GB model weights){Style.RESET_ALL}")
    
    try:
        # load_model handles device placement and fp16 conversion automatically
        # fp16_encoder=True for faster inference on CUDA GPU (default)
        # For MPS, we disable fp16 as it can cause issues on some systems
        fp16 = device.type != "mps"
        model = gigaam.load_model("v3_e2e_rnnt", device=device, fp16_encoder=fp16)
        print(f"{Fore.GREEN}✓ Model loaded successfully on {device}{Style.RESET_ALL}")
        return model
    except Exception as e:
        print(f"{Fore.RED}✗ Failed to load model: {e}{Style.RESET_ALL}")
        sys.exit(1)


# ============================================================================
# AUDIO BUFFER
# ============================================================================

class RollingAudioBuffer:
    """
    Thread-safe rolling buffer for audio samples.
    
    Maintains a fixed-size buffer of the most recent audio,
    discarding old samples as new ones arrive.
    """
    
    def __init__(self, max_samples: int, sample_rate: int):
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.samples_since_update = 0
    
    def append(self, audio: np.ndarray) -> None:
        """Add new audio samples to the buffer."""
        with self.lock:
            # Flatten and ensure float32
            audio_flat = audio.flatten().astype(np.float32)
            
            # Append new audio
            self.buffer = np.concatenate([self.buffer, audio_flat])
            self.samples_since_update += len(audio_flat)
            
            # Trim to max size (keep recent samples)
            if len(self.buffer) > self.max_samples:
                self.buffer = self.buffer[-self.max_samples:]
    
    def get_audio(self) -> np.ndarray:
        """Get a copy of the current buffer."""
        with self.lock:
            return self.buffer.copy()
    
    def get_samples_since_update(self) -> int:
        """Get count of new samples since last reset."""
        with self.lock:
            return self.samples_since_update
    
    def reset_update_counter(self) -> None:
        """Reset the new samples counter."""
        with self.lock:
            self.samples_since_update = 0
    
    def get_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self.lock:
            return len(self.buffer) / self.sample_rate


# ============================================================================
# TRANSCRIPTION ENGINE
# ============================================================================

class TranscriptionEngine:
    """
    Handles transcription using GigaAM model.
    
    Implements pseudo-streaming by running the model on the rolling buffer
    and tracking changes between transcriptions.
    
    Note: GigaAM's transcribe() expects file paths, so we use the internal
    forward() and decoding methods directly for in-memory audio processing.
    """
    
    def __init__(self, model, device: torch.device, sample_rate: int):
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.last_transcription = ""
        
        # Get model's dtype for proper tensor conversion
        self.dtype = next(model.parameters()).dtype
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe audio samples using GigaAM's internal methods.
        
        Args:
            audio: 1D numpy array of float32 samples at 16kHz
        
        Returns:
            Transcribed text or None if audio too short
        """
        if len(audio) < self.sample_rate * 0.3:  # Min 0.3 seconds
            return None
        
        try:
            with torch.inference_mode():
                # Convert numpy array to torch tensor
                # Shape: (1, num_samples) - batch of 1
                wav = torch.from_numpy(audio).to(self.device).to(self.dtype).unsqueeze(0)
                
                # Create length tensor
                length = torch.tensor([wav.shape[-1]], device=self.device)
                
                # Run encoder forward pass
                encoded, encoded_len = self.model.forward(wav, length)
                
                # Decode using the model's decoding module
                result = self.model.decoding.decode(self.model.head, encoded, encoded_len)[0]
                
                return result.strip() if result else ""
                
        except Exception as e:
            print(f"\n{Fore.RED}Transcription error: {e}{Style.RESET_ALL}")
            return None
    
    def get_update(self, new_text: str) -> tuple[str, bool]:
        """
        Compare new transcription with previous one.
        
        Returns:
            Tuple of (text_to_display, is_changed)
        """
        if new_text == self.last_transcription:
            return new_text, False
        
        self.last_transcription = new_text
        return new_text, True


# ============================================================================
# AUDIO STREAM
# ============================================================================

class AudioStreamHandler:
    """Handles microphone input streaming."""
    
    def __init__(self, config: Config, audio_queue: queue.Queue):
        self.config = config
        self.audio_queue = audio_queue
        self.stream: Optional[sd.InputStream] = None
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info, status) -> None:
        """Callback for sounddevice InputStream."""
        if status:
            print(f"\n{Fore.YELLOW}Audio status: {status}{Style.RESET_ALL}")
        
        # Put audio data into queue (copy to avoid buffer reuse issues)
        self.audio_queue.put(indata.copy())
    
    def start(self) -> None:
        """Start the audio input stream."""
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.block_size,
                dtype=np.float32,
                callback=self._audio_callback
            )
            self.stream.start()
        except sd.PortAudioError as e:
            print(f"{Fore.RED}✗ Microphone error: {e}{Style.RESET_ALL}")
            print("Please ensure microphone access is granted in System Settings.")
            sys.exit(1)
    
    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# ============================================================================
# DISPLAY
# ============================================================================

def clear_line() -> None:
    """Clear the current console line."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def display_transcription(text: str, is_update: bool = True) -> None:
    """Display transcription in the console."""
    clear_line()
    if text:
        prefix = f"{Fore.GREEN}▶{Style.RESET_ALL} " if is_update else "  "
        # Truncate for display if too long
        max_width = 100
        display_text = text if len(text) <= max_width else f"...{text[-(max_width-3):]}"
        sys.stdout.write(f"{prefix}{display_text}")
    else:
        sys.stdout.write(f"{Fore.CYAN}(listening...){Style.RESET_ALL}")
    sys.stdout.flush()


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_realtime_transcription(config: Config) -> None:
    """
    Main loop for real-time transcription.
    
    This implements a pseudo-streaming approach:
    1. Audio is captured in small blocks via callback
    2. Blocks are accumulated in a rolling buffer
    3. When enough new audio arrives, we run transcription on the full buffer
    4. Results are compared with previous transcription to show updates
    
    Trade-off: Shorter update_interval = lower latency but more CPU usage.
    The rolling buffer provides context for punctuation and normalization.
    """
    # Determine device
    device = get_device(config.device)
    print(f"{Fore.CYAN}Resolved inference device: {device}{Style.RESET_ALL}")
    
    # Load model
    model = load_gigaam_model(device)
    
    # Initialize components
    audio_queue: queue.Queue = queue.Queue()
    audio_buffer = RollingAudioBuffer(config.max_buffer_samples, config.sample_rate)
    transcription_engine = TranscriptionEngine(model, device, config.sample_rate)
    stream_handler = AudioStreamHandler(config, audio_queue)
    
    # Start audio capture
    stream_handler.start()
    
    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}🎤 Listening from microphone... Press Ctrl+C to stop.{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")
    
    try:
        last_transcription_time = time.time()
        
        while True:
            # Collect audio from queue (non-blocking with timeout)
            try:
                while True:
                    audio_chunk = audio_queue.get(timeout=0.01)
                    audio_buffer.append(audio_chunk)
            except queue.Empty:
                pass
            
            # Check if we should run transcription
            new_samples = audio_buffer.get_samples_since_update()
            current_time = time.time()
            
            # Run transcription if enough new audio OR timeout
            should_transcribe = (
                new_samples >= config.update_samples or
                (current_time - last_transcription_time >= config.update_interval * 1.5 
                 and new_samples > 0)
            )
            
            if should_transcribe:
                audio_data = audio_buffer.get_audio()
                
                if len(audio_data) > 0:
                    text = transcription_engine.transcribe(audio_data)
                    
                    if text is not None:
                        display_text, is_changed = transcription_engine.get_update(text)
                        display_transcription(display_text, is_changed)
                    
                    audio_buffer.reset_update_counter()
                    last_transcription_time = current_time
            
            # Small sleep to prevent busy-waiting
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Stopping...{Style.RESET_ALL}")
    
    finally:
        stream_handler.stop()
        
        # Print final transcription
        final_text = transcription_engine.last_transcription
        if final_text:
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Final transcription:{Style.RESET_ALL}")
            print(f"{final_text}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}👋 Goodbye!{Style.RESET_ALL}")


# ============================================================================
# CLI ARGUMENTS
# ============================================================================

# def parse_args() -> Config:
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Real-time Russian speech recognition with GigaAM",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python realtime_gigaam.py
#   python realtime_gigaam.py --device cuda
#   python realtime_gigaam.py --device mps
#   python realtime_gigaam.py --block-duration 0.3 --update-interval 0.5
#         """
#     )
    
#     parser.add_argument(
#         "--device",
#         type=str,
#         choices=["auto", "cuda", "mps", "cpu"],
#         default="auto",
#         help="Compute device: auto (default), cuda, mps (Metal), or cpu"
#     )
    
#     parser.add_argument(
#         "--block-duration",
#         type=float,
#         default=0.5,
#         help="Audio block duration in seconds (default: 0.5)"
#     )
    
#     parser.add_argument(
#         "--max-buffer-duration",
#         type=float,
#         default=10.0,
#         help="Maximum audio buffer duration in seconds (default: 10.0)"
#     )
    
#     parser.add_argument(
#         "--update-interval",
#         type=float,
#         default=0.8,
#         help="Transcription update interval in seconds (default: 0.8)"
#     )
    
#     args = parser.parse_args()
    
#     return Config(
#         device=args.device,
#         block_duration=args.block_duration,
#         max_buffer_duration=args.max_buffer_duration,
#         update_interval=args.update_interval
#     )


def parse_args() -> Config:
    base = Config()  # <-- берём дефолты из dataclass

    parser = argparse.ArgumentParser(...)
    parser.add_argument("--device", choices=["auto","cuda","gpu","mps","cpu"], default=base.device)
    parser.add_argument("--block-duration", type=float, default=base.block_duration)
    parser.add_argument("--max-buffer-duration", type=float, default=base.max_buffer_duration)
    parser.add_argument("--update-interval", type=float, default=base.update_interval)

    args = parser.parse_args()

    return Config(
        device=args.device,
        block_duration=args.block_duration,
        max_buffer_duration=args.max_buffer_duration,
        update_interval=args.update_interval,
        sample_rate=base.sample_rate,  
        channels=base.channels
    )


def main() -> None:
    """Main entry point."""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}  Real-time Russian Speech Recognition with GigaAM{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    config = parse_args()
    
    # Display configuration
    print(f"{Fore.CYAN}Configuration:{Style.RESET_ALL}")
    print(f"  • Device: {config.device}")
    print(f"  • Sample rate: {config.sample_rate} Hz")
    print(f"  • Block duration: {config.block_duration} s")
    print(f"  • Buffer duration: {config.max_buffer_duration} s")
    print(f"  • Update interval: {config.update_interval} s")
    print()
    
    run_realtime_transcription(config)


if __name__ == "__main__":
    main()
