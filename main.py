"""
Meeting Recording and Analysis Tool
---------------------------------
CLI tool for analyzing meeting recordings (WAV, MP4, etc.) using OpenAI's Whisper and GPT/Claude.

Usage:
    python main.py --file path/to/video.mp4 --ai openai
    python main.py --record 60 --ai claude
"""

import logging
import os
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal
from pathlib import Path
import pyaudio
import wave
from anthropic import Anthropic
import openai
import requests
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import base64
from pydub import AudioSegment
import math
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm.asyncio import tqdm_asyncio


async def identify_attendees(transcript: str) -> List[str]:
    """TODO: Identify and extract list of attendees from transcript"""
    pass


async def detect_unclear_speech(audio_segment: AudioSegment) -> List[Dict[str, any]]:
    """TODO: Detect timestamps where speech is unclear or uninterpretable

    Returns:
        List of dicts containing:
        - timestamp: float (in seconds)
        - duration: float (length of unclear segment)
        - confidence: float (certainty of detection)
    """
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('meeting_recorder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioConverter:
    """Handles conversion of various audio/video formats to WAV"""
    
    @staticmethod
    def convert_to_wav(input_path: Path) -> Path:
        """
        Convert audio/video file to WAV format suitable for transcription
        
        Args:
            input_path: Path to input file
            
        Returns:
            Path: Path to converted WAV file
        """
        input_format = input_path.suffix.lower()
        
        # If already WAV, return original path
        if input_format == '.wav':
            return input_path
            
        try:
            # Create temporary WAV file
            temp_wav = Path(tempfile.gettempdir()) / f"temp_{input_path.stem}.wav"
            
            logger.info(f"Converting {input_format} to WAV...")
            
            if input_format in ['.mp4', '.avi', '.mov', '.mkv']:
                # Handle video files
                video = VideoFileClip(str(input_path))
                audio = video.audio
                audio.write_audiofile(
                    str(temp_wav),
                    fps=16000,  # Match our desired sample rate
                    nbytes=2,   # 16-bit audio
                    codec='pcm_s16le'  # Standard WAV codec
                )
                video.close()
                audio.close()
            elif input_format in ['.mp3', '.m4a', '.aac', '.ogg']:
                # Handle audio files
                audio = AudioFileClip(str(input_path))
                audio.write_audiofile(
                    str(temp_wav),
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le'
                )
                audio.close()
            else:
                raise ValueError(f"Unsupported file format: {input_format}")
                
            logger.info(f"Conversion successful: {temp_wav}")
            return temp_wav
            
        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            raise

@dataclass
class AudioConfig:
    """Audio recording configuration parameters"""
    chunk: int = 1024
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000
    seconds: int = 10
    output_file: str = "recorded_audio.wav"

class AudioRecorder:
    """Handles audio recording functionality"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate audio configuration parameters"""
        if self.config.seconds <= 0:
            raise ValueError("Recording duration must be positive")
        if self.config.rate <= 0:
            raise ValueError("Sample rate must be positive")
    
    def record(self) -> Path:
        """Record audio using PyAudio"""
        logger.info("Starting audio recording...")
        try:
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                frames_per_buffer=self.config.chunk
            )
            
            frames: List[bytes] = []
            for _ in range(0, int(self.config.rate / self.config.chunk * self.config.seconds)):
                frames.append(stream.read(self.config.chunk))
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            output_path = Path(self.config.output_file)
            self._save_audio(output_path, audio, frames)
            logger.info(f"Audio recorded successfully: {output_path}")
            return output_path
            
        except OSError as e:
            logger.error(f"Error recording audio: {e}")
            raise
    
    def _save_audio(self, path: Path, audio: pyaudio.PyAudio, frames: List[bytes]) -> None:
        """Save recorded audio frames to a WAV file"""
        try:
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.rate)
                wf.writeframes(b''.join(frames))
        except IOError as e:
            logger.error(f"Error saving audio file: {e}")
            raise


class TranscriptionService:
    """Handles audio transcription using OpenAI's Whisper model with async chunk processing"""

    def __init__(self, openai_client, anthropic_client=None, max_concurrent=3):
        """
        Initialize the transcription service

        Args:
            openai_client: OpenAI client instance
            anthropic_client: Optional Anthropic client instance
            max_concurrent: Maximum number of concurrent API calls
        """
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.converter = AudioConverter()
        self.max_file_size = 25 * 1024 * 1024  # 25MB in bytes
        self.max_concurrent = max_concurrent

    async def transcribe(self, audio_file: Path) -> Optional[str]:
        """Async transcription with chunk processing"""
        logger.info(f"Processing {audio_file}")
        try:
            # Convert to WAV if needed
            wav_file = self.converter.convert_to_wav(audio_file)

            # Check file size
            file_size = os.path.getsize(wav_file)
            logger.info(f"WAV file size: {file_size / (1024 * 1024):.2f} MB")

            try:
                # Try Whisper first
                try:
                    if file_size > self.max_file_size:
                        logger.info("File too large, splitting into chunks")
                        transcript = await self._transcribe_large_file(wav_file)
                    else:
                        transcript = await self._transcribe_single_file(wav_file)

                    if transcript:
                        return transcript
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")

                    # Try Claude fallback if available
                    if self.anthropic_client and file_size <= self.max_file_size:
                        logger.info("Attempting transcription with Claude")
                        return await self._transcribe_with_claude(wav_file)

                return None

            finally:
                # Clean up temporary file if it was created
                if wav_file != audio_file and wav_file.parent == Path(tempfile.gettempdir()):
                    wav_file.unlink()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def _transcribe_single_file(self, wav_file: Path) -> Optional[str]:
        """Transcribe a single file asynchronously"""
        with open(wav_file, "rb") as audio:
            # Use ThreadPoolExecutor for CPU-bound operations
            with ThreadPoolExecutor() as pool:
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    pool,
                    lambda: self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        language="en"
                    )
                )
                return transcript.text if transcript else None

    async def _transcribe_large_file(self, wav_file: Path) -> Optional[str]:
        """Handle transcription of large files with async chunk processing"""
        try:
            # Load audio file
            audio = AudioSegment.from_wav(str(wav_file))

            # Calculate chunk parameters
            total_size = os.path.getsize(wav_file)
            chunk_size_ratio = self.max_file_size / total_size * 0.8
            chunk_duration = len(audio) * chunk_size_ratio

            # Create chunks
            chunks = []
            for start in range(0, len(audio), int(chunk_duration)):
                end = min(start + int(chunk_duration), len(audio))
                chunk = audio[start:end]
                chunks.append(chunk)

            logger.info(f"Split audio into {len(chunks)} chunks")

            # Create temporary files for chunks
            chunk_files = []
            for i, chunk in enumerate(chunks, 1):
                temp_chunk = Path(tempfile.gettempdir()) / f"chunk_{i}.wav"
                chunk.export(str(temp_chunk), format='wav')
                chunk_files.append(temp_chunk)

            try:
                # Process chunks concurrently with rate limiting
                semaphore = asyncio.Semaphore(self.max_concurrent)
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for chunk_file in chunk_files:
                        task = asyncio.create_task(
                            self._process_chunk(chunk_file, semaphore)
                        )
                        tasks.append(task)

                    # Wait for all chunks to be processed
                    transcripts = await tqdm_asyncio.gather(*tasks, desc="Processing chunks")

                # Filter out None values and join transcripts
                valid_transcripts = [t for t in transcripts if t]
                if valid_transcripts:
                    return " ".join(valid_transcripts)
                return None

            finally:
                # Clean up chunk files
                for chunk_file in chunk_files:
                    chunk_file.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Error processing large file: {e}")
            return None

    async def _process_chunk(self, chunk_file: Path, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Process a single chunk with rate limiting"""
        async with semaphore:
            try:
                return await self._transcribe_single_file(chunk_file)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_file}: {e}")
                return None

    async def _transcribe_with_claude(self, audio_file: Path) -> Optional[str]:
        """Attempt transcription with Claude asynchronously"""
        try:
            with open(audio_file, "rb") as audio:
                # Use ThreadPoolExecutor for CPU-bound operations
                with ThreadPoolExecutor() as pool:
                    loop = asyncio.get_event_loop()
                    message = await loop.run_in_executor(
                        pool,
                        lambda: self.anthropic_client.messages.create(
                            model="claude-3-sonnet-20240229",
                            max_tokens=1500,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Please transcribe this audio file accurately, maintaining all speaker changes and important details:"
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "audio/wav",
                                            "data": base64.b64encode(audio.read()).decode()
                                        }
                                    }
                                ]
                            }]
                        )
                    )

                    if message.content:
                        logger.info("Claude transcription successful")
                        return message.content
            return None
        except Exception as e:
            logger.error(f"Claude transcription failed: {e}")
            return None

class IncidentAnalyzer:
    """Analyzes transcribed text using AI services (OpenAI/Claude)"""
    
    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.openai_client = openai.Client(api_key=openai_api_key)
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)
    
    def analyze(self, transcript: str, ai_service: Literal["auto", "openai", "claude"] = "auto") -> Optional[str]:
        """
        Analyze transcript using specified AI service
        
        Args:
            transcript: Text to analyze
            ai_service: Which AI service to use ("auto", "openai", or "claude")
            
        Returns:
            Optional[str]: Analysis results or None if analysis fails
        """
        if ai_service == "claude":
            return self._analyze_with_claude(transcript)
        elif ai_service == "openai":
            return self._analyze_with_openai(transcript)
        else:  # auto mode with fallback
            try:
                content = self._analyze_with_openai(transcript)
                logger.info("OpenAI analysis completed successfully")
                return content
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
                logger.info("Attempting analysis with Claude...")
                try:
                    content = self._analyze_with_claude(transcript)
                    logger.info("Claude analysis completed successfully")
                    return content
                except Exception as e:
                    logger.error(f"Claude analysis failed: {e}")
                return None
    
    def _analyze_with_openai(self, transcript: str) -> str:
        """Analyze transcript using OpenAI's GPT-4"""
        prompt = self._get_analysis_prompt(transcript)
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def _analyze_with_claude(self, transcript: str) -> str:
        """Analyze transcript using Anthropic's Claude"""
        prompt = self._get_analysis_prompt(transcript)
        message = self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content
    
    @staticmethod
    def _get_analysis_prompt(transcript: str) -> str:
        """Generate analysis prompt template"""
        return f"""Based on the following incident transcript, please provide:
1. Timeline of major events
2. List of services impacted
3. Initial root cause analysis
4. Mitigation steps

Transcript:
{transcript}"""

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Meeting Recording and Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Path to existing audio/video file to analyze"
    )
    group.add_argument(
        "--record",
        type=int,
        help="Record new audio for specified number of seconds"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_output.txt",
        help="Path to save analysis results (default: analysis_output.txt)"
    )
    
    parser.add_argument(
        "--ai",
        type=str,
        choices=["auto", "openai", "claude"],
        default="auto",
        help="Choose AI service (default: auto with fallback)"
    )
    
    return parser.parse_args()

def save_analysis(analysis: str, output_path: str) -> None:
    """Save analysis results to file"""
    try:
        with open(output_path, 'w') as f:
            f.write(analysis)
        logger.info(f"Analysis saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving analysis: {e}")
        raise


async def async_main():
    """Async main execution function"""
    args = parse_args()

    try:
        # Initialize clients
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY", ""))
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

        # Initialize services
        transcriber = TranscriptionService(openai_client, anthropic_client)
        analyzer = IncidentAnalyzer(
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("ANTHROPIC_API_KEY", "")
        )

        # Get audio file path
        if args.file:
            audio_path = Path(args.file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.info(f"Using existing file: {audio_path}")
        else:
            # Record new audio
            config = AudioConfig(seconds=args.record)
            recorder = AudioRecorder(config)
            audio_path = recorder.record()
            logger.info(f"Recorded new audio file: {audio_path}")

        # Process audio
        if transcript := await transcriber.transcribe(audio_path):
            logger.info("Transcription successful")
            print("\nTranscript:")
            print(transcript)

            if analysis := analyzer.analyze(transcript, args.ai):
                logger.info("Analysis successful")
                save_analysis(analysis, args.output)
                print("\nAnalysis saved to:", args.output)
            else:
                logger.error("Analysis failed")
        else:
            logger.error("Transcription failed")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


def main():
    """Entry point wrapper for async main"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()