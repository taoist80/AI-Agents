"""
Meeting Recording and Analysis Tool
---------------------------------
CLI tool for analyzing meeting audio files or recording new audio.

Usage:
    python meeting_recorder.py --file path/to/audio.wav  # Analyze existing file
    python meeting_recorder.py --record 60  # Record for 60 seconds
    python meeting_recorder.py --help  # Show help

Dependencies:
    - pyaudio
    - wave
    - speech_recognition
    - google.cloud.speech
    - anthropic
    - openai
    - requests
"""

import logging
import os
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import pyaudio
import wave
import speech_recognition as sr
from anthropic import Anthropic
import openai
import requests
from datetime import datetime

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
        """
        Record audio using PyAudio

        Returns:
            Path: Path to the recorded audio file

        Raises:
            OSError: If there's an error accessing the audio device
            IOError: If there's an error saving the audio file
        """
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
    """Handles audio transcription using Google Speech Recognition"""

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_file: Path) -> Optional[str]:
        """
        Transcribe audio file using Google Speech Recognition

        Args:
            audio_file: Path to the audio file

        Returns:
            Optional[str]: Transcribed text or None if transcription fails
        """
        logger.info(f"Starting transcription of {audio_file}")
        try:
            with sr.AudioFile(str(audio_file)) as source:
                audio_data = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio_data)
            logger.info("Transcription completed successfully")
            return text

        except sr.UnknownValueError:
            logger.error("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
            return None

class IncidentAnalyzer:
    """Analyzes transcribed text using AI services (OpenAI/Claude)"""

    def __init__(self, openai_api_key: str, anthropic_api_key: str):
        self.openai_client = openai.Client(api_key=openai_api_key)
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

    def analyze(self, transcript: str, retry: bool = True) -> Optional[str]:
        """
        Analyze transcript using AI services with fallback

        Args:
            transcript: Text to analyze
            retry: Whether to retry with Claude if OpenAI fails

        Returns:
            Optional[str]: Analysis results or None if both services fail
        """
        try:
            content = self._analyze_with_openai(transcript)
            logger.info("OpenAI analysis completed successfully")
            return content
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            if retry:
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

class XMattersNotifier:
    """Handles sending analysis results to xMatters"""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

    def send_notification(self, content: str) -> bool:
        """
        Send analysis results to xMatters

        Args:
            content: Analysis content to send

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "properties": {
                    "title": f"Incident Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "analysis": content
                },
                "recipients": [{"targetName": "Incident Response Team"}]
            }

            response = requests.post(self.url, headers=headers, json=payload)

            if response.status_code == 202:
                logger.info("Analysis sent to xMatters successfully")
                return True
            else:
                logger.error(f"xMatters API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending to xMatters: {e}")
            return False

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
        help="Path to existing audio file to analyze"
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
        "--no-xmatters",
        action="store_true",
        help="Skip sending results to xMatters"
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

def main():
    """Main execution function"""
    args = parse_args()

    try:
        # Initialize services
        transcriber = TranscriptionService()
        analyzer = IncidentAnalyzer(
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("ANTHROPIC_API_KEY", "")
        )

        if not args.no_xmatters:
            notifier = XMattersNotifier(
                os.getenv("XMATTERS_URL", ""),
                os.getenv("XMATTERS_API_KEY", "")
            )

        # Get audio file path
        if args.file:
            audio_path = Path(args.file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.info(f"Using existing audio file: {audio_path}")
        else:
            # Record new audio
            config = AudioConfig(seconds=args.record)
            recorder = AudioRecorder(config)
            audio_path = recorder.record()
            logger.info(f"Recorded new audio file: {audio_path}")

        # Process audio
        if transcript := transcriber.transcribe(audio_path):
            logger.info("Transcription successful")
            if analysis := analyzer.analyze(transcript):
                logger.info("Analysis successful")

                # Save analysis to file
                save_analysis(analysis, args.output)

                # Send to xMatters if enabled
                if not args.no_xmatters:
                    notifier.send_notification(analysis)
            else:
                logger.error("Analysis failed")
        else:
            logger.error("Transcription failed")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()
