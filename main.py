import logging
import os
import argparse
import json
import wave
import asyncio
import tempfile
import base64
import math
import subprocess
import threading
import time
import re
import queue
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, List, Dict, Literal, Union
from pathlib import Path
from urllib.parse import urlparse
import shutil

# Third-party imports
import soundcard as sc
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from anthropic import Anthropic
import openai
from tqdm.asyncio import tqdm_asyncio
import pyaudio

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

class MeetingURL:
    """Handles parsing and validation of meeting URLs"""

    @staticmethod
    def parse_url(url: str) -> tuple[str, str]:
        """
        Parse meeting URL to determine platform and meeting ID

        Args:
            url: Meeting URL for Zoom or Google Meet

        Returns:
            tuple: (platform, meeting_id)
        """
        parsed = urlparse(url)

        if "zoom.us" in parsed.netloc:
            meeting_id = re.search(r'/j/(\d+)', parsed.path)
            if meeting_id:
                return "zoom", meeting_id.group(1)
            raise ValueError("Invalid Zoom URL format")

        elif "meet.google.com" in parsed.netloc:
            meet_code = parsed.path.strip('/')
            if meet_code:
                return "google_meet", meet_code
            raise ValueError("Invalid Google Meet URL format")

        raise ValueError("Unsupported meeting platform")

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
    """Basic audio recorder for local recordings"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
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
        try:
            with wave.open(str(path), 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.rate)
                wf.writeframes(b''.join(frames))
        except IOError as e:
            logger.error(f"Error saving audio file: {e}")
            raise

class AudioRecorderStream:
    """Records system audio with timestamps for online meetings"""

    def __init__(self, output_path: str, sample_rate: int = 44100):
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.is_recording = False
        self._audio_queue = queue.Queue()
        self._recording_thread = None
        self.start_time = None

        try:
            self.audio_device = sc.default_speaker()
            logger.info(f"Using audio device: {self.audio_device.name}")
        except Exception as e:
            logger.error(f"Error initializing audio device: {e}")
            raise

    def start_recording(self):
        if self.is_recording:
            return

        self.is_recording = True
        self.start_time = datetime.now()
        self._recording_thread = threading.Thread(target=self._record)
        self._recording_thread.start()
        logger.info(f"Started recording at {self.start_time}")

    def stop_recording(self) -> Path:
        if not self.is_recording:
            return Path(self.output_path)

        self.is_recording = False
        if self._recording_thread:
            self._recording_thread.join()

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logger.info(f"Stopped recording. Duration: {duration:.2f} seconds")

        return self._save_recording()

    def _record(self):
        chunk_size = 1024

        with self.audio_device.recorder(samplerate=self.sample_rate) as mic:
            while self.is_recording:
                try:
                    timestamp = (datetime.now() - self.start_time).total_seconds()
                    audio_chunk = mic.record(numframes=chunk_size)
                    self._audio_queue.put((timestamp, audio_chunk))
                except Exception as e:
                    logger.error(f"Error recording audio chunk: {e}")
                    break

    def _save_recording(self) -> Path:
        try:
            with wave.open(self.output_path, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)

                timestamps = []
                while not self._audio_queue.empty():
                    timestamp, chunk = self._audio_queue.get()
                    timestamps.append(timestamp)
                    wav_file.writeframes(chunk.tobytes())

            timestamp_file = Path(self.output_path).with_suffix('.json')
            with open(timestamp_file, 'w') as f:
                json.dump({
                    'start_time': self.start_time.isoformat(),
                    'timestamps': timestamps,
                    'sample_rate': self.sample_rate
                }, f, indent=2)

            logger.info(f"Saved audio file: {self.output_path}")
            logger.info(f"Saved timestamps: {timestamp_file}")

            return Path(self.output_path)

        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            raise


class BrowserMeetingRecorder:
    """Handles browser automation for joining meetings using existing Chrome profile on macOS"""

    def __init__(self, url: str):
        self.url = url
        self.platform, self.meeting_id = MeetingURL.parse_url(url)
        self.driver = None

        # Use Profile 5 as your logged-in profile
        self.chrome_profile = "Profile 5"  # Changed to Profile 5
        self.user_data_dir = os.path.expanduser('~/Library/Application Support/Google/Chrome')
        self.profile_dir = os.path.join(self.user_data_dir, self.chrome_profile)

        # Verify Chrome directories and profiles
        self._verify_chrome_setup()

    def _verify_chrome_setup(self):
        """Verify Chrome installation and profile setup"""
        logger.info(f"Checking Chrome setup...")

        if not os.path.exists(self.user_data_dir):
            raise ValueError(f"Chrome user data directory not found: {self.user_data_dir}")

        # List all available profiles
        profiles = [d for d in os.listdir(self.user_data_dir)
                    if os.path.isdir(os.path.join(self.user_data_dir, d))]
        logger.info(f"Available Chrome profiles: {profiles}")

        # Verify our target profile exists
        if not os.path.exists(self.profile_dir):
            raise ValueError(f"Target profile directory not found: {self.profile_dir}")

        # Try to read profile preferences to verify it's a valid profile
        pref_file = os.path.join(self.profile_dir, 'Preferences')
        if os.path.exists(pref_file):
            try:
                with open(pref_file, 'r') as f:
                    prefs = json.load(f)
                    profile_name = prefs.get('profile', {}).get('name', 'Unknown')
                    logger.info(f"Found profile '{self.chrome_profile}' with name: {profile_name}")
            except Exception as e:
                logger.warning(f"Could not read profile preferences: {e}")

        logger.info(f"Using Chrome profile at: {self.profile_dir}")

    def join_meeting(self) -> bool:
        try:
            # Make sure no existing Chrome processes are using the profile
            try:
                subprocess.run(['pkill', '-f', 'Google Chrome'], check=False)
                time.sleep(2)
            except Exception:
                pass

            options = webdriver.ChromeOptions()

            # Use verified profile paths
            options.add_argument(f"--user-data-dir={self.user_data_dir}")
            options.add_argument(f"--profile-directory={self.chrome_profile}")

            # Debug logging for profile usage
            logger.info(f"Chrome user data directory: {self.user_data_dir}")
            logger.info(f"Chrome profile directory: {self.chrome_profile}")

            # Essential options only
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--use-fake-ui-for-media-stream")
            options.add_argument("--remote-debugging-port=0")

            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_experimental_option("prefs", {
                "profile.default_content_setting_values.media_stream_mic": 1,
                "profile.default_content_setting_values.media_stream_camera": 2,
                "profile.default_content_setting_values.notifications": 2
            })

            logger.info("Initializing Chrome with profile...")

            service = webdriver.ChromeService(
                log_output=os.path.join(os.getcwd(), "chromedriver.log")
            )

            # Start Chrome with Profile 5
            try:
                self.driver = webdriver.Chrome(
                    service=service,
                    options=options
                )

                # Verify we're using the correct profile
                self.driver.get('chrome://version')
                time.sleep(2)
                page_source = self.driver.page_source
                if self.chrome_profile in page_source:
                    logger.info("Successfully loaded Profile 5")
                else:
                    logger.warning("Could not verify Profile 5 was loaded")

            except Exception as e:
                logger.error(f"Failed to start Chrome: {e}")
                raise e

            # Configure window and timeouts
            self.driver.set_window_size(1200, 800)
            self.driver.set_page_load_timeout(60)
            self.driver.implicitly_wait(20)

            if self.platform == "zoom":
                return self._join_zoom()
            elif self.platform == "google_meet":
                return self._join_google_meet()

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
            return False

    def _get_chrome_version(self) -> str:
        """Get the installed Chrome version."""
        try:
            chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            output = subprocess.check_output([chrome_path, "--version"]).decode()
            version = output.strip().split()[-1]
            return version
        except Exception as e:
            logger.warning(f"Could not determine Chrome version: {e}")
            return "unknown"

    def _join_google_meet(self) -> bool:
        try:
            # Navigate directly to the meeting URL
            logger.info("Navigating to Google Meet...")
            self.driver.get(self.url)
            time.sleep(10)  # Wait for initial load

            # Capture initial page state
            logger.info("Capturing initial page state...")
            self._debug_page_state()

            # Log current URL
            logger.info(f"Current URL: {self.driver.current_url}")

            # Check for various possible states with more specific logging
            logger.info("Checking for page elements...")

            possible_elements = {
                "ready_to_join": "//div[text()='Ready to join?']",
                "join_now": "//span[contains(text(), 'Join now')]",
                "ask_to_join": "//span[contains(text(), 'Ask to join')]",
                "cant_join": "//h1[contains(text(), 'You can't join')]",
                "meeting_safe": "//div[contains(text(), 'Your meeting is safe')]",
                # Add more potential elements
                "generic_button": "//button",
                "meeting_code": "//div[contains(@data-meeting-code, '')]"
            }

            found_elements = {}
            for name, xpath in possible_elements.items():
                elements = self.driver.find_elements(By.XPATH, xpath)
                if elements:
                    found_elements[name] = len(elements)
                    logger.info(f"Found {len(elements)} {name} elements")

            if not found_elements:
                logger.error("No expected elements found on page")
                logger.info("Capturing page state after failure to find elements...")
                self._debug_page_state()

                # Try refreshing the page
                logger.info("Attempting page refresh...")
                self.driver.refresh()
                time.sleep(10)

                # Check again after refresh
                for name, xpath in possible_elements.items():
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    if elements:
                        found_elements[name] = len(elements)
                        logger.info(f"After refresh: Found {len(elements)} {name} elements")

                # Capture state after refresh
                logger.info("Capturing page state after refresh...")
                self._debug_page_state()

            # Check for access denial
            access_denied_elements = self.driver.find_elements(
                By.XPATH,
                "//*[contains(text(), 'You can't join') or contains(text(), 'Your meeting is safe')]"
            )

            if access_denied_elements:
                logger.error("Access denied to meeting")
                self._debug_page_state()  # Capture final state
                return False

            # If we get here and still haven't found any join buttons, try one last time
            if not any(k in found_elements for k in ['join_now', 'ask_to_join']):
                logger.info("No join buttons found, waiting longer...")
                time.sleep(10)

                join_buttons = self.driver.find_elements(
                    By.XPATH,
                    "//button[contains(., 'Join now') or contains(., 'Ask to join')]"
                )

                if join_buttons:
                    logger.info(f"Found {len(join_buttons)} join buttons after extended wait")
                else:
                    logger.error("Still no join buttons found")
                    self._debug_page_state()  # Capture final state
                    return False

            # Try to join if possible
            join_buttons = self.driver.find_elements(
                By.XPATH,
                "//button[contains(., 'Join now') or contains(., 'Ask to join')]"
            )

            if join_buttons:
                # Turn off camera and microphone first
                try:
                    for device in ['camera', 'microphone']:
                        buttons = self.driver.find_elements(
                            By.XPATH,
                            f"//button[contains(@aria-label, '{device}')]"
                        )
                        for button in buttons:
                            if button.is_displayed() and "Turn off" in button.get_attribute("aria-label"):
                                button.click()
                                logger.info(f"Turned off {device}")
                                time.sleep(1)
                except Exception as e:
                    logger.warning(f"Could not configure audio/video: {e}")

                # Try to click join button
                for button in join_buttons:
                    if button.is_displayed():
                        try:
                            logger.info("Attempting to click join button...")
                            button.click()
                            logger.info("Successfully clicked join button")
                            time.sleep(5)  # Wait after clicking

                            # Verify we're in the meeting
                            in_meeting = any([
                                len(self.driver.find_elements(By.CSS_SELECTOR, "[aria-label*='Meeting']")) > 0,
                                len(self.driver.find_elements(By.CSS_SELECTOR, "[data-self-name]")) > 0,
                                len(self.driver.find_elements(By.XPATH,
                                                              "//div[contains(text(), 'You're in the meeting')]")) > 0
                            ])

                            if in_meeting:
                                logger.info("Successfully joined meeting")
                                return True
                            else:
                                logger.warning("Join button clicked but not in meeting yet")
                                self._debug_page_state()  # Capture state after failed join attempt
                        except Exception as e:
                            logger.error(f"Error clicking join button: {e}")
                            continue

            logger.error("Could not join meeting after all attempts")
            self._debug_page_state()  # Capture final state
            return False

        except Exception as e:
            logger.error(f"Failed to join Google Meet: {e}")
            self._debug_page_state()  # Capture state on error
            return False

    def _debug_page_state(self):
        """Capture complete page state for debugging"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = Path("debug_logs")
            debug_dir.mkdir(exist_ok=True)

            # Save screenshot
            screenshot_path = debug_dir / f"meet_page_{timestamp}.png"
            self.driver.save_screenshot(str(screenshot_path))
            logger.info(f"Saved screenshot to {screenshot_path}")

            # Save complete page source
            source_path = debug_dir / f"page_source_{timestamp}.html"
            with open(source_path, "w", encoding='utf-8') as f:
                f.write(self.driver.page_source)
            logger.info(f"Saved page source to {source_path}")

            # Log all visible elements with their attributes
            elements_log_path = debug_dir / f"elements_{timestamp}.txt"
            with open(elements_log_path, "w", encoding='utf-8') as f:
                f.write("=== ALL PAGE ELEMENTS ===\n\n")

                # Get all elements
                elements = self.driver.find_elements(By.XPATH, "//*")
                for elem in elements:
                    try:
                        # Get element details
                        tag_name = elem.tag_name
                        text = elem.text
                        is_displayed = elem.is_displayed()
                        location = elem.location

                        # Get all attributes
                        attributes = self.driver.execute_script(
                            'var items = {}; for (var i = 0; i < arguments[0].attributes.length; i++) { items[arguments[0].attributes[i].name] = arguments[0].attributes[i].value }; return items;',
                            elem
                        )

                        # Write element details
                        f.write(f"Element: {tag_name}\n")
                        f.write(f"Text: {text}\n")
                        f.write(f"Displayed: {is_displayed}\n")
                        f.write(f"Location: {location}\n")
                        f.write("Attributes:\n")
                        for attr, value in attributes.items():
                            f.write(f"  {attr}: {value}\n")
                        f.write("\n---\n\n")
                    except Exception as e:
                        f.write(f"Error getting element details: {e}\n")
                        continue

            # Log specific elements we're looking for
            target_elements = {
                "Join buttons": "//button[contains(., 'Join now') or contains(., 'Ask to join')]",
                "Camera controls": "//button[contains(@aria-label, 'camera')]",
                "Microphone controls": "//button[contains(@aria-label, 'microphone')]",
                "Access denied message": "//*[contains(text(), 'You can't join')]",
                "Meeting safe message": "//div[contains(text(), 'Your meeting is safe')]",
                "Ready to join": "//div[text()='Ready to join?']",
                "Meeting header": "//h1",
                "All buttons": "//button",
                "All clickable elements": "//*[contains(@role, 'button')]"
            }

            elements_status_path = debug_dir / f"target_elements_{timestamp}.txt"
            with open(elements_status_path, "w", encoding='utf-8') as f:
                f.write("=== TARGET ELEMENTS STATUS ===\n\n")
                for name, xpath in target_elements.items():
                    elements = self.driver.find_elements(By.XPATH, xpath)
                    f.write(f"\n{name}:\n")
                    if elements:
                        for i, elem in enumerate(elements, 1):
                            try:
                                f.write(f"  {i}. Text: {elem.text}\n")
                                f.write(f"     Visible: {elem.is_displayed()}\n")
                                f.write(f"     Enabled: {elem.is_enabled()}\n")
                                f.write(f"     Location: {elem.location}\n")
                                f.write(f"     Size: {elem.size}\n")
                                f.write("     Attributes:\n")
                                attributes = self.driver.execute_script(
                                    'var items = {}; for (var i = 0; i < arguments[0].attributes.length; i++) { items[arguments[0].attributes[i].name] = arguments[0].attributes[i].value }; return items;',
                                    elem
                                )
                                for attr, value in attributes.items():
                                    f.write(f"       {attr}: {value}\n")
                            except Exception as e:
                                f.write(f"     Error getting element details: {e}\n")
                    else:
                        f.write("  No elements found\n")

            # Log current URL and page title
            state_path = debug_dir / f"page_state_{timestamp}.txt"
            with open(state_path, "w", encoding='utf-8') as f:
                f.write(f"Current URL: {self.driver.current_url}\n")
                f.write(f"Page Title: {self.driver.title}\n")
                f.write(f"Ready State: {self.driver.execute_script('return document.readyState')}\n")
                f.write(
                    f"Page Load Status: {self.driver.execute_script('return window.performance.timing.loadEventEnd > 0')}\n")

            logger.info(f"Saved complete debug information to {debug_dir}")
            return True

        except Exception as e:
            logger.error(f"Error capturing debug information: {e}")
            return False


class TranscriptionService:
    """Handles audio transcription using OpenAI's Whisper model"""

    def __init__(self, openai_client, anthropic_client=None, max_concurrent=3):
        self.openai_client = openai_client
        self.anthropic_client = anthropic_client
        self.max_file_size = 25 * 1024 * 1024  # 25MB
        self.max_concurrent = max_concurrent

    async def transcribe(self, audio_file: Path) -> Optional[Dict]:
        logger.info(f"Processing audio file: {audio_file}")
        try:
            # Check file size
            file_size = os.path.getsize(audio_file)
            logger.info(f"Audio file size: {file_size / (1024 * 1024):.2f} MB")

            # Determine processing strategy
            if file_size > self.max_file_size:
                logger.info("File too large, splitting into chunks")
                result = await self._transcribe_large_file(audio_file)
            else:
                result = await self._transcribe_single_file(audio_file)

            if result:
                logger.info(f"Transcription successful for file: {audio_file}")
                return result
            else:
                logger.error("Transcription failed: No valid result returned")
                return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def _transcribe_single_file(self, audio_file: Path) -> Optional[str]:
        with open(audio_file, "rb") as audio:
            try:
                transcript = await asyncio.to_thread(
                    lambda: self.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        language="en"
                    )
                )
                return transcript.text if transcript else None
            except Exception as e:
                logger.error(f"Error in single file transcription: {e}")
                return None

    async def _transcribe_large_file(self, audio_file: Path) -> Optional[Dict]:
        try:
            # Load and split audio
            audio = await asyncio.to_thread(lambda: sc.WaveObject.from_wave_file(str(audio_file)))
            total_frames = len(audio)
            chunk_size = int(total_frames * (self.max_file_size / os.path.getsize(audio_file)))

            chunks = []
            for i in range(0, total_frames, chunk_size):
                chunk = audio[i:min(i + chunk_size, total_frames)]
                chunks.append(chunk)

            logger.info(f"Split audio into {len(chunks)} chunks")

            # Process chunks
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._process_chunk(chunk, i, semaphore)
                for i, chunk in enumerate(chunks)
            ]

            results = await tqdm_asyncio.gather(*tasks, desc="Processing chunks")

            # Combine results
            if not any(results):
                logger.error("All chunk transcriptions failed")
                return None

            combined_transcript = " ".join(r['text'] for r in results if r)
            segments = []
            current_time = 0.0

            for result in results:
                if result:
                    for segment in result['segments']:
                        segment['start'] += current_time
                        segment['end'] += current_time
                        segments.append(segment)
                    current_time += result['duration']

            return {
                'transcript': combined_transcript,
                'segments': segments
            }

        except Exception as e:
            logger.error(f"Error in large file transcription: {e}")
            return None

    async def _process_chunk(self, chunk, chunk_index: int, semaphore: asyncio.Semaphore) -> Optional[Dict]:
        async with semaphore:
            try:
                # Save chunk to temporary file
                temp_file = Path(tempfile.gettempdir()) / f"chunk_{chunk_index}.wav"
                await asyncio.to_thread(lambda: chunk.save(str(temp_file)))

                # Transcribe chunk
                result = await self._transcribe_single_file(temp_file)

                # Clean up
                temp_file.unlink()

                return result

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_index}: {e}")
                return None


class IncidentAnalyzer:
    """Analyzes transcribed text using AI services"""

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
            model="chatgpt-4o-latest",
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
        return f"""Based on the following transcript, please provide a detailed meeting analysis in the following format:

        Meeting Summary:
        [2-3 sentences summarizing the key points discussed]

        Key Topics Discussed:
        1. [Topic 1]
           - Key points
           - Decisions made
        2. [Topic 2]
           - Key points
           - Decisions made
        [etc.]

        Action Items:
        1. [Action item] - Assigned to: [Name] - Due: [Date if mentioned]
        2. [Action item] - Assigned to: [Name] - Due: [Date if mentioned]
        [etc.]

        Timeline of Key Events:
        [List key events with their timestamps in chronological order]

        Participants:
        [List identified speakers/participants]

        Follow-up Required:
        [List any items requiring follow-up or scheduled for next meeting]

        Transcript:
        {transcript}"""


async def record_online_meeting(url: str, output_path: str) -> Optional[Path]:
    """
    Record an online meeting from URL

    Args:
        url: Meeting URL (Zoom or Google Meet)
        output_path: Where to save recording

    Returns:
        Optional[Path]: Path to recording file if successful
    """
    try:
        # Initialize recorders
        browser = BrowserMeetingRecorder(url)
        audio_recorder = AudioRecorderStream(output_path)

        # Join meeting
        if not browser.join_meeting():
            raise RuntimeError("Failed to join meeting")

        # Start recording
        logger.info("Starting meeting recording...")
        audio_recorder.start_recording()

        # Wait for keyboard interrupt
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping recording...")
            recording_path = audio_recorder.stop_recording()
            browser.leave_meeting()
            return recording_path

    except Exception as e:
        logger.error(f"Error recording meeting: {e}")
        return None


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
    group.add_argument(
        "--meeting-url",
        type=str,
        help="URL for Zoom or Google Meet to record"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="analysis_output.txt",
        help="Path to save analysis results (default: analysis_output.txt)"
    )

    parser.add_argument(
        "--recording-output",
        type=str,
        default="meeting_recording.wav",
        help="Path to save meeting recording (default: meeting_recording.wav)"
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
        # Initialize AI clients
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY", ""))
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

        # Initialize services
        transcriber = TranscriptionService(openai_client, anthropic_client)
        analyzer = IncidentAnalyzer(
            os.getenv("OPENAI_API_KEY", ""),
            os.getenv("ANTHROPIC_API_KEY", "")
        )

        # Handle input source
        if args.meeting_url:
            logger.info(f"Recording meeting from URL: {args.meeting_url}")
            audio_path = await record_online_meeting(args.meeting_url, args.recording_output)
            if not audio_path:
                raise RuntimeError("Failed to record meeting")
        elif args.file:
            audio_path = Path(args.file)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            logger.info(f"Using existing file: {audio_path}")
        else:
            config = AudioConfig(seconds=args.record)
            recorder = AudioRecorder(config)
            audio_path = recorder.record()
            logger.info(f"Recorded new audio file: {audio_path}")

        # Transcribe and analyze
        logger.info("Starting transcription...")
        result = await transcriber.transcribe(audio_path)

        if result and "segments" in result:
            segments = result["segments"]
            transcript = result["transcript"]

            logger.info(f"Transcription successful with {len(segments)} segments")

            # Save transcript with timestamps
            transcript_path = Path("transcript_with_timestamps.json")

            # Merge audio timestamps with transcript segments
            audio_timestamps = None
            timestamp_file = audio_path.with_suffix('.json')
            if timestamp_file.exists():
                with open(timestamp_file) as f:
                    audio_timestamps = json.load(f)

            if audio_timestamps:
                for segment, timestamp in zip(segments, audio_timestamps['timestamps']):
                    segment['audio_timestamp'] = timestamp

            save_analysis(json.dumps(segments, indent=2), str(transcript_path))
            logger.info(f"Saved timestamped transcript to {transcript_path}")

            # Analyze transcript
            logger.info("Starting analysis...")
            analysis = analyzer.analyze(transcript, args.ai)
            if analysis:
                save_analysis(analysis, args.output)
                logger.info(f"Analysis saved to: {args.output}")
            else:
                logger.error("Analysis failed")
        else:
            logger.error("Transcription failed or returned incomplete data")

    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


def main():
    """Entry point wrapper for async main"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()