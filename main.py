import pyaudio
import wave
import speech_recognition as sr
from google.cloud import speech
from anthropic import Anthropic
import openai
from itertools import cycle
import requests
import json
from datetime import datetime

# Audio recording setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10
OUTPUT_FILE = "recorded_audio.wav"

def record_audio():
    print("Recording audio...")
    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a file
    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio recorded and saved to {OUTPUT_FILE}")
    return OUTPUT_FILE

def transcribe_audio_google(file_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Load the audio file
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)

    try:
        # Transcribe audio using Google Web Speech API
        print("Transcribing audio...")
        text = recognizer.recognize_google(audio_data)
        print("\nTranscription:")
        print(text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def send_to_xmatters(content):
    try:
        # xMatters API configuration
        XMATTERS_URL = "YOUR_XMATTERS_INSTANCE_URL"  # e.g., "https://company.xmatters.com/api/xm/1/events"
        XMATTERS_API_KEY = "YOUR_XMATTERS_API_KEY"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {XMATTERS_API_KEY}"
        }
        
        # Format the payload according to xMatters requirements
        payload = {
            "properties": {
                "title": f"Incident Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "analysis": content
            },
            "recipients": [
                {
                    "targetName": "YOUR_TARGET_GROUP"  # e.g., "Incident Response Team"
                }
            ]
        }
        
        response = requests.post(
            XMATTERS_URL,
            headers=headers,
            json=payload
        )
        
        if response.status_code == 202:  # xMatters typically returns 202 for successful submissions
            print("\nAnalysis sent to xMatters successfully")
            return True
        else:
            print(f"\nError sending to xMatters: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending to xMatters: {e}")
        return False

def analyze_with_ai(transcript, retry=True):
    try:
        content = analyze_with_openai(transcript)
        if content:
            send_to_xmatters(content)
        return content
    except Exception as e:
        print(f"\nError with OpenAI API: {e}")
        if retry:
            print("Falling back to Claude API...")
            try:
                content = analyze_with_claude(transcript)
                if content:
                    send_to_xmatters(content)
                return content
            except Exception as e:
                print(f"\nError with Claude API: {e}")
                return None
        return None

def analyze_with_claude(transcript):
    anthropic = Anthropic()
    
    prompt = f"""Based on the following incident transcript, please provide:
1. Timeline of major events
2. List of services impacted
3. Initial root cause analysis
4. Mitigation steps

Transcript:
{transcript}"""

    message = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    print("\nClaude Analysis Results:")
    print(message.content)
    return message.content

def analyze_with_openai(transcript):
    prompt = f"""Based on the following incident transcript, please provide a structured analysis with these specific sections:
1. Restoration Team involved
2. Escalation method used
3. Timeline of events (in MST)
4. Description of the incident
5. Services impacted
6. Root cause (if known)
7. Mitigation steps taken
8. Immediate actions required
9. Long-term solutions proposed
10. Follow-up actions needed

Please format each section clearly with headers.

Transcript:
{transcript}"""

    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=1000
    )
    
    print("\nOpenAI Analysis Results:")
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def format_description(content):
    """
    Formats the AI analysis into the standardized incident format
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    return f"""
Restoration Team: {get_team_from_content(content)}

Escalation method: {get_escalation_method(content)}

Timeline: 
{extract_timeline(content)}

Description: {extract_description(content)}

Services Impacted: {extract_services(content)}

Root Cause: {extract_root_cause(content)}

Mitigation Steps: {extract_mitigation(content)}

Immediate Actions:
{extract_immediate_actions(content)}

Long-Term Solutions:
{extract_long_term_solutions(content)}

Follow-up Actions:
{extract_follow_up_actions(content)}
"""

def extract_timeline(content):
    # Extract and format timeline events from AI content
    # This would need to parse the AI response and format times in MST
    events = []
    # Logic to extract timeline events
    return "\n".join(events)

# Helper functions to extract specific sections from AI content
def get_team_from_content(content):
    # Logic to determine restoration team
    return "TBD"

def get_escalation_method(content):
    # Logic to determine escalation method
    return "TBD"

def extract_description(content):
    # Logic to extract main description
    return content.get('description', 'TBD')

def extract_services(content):
    # Logic to extract impacted services
    return content.get('services', 'TBD')

def extract_root_cause(content):
    # Logic to extract root cause
    return content.get('root_cause', 'Unknown')

def extract_mitigation(content):
    # Logic to extract mitigation steps
    return content.get('mitigation', 'TBD')

def extract_immediate_actions(content):
    # Logic to extract immediate actions
    return content.get('immediate_actions', 'TBD')

def extract_long_term_solutions(content):
    # Logic to extract long-term solutions
    return content.get('long_term', 'TBD')

def extract_follow_up_actions(content):
    # Logic to extract follow-up actions
    return content.get('follow_up', 'TBD')

def main():
    recorded_file = record_audio()
    transcript = transcribe_audio_google(recorded_file)
    if transcript:
        analyze_with_ai(transcript)

if __name__ == "__main__":
    main()
