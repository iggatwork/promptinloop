from pyexpat import model
import pandas as pd
from groq import Groq
import httpx
import os
import sys
import time
import logging
from datetime import datetime, timedelta
import json
import signal
import atexit
from tqdm import tqdm
import random  # Import random for delay jitter
import openai

# Load API keys from environment variables
import os
from dotenv import load_dotenv
load_dotenv()
my_api_key = os.getenv("MY_API_KEY")
my_api_key2 = os.getenv("MY_API_KEY2")
my_OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
my_OPENAI_API_KEY1 = os.getenv("MY_OPENAI_API_KEY1")

# Rate limits are measured in five ways: 
    # RPM (requests per minute), 
    # RPD (requests per day), 
    # TPM (tokens per minute), 
    # TPD (tokens per day), 
    # and IPM (images per minute). 

# Valid models for OpenAI (update if they change)
OPENAI_MODELS = { 
    "gpt-5", 
    "gpt-5-mini", 
    "gpt-5-nano", 
    "gpt-4.1", 
    "gpt-4.1-mini", 
    "gpt-4.1-nano", 
    "o3", 
    "o4-mini", 
    "gpt-4o",
    "gpt-4o-mini" }

# Per-model rate limits (hardcoded); OpenAI handles limits internally
OPENAI_LIMITS = { 
    "gpt-5": {"TPM": 10000, "TPD": 900000, "RPM": 3, "RPD": 200}, 
    "gpt-5-mini": {"TPM": 60000, "TPD": 200000, "RPM": 3, "RPD": 200}, 
    "gpt-5-nano": {"TPM": 60000, "TPD": 200000, "RPM": 3, "RPD": 200}, 
    "gpt-4.1": {"TPM": 10000, "TPD": 900000, "RPM": 3, "RPD": 200}, 
    "gpt-4.1-mini": {"TPM": 60000, "TPD": 200000, "RPM": 3, "RPD": 200}, 
    "gpt-4.1-nano": {"TPM": 60000, "TPD": 200000, "RPM": 3, "RPD": 200}, 
    "o3": {"TPM": 100000, "TPD": 90000, "RPM": 3, "RPD": 200}, 
    "o4-mini": {"TPM": 100000, "TPD": 90000, "RPM": 3, "RPD": 200}, 
    "gpt-4o": {"TPM": 200000, "TPD": 900000, "RPM": 3, "RPD": 200}, 
    "gpt-4o-mini": {"TPM": 200000, "TPD": 900000, "RPM": 3, "RPD": 200}
}


# Valid models for Groq (update if they change)
GROQ_MODELS = [
    "llama-3.1-8b-instant",      # cheaper + faster
    "openai/gpt-oss-120b",       # large open-source GPT
    "openai/gpt-oss-20b",        # smaller open-source GPT
    "llama-3.1-70b-versatile",   # strong general-purpose
    "moonshotai/kimi-k2-instruct", # higher RPM: 60
    "qwen/qwen3-32b"               # higher RPM: 60
]

# Per-model rate limits (hardcoded)
MODEL_LIMITS = {
    "llama-3.1-8b-instant": {"RPM": 30, "RPD": 14400, "TPM": 6000, "TPD": 500000},
    "openai/gpt-oss-120b": {"RPM": 30, "RPD": 1000, "TPM": 8000, "TPD": 200000},
    "openai/gpt-oss-20b": {"RPM": 30, "RPD": 1000, "TPM": 8000, "TPD": 200000},
    "llama-3.1-70b-versatile": {"RPM": 30, "RPD": 1000, "TPM": 12000, "TPD": 100000},
    "moonshotai/kimi-k2-instruct": {"RPM": 60, "RPD": 1000, "TPM": 10000, "TPD": 300000},
    "qwen/qwen3-32b": {"RPM": 60, "RPD": 1000, "TPM": 6000, "TPD": 500000},
}

# Use the latest supported model, Groq supports Meta LLaMA 3.1, Gemma 2, Mixtral, etc.
# model = "claude-3-opus-20240229"  # That’s an Anthropic Claude model, which Groq does not host.
# Mymodel = "llama-3.1-70b-versatile", # strong general-purpose
# Mymodel = "llama-3.1-8b-instant", # cheaper + faster 
# Mymodel = "openai/gpt-oss-120b", # large open-source GPT
# Mymodel = "openai/gpt-oss-20b"   # smaller open-source GPT


class GroqRateLimiter:
    def __init__(self, model_name):
        limits = MODEL_LIMITS.get(model_name, MODEL_LIMITS["llama-3.1-8b-instant"])
        self.model_name = model_name
        self.requests_per_minute = limits["RPM"]
        self.requests_per_day = limits["RPD"]
        self.tokens_per_minute = limits["TPM"]
        self.tokens_per_day = limits["TPD"]
        self.remaining_requests = self.requests_per_day
        self.remaining_tokens = self.tokens_per_minute
        self.request_reset_time = None
        self.token_reset_time = None
        self.last_update = datetime.now()

    def update_limits(self, response_headers):
        """Update rate limits from response headers"""
        try:
            # Update request limits
            self.remaining_requests = int(response_headers.get('x-ratelimit-remaining-requests', self.remaining_requests))
            self.remaining_tokens = int(response_headers.get('x-ratelimit-remaining-tokens', self.remaining_tokens))
            
            # Parse reset times
            request_reset = response_headers.get('x-ratelimit-reset-requests', '0s')
            token_reset = response_headers.get('x-ratelimit-reset-tokens', '0s')
            
            # Convert reset times to seconds
            def parse_time(time_str):
                if 'm' in time_str and 's' in time_str:
                    mins, secs = time_str.replace('s','').split('m')
                    return float(mins) * 60 + float(secs)
                return float(time_str.replace('s',''))
            
            self.request_reset_time = datetime.now() + timedelta(seconds=parse_time(request_reset))
            self.token_reset_time = datetime.now() + timedelta(seconds=parse_time(token_reset))
            
            self.last_update = datetime.now()
            
        except Exception as e:
            logging.warning(f"Error updating rate limits: {e}")

    def should_wait(self):
        """Check if we should wait before next request"""
        now = datetime.now()
        
        # If we're close to limits, calculate wait time
        # Use model-specific thresholds (10% of limit)
        req_threshold = max(1, int(0.1 * self.requests_per_day))
        tok_threshold = max(1, int(0.1 * self.tokens_per_minute))
        if self.remaining_requests < req_threshold or self.remaining_tokens < tok_threshold:
            if self.request_reset_time and now < self.request_reset_time:
                return (self.request_reset_time - now).total_seconds()
            if self.token_reset_time and now < self.token_reset_time:
                return (self.token_reset_time - now).total_seconds()
        return 0

class ProgressTracker:
    def __init__(self, total_rows, output_file, sheet_name):
        self.total_rows = total_rows
        self.current_row = 0
        self.output_file = output_file
        self.sheet_name = sheet_name
        self.start_time = datetime.now()
        self.paused = False
        self.progress_file = "progress.json"
        self.load_progress()
        
    def load_progress(self):
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                if data['output_file'] == self.output_file:
                    self.current_row = data['current_row']
                    print(f"Resuming from row {self.current_row}")
        except FileNotFoundError:
            pass
            
    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump({
                'output_file': self.output_file,
                'current_row': self.current_row,
                'total_rows': self.total_rows,
                'sheet_name': self.sheet_name
            }, f)
            
    def update(self, row_index):
        self.current_row = row_index
        self.save_progress()
        
    def get_stats(self):
        elapsed = datetime.now() - self.start_time
        rows_per_hour = (self.current_row / elapsed.total_seconds()) * 3600 if elapsed.total_seconds() > 0 else 0
        remaining_rows = self.total_rows - self.current_row
        estimated_hours = remaining_rows / rows_per_hour if rows_per_hour > 0 else 0
        
        return {
            'progress': f"{self.current_row}/{self.total_rows}",
            'percent': (self.current_row / self.total_rows) * 100,
            'elapsed': str(elapsed).split('.')[0],
            'estimated_remaining': f"{estimated_hours:.1f} hours",
            'rows_per_hour': f"{rows_per_hour:.1f}"
        }

def setup_signal_handlers(tracker):
    """Setup signal handlers for pause/resume functionality"""
    interrupt_count = 0
    last_interrupt_time = 0
    
    def handle_interrupt(signum, frame):
        nonlocal interrupt_count, last_interrupt_time
        current_time = time.time()
        
        # Reset counter if more than 1 second between interrupts
        if current_time - last_interrupt_time > 1:
            interrupt_count = 0
        
        interrupt_count += 1
        last_interrupt_time = current_time
        
        # Force exit on double Ctrl+C
        if interrupt_count >= 2:
            print("\n\n=== Forced Exit ===")
            print("Saving current progress...")
            stats = tracker.get_stats()
            print(f"Processed: {stats['progress']} rows ({stats['percent']:.1f}%)")
            tracker.save_progress()
            sys.exit(1)
            
        # Normal pause menu on single Ctrl+C
        tracker.paused = True
        print("\n\n=== Processing Paused ===")
        print("(Press Ctrl+C again within 1 second to force stop)")
        
        while True:
            choice = input("\nChoose action:\n1. Resume processing\n2. Exit and save progress\nEnter choice (1/2): ").strip()
            
            if choice == '1':
                tracker.paused = False
                print("\n>>> Processing Resumed <<<")
                break
            elif choice == '2':
                stats = tracker.get_stats()
                print("\nSaving progress and exiting...")
                print(f"Processed: {stats['progress']} rows ({stats['percent']:.1f}%)")
                print(f"Time elapsed: {stats['elapsed']}")
                tracker.save_progress()
                sys.exit(0)
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    # Use CTRL+C (SIGINT) for pause/resume/stop on Windows
    signal.signal(signal.SIGINT, handle_interrupt)

def handle_error(message, exit_code=1):
    """Handle errors gracefully"""
    print(f"Error: {message}")
    print("Press Enter to exit...")
    input()
    sys.exit(exit_code)

def validate_excel_file(filepath):
    """Validate Excel file existence and readability"""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    try:
        pd.read_excel(filepath, sheet_name=None)
        return True, None
    except Exception as e:
        return False, str(e)

def safe_input(prompt, validate_func=None, default=None):
    """Safe input handling with validation"""
    while True:
        try:
            value = input(prompt).strip()
            if not value and default is not None:
                return default
            if validate_func and not validate_func(value):
                print("Invalid input. Please try again.")
                continue
            return value
        except Exception as e:
            print(f"Input error: {e}")
            if default is not None:
                return default

def select_file_console():
    """Console-based file selector"""
    print("\nAvailable Excel files in current directory:")
    excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
    
    if not excel_files:
        print("No Excel files found in current directory.")
        filepath = input("Enter full path to Excel file: ").strip()
        return filepath if os.path.exists(filepath) else None
    
    for idx, file in enumerate(excel_files, 1):
        print(f"{idx}. {file}")
    
    while True:
        try:
            choice = input("\nSelect file number or enter full path: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(excel_files):
                return excel_files[int(choice) - 1]
            elif os.path.exists(choice):
                return choice
            else:
                print("Invalid selection. Try again.")
        except (ValueError, IndexError):
            print("Invalid input. Try again.")

def select_file():
    """Try GUI file dialog first, fallback to console"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename(
            title="Select Excel file to process",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        return file if file else select_file_console()
    except ImportError:
        return select_file_console()


def test_api_connection(api_key, model, client_choice="both"):
    """Test basic API connectivity for given model and client type using the specified API key.

    Parameters:
      api_key: API key string.
      model: The model to test connectivity for.
      client_choice: "groq", "openai", or "both" (default) determines which client(s) to test.

    Returns:
      A dictionary with keys for "Groq" and/or "OpenAI" depending on the client_choice, each mapping to a tuple (bool, client or error message).
    """
    results = {}
    messages = [{"role": "user", "content": "test"}]

    if client_choice.lower() in ("both", "groq"):
        # Test Groq connectivity
        try:
            groq_client = Groq(
                api_key=api_key,
                timeout=httpx.Timeout(30.0, read=15.0, write=15.0, connect=10.0),
                max_retries=2
            )
            response = groq_client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=1
            )
            if hasattr(response, 'choices'):
                results["Groq"] = (True, groq_client)
            else:
                results["Groq"] = (False, "Invalid API response")
        except httpx.ConnectError as e:
            results["Groq"] = (False, f"Connection error: {str(e)} - Check your network/proxy settings")
        except httpx.TimeoutException as e:
            results["Groq"] = (False, f"Timeout error: {str(e)} - Try increasing timeouts")
        except Exception as e:
            results["Groq"] = (False, f"API error: {str(e)}")

    if client_choice.lower() in ("both", "openai"):
        # Test OpenAI connectivity using Model.list as a connectivity check
        try:
            openai.api_key = os.getenv("MY_OPENAI_API_KEY")
            # Using Model.list to test connectivity
            _ = openai.Model.list()
            results["OpenAI"] = (True, None)
        except Exception as e:
            results["OpenAI"] = (False, str(e))
    
    return results

def pick_Groq_valid_model(client):
    """Try available Groq models with better error handling"""
    print("\nTesting available Groq models...")
    
    # First test with minimal token request
    test_models = ["llama-3.1-8b-instant"]  # Start with fastest model
    
    for model in test_models:
        print(f"Quick test: {model}...", end=' ', flush=True)
        try:
            test = client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model=model,
                max_tokens=1,
                timeout=10
            )
            print("✅ Success!")
            return model
        except Exception as e:
            print(f"❌ Failed: {str(e)[:80]}...")
            continue
    
    print("\nAttempting connection with backup models...")
    for model in [m for m in GROQ_MODELS if m not in test_models]:
        print(f"Testing: {model}...", end=' ', flush=True)
        try:
            test = client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                model=model,
                max_tokens=1,
                timeout=15
            )
            print("✅ Success!")
            return model
        except Exception as e:
            print(f"❌ Failed: {str(e)[:80]}...")
            continue
    
    raise RuntimeError("No working models found for Groq. Please check your API access.")

# New function for testing OpenAI models

def pick_OPENAI_valid_model(client):
    """Try available OpenAI models with better error handling"""
    print("\nTesting available OpenAI models...")
    # Iterate over the valid OpenAI models defined in OPENAI_MODELS
    for model in OPENAI_MODELS:
        print(f"Testing: {model}...", end=' ', flush=True)
        try:
            # Use test_api_connection to test OpenAI connectivity
            res = test_api_connection(api_key=os.getenv("MY_OPENAI_API_KEY"), model=model, client_choice="openai")
            if res.get("OpenAI") and res.get("OpenAI")[0] == True:
                print("✅ Success!")
                return model
            else:
                print(f"❌ Failed: {res.get('OpenAI')[1]}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    raise RuntimeError("No working OpenAI models found. Please check your API access.")

def get_output_filename(input_filename):
    """Generate output filename by adding '-processed' before extension"""
    try:
        base, ext = os.path.splitext(input_filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        files_dir = os.path.join(os.getcwd(), "files")
        if not os.path.exists(files_dir):
            os.makedirs(files_dir)
        output_path = os.path.join(
            files_dir,
            f"{os.path.basename(base)}-processed-{timestamp}{ext}"
        )
        return output_path
    except Exception as e:
        raise Exception(f"Error generating output filename: {e}")

def validate_output_path(filepath):
    """Validate output path is writable"""
    try:
        # Check directory exists or create it
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        # Test if file is writable
        if os.path.exists(filepath):
            # Try opening existing file
            with open(filepath, 'a') as f:
                pass
        else:
            # Try creating new file
            with open(filepath, 'w') as f:
                pass
            os.remove(filepath)  # Clean up test file
            
        return True, None
    except Exception as e:
        return False, str(e)

def get_company_info(client, company_name, rate_limiter, additional_info=None, model=None):
    if model is None:
        model = "llama-3.1-8b-instant"
    if rate_limiter is None:
        rate_limiter = DummyRateLimiter(model)
    system_prompt = """You are a professional business analyst tasked with researching companies in the healthcare sector. 
                    Use all reliable sources of knowledge you have access to, including public websites, press releases, 
                    regulatory filings, and known business databases. 

                    Do not limit your reasoning to any reference context provided — treat that only as a hint for disambiguation. 
                    If information is not available, say \"Unknown\" or \"Not available\", and avoid speculation.."""

    results = {}
    
    # Helper function to send chat completions for both OpenAI and Groq clients
    def send_completion(messages, model, temperature, max_tokens, top_p, stream=False):
        import openai
        # For OpenAI: use openai.ChatCompletion.create if client is the openai module
        if getattr(client, '__name__', '') == 'openai':
            return openai.ChatCompletion.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream
            )
        # For other clients that support ChatCompletion attribute
        elif hasattr(client, 'ChatCompletion') and client.ChatCompletion is not None:
            return client.ChatCompletion.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream
            )
        # For Groq clients that use client.chat.completions.create
        elif hasattr(client, 'chat') and client.chat is not None:
            return client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream
            )
        else:
            raise Exception("Client does not support a valid ChatCompletion interface")

    max_retries = 3
    base_delay = 2
    def handle_429(e):
        if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429:
            print("\n=== API rate limit reached (429 Too Many Requests) ===")
            print("You have reached the daily request limit for this model/API key. Processing will stop.")
            print("Please try again tomorrow or switch to a different API key/model.")
            sys.exit(1)
        elif '429' in str(e):
            print("\n=== API rate limit reached (429 Too Many Requests) ===")
            print("You have reached the daily request limit for this model/API key. Processing will stop.")
            print("Please try again tomorrow or switch to a different API key/model.")
            sys.exit(1)

    # Step 1: Use additional_info to focus on the right company (unchanged)
    if additional_info:
        focus_prompt = f"First goal: Find the official website of a company named '{company_name}'. This company operates in the healthcare sector and is attending this tradefair: '{additional_info}'.Steps: 1. Run a search in the web for:{company_name} official website. 2. If the first search yields no result, run a second web search with country‑wildcards: \"{company_name}\" \"official website\" site:.de OR .ch OR .nl OR .fr OR .it OR .es OR .pl OR .uk OR .us 3. For every URL returned in step 1 or 2, fetch the page title and the snippet. Return them in a list. 4. For each domain that looks promising, do a WHOIS lookup to confirm the company is attending the indicated tradefair. Note any “website” field that appears. 7. Summarise the findings in one short paragraph: include the official website URL (or note if none exists), the country that the company is registered in (if found), and the sources you used. Return the answer in plain text, no markdown. If no active website can be found, state that clearly and provide the best evidence you could gather (e.g., archived page, registry entry)."
        retries = 0
        while retries < max_retries:
            try:
                wait_time = rate_limiter.should_wait()
                if wait_time > 0:
                    print(f"\nRate limit approaching - waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

                response = send_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": focus_prompt}
                    ],
                    model=model,
                    temperature=0.5,
                    max_tokens=300,
                    top_p=0.7,
                    stream=False
                )
                if hasattr(response, '_headers'):
                    rate_limiter.update_limits(response._headers)
                focus_content = response.choices[0].message.content.strip()
                results["AdditionalInfoSummary"] = focus_content if focus_content else "No info found"
                break
            except Exception as e:
                handle_429(e)
                retries += 1
                delay = (2 ** retries) + random.uniform(0, 1)
                print(f"Error getting AdditionalInfoSummary (attempt {retries}/{max_retries}): {str(e)[:200]}")
                print(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
                if retries == max_retries:
                    results["AdditionalInfoSummary"] = "Information not available"

    # Step 2: Use the obtained info to focus subsequent prompts
    context_info = results.get("AdditionalInfoSummary", "")
    def with_context(prompt):
        if context_info:
            return (
                f"You are researching the company '{company_name}'. "
                f"Here is some optional reference information which may or may not be useful: {context_info} "
                f"Use this reference only to help disambiguate the company if necessary, "
                f"but otherwise answer based on the most reliable and broad information available. "
                f"\n\n{prompt}"
            )
        return prompt

    # --- Optimized prompts ---
    # 1. Identity info (name, country, website)
    identity_prompt = with_context(f"""
    For the company known as {company_name}, provide:
    1. Full company name (or 'Unknown'), do not add extra details 
    2. Country of headquarters (or 'Unknown'), do not add extra details
    3. Official website (URL only, or 'Unknown'), do not add extra details

    Format your answer as:
    Company_name: ...
    Country: ...
    Website: ...
    """)

    retries = 0
    while retries < max_retries:
        try:
            wait_time = rate_limiter.should_wait()
            if wait_time > 0:
                print(f"\nRate limit approaching - waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

            response = send_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": identity_prompt}
                ],
                model=model,
                temperature=0.5,
                max_tokens=300,
                top_p=0.7,
                stream=False
            )
            if hasattr(response, '_headers'):
                rate_limiter.update_limits(response._headers)
            content = response.choices[0].message.content.strip()
            if content:
                for line in content.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        results[key.strip()] = value.strip()
                break
        except Exception as e:
            handle_429(e)
            retries += 1
            delay = (2 ** retries) + random.uniform(0, 1)
            print(f"Error getting identity info (attempt {retries}/{max_retries}): {str(e)[:200]}")
            print(f"Waiting {delay:.1f} seconds before retry...")
            time.sleep(delay)
            if retries == max_retries:
                results["Company_name"] = "Information not available"
                results["Country"] = "Information not available"
                results["Website"] = "Information not available"

    # 2. Electronics/outsourcing block (yes/no)
    electronics_prompt = with_context(f"""
    For the company {company_name}, answer only with Yes/No/Unknown:
    1. Does the company produce, sell, or integrate electronics in their products?
    2. Does the company outsource any electronic manufacturing work?
    3. Does the company outsource any electronic R&D activities?

    Format your answer as:
    HasElectronicInhisproducts: Yes/No/Unknown
    DoOutsourceElectronicManufacturing: Yes/No/Unknown
    DoOutsourceElectronicR&D: Yes/No/Unknown
    """)

    retries = 0
    while retries < max_retries:
        try:
            wait_time = rate_limiter.should_wait()
            if wait_time > 0:
                print(f"\nRate limit approaching - waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

            response = send_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": electronics_prompt}
                ],
                model=model,
                temperature=0.5,
                max_tokens=300,
                top_p=0.7,
                stream=False
            )
            if hasattr(response, '_headers'):
                rate_limiter.update_limits(response._headers)
            content = response.choices[0].message.content.strip()
            if content:
                for line in content.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        results[key.strip()] = value.strip()
                break
        except Exception as e:
            handle_429(e)
            retries += 1
            delay = (2 ** retries) + random.uniform(0, 1)
            print(f"Error getting electronics/outsourcing info (attempt {retries}/{max_retries}): {str(e)[:200]}")
            print(f"Waiting {delay:.1f} seconds before retry...")
            time.sleep(delay)
            if retries == max_retries:
                results["HasElectronicInhisproducts"] = "Information not available"
                results["DoOutsourceElectronicManufacturing"] = "Information not available"
                results["DoOutsourceElectronicR&D"] = "Information not available"

    # 3. Company profile (detailed description)
    profile_prompt = with_context(f"""
    Write a concise factual profile of {company_name}. Include: company type, size (employees/revenue), headquarters and global presence, main products/services, industries, brands, manufacturing model (OEM, outsourcing, suppliers, PCBs), R&D capabilities, integration of R&D and manufacturing, key personnel (CEO, CTO, R&D heads), and contact information if available. Clearly state if electronics, outsourcing of design, or outsourcing of R&D were confirmed. Use only verified information. If some details are not available, say 'Unknown'.
    """)

    retries = 0
    while retries < max_retries:
        try:
            wait_time = rate_limiter.should_wait()
            if wait_time > 0:
                print(f"\nRate limit approaching - waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

            response = send_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": profile_prompt}
                ],
                model=model,
                temperature=0.5,
                max_tokens=400,
                top_p=0.7,
                stream=False
            )
            if hasattr(response, '_headers'):
                rate_limiter.update_limits(response._headers)
            content = response.choices[0].message.content.strip()
            if content:
                results["BriefCompanyDescription"] = content
                break
        except Exception as e:
            handle_429(e)
            retries += 1
            delay = (2 ** retries) + random.uniform(0, 1)
            print(f"Error getting company profile (attempt {retries}/{max_retries}): {str(e)[:200]}")
            print(f"Waiting {delay:.1f} seconds before retry...")
            time.sleep(delay)
            if retries == max_retries:
                results["BriefCompanyDescription"] = "Information not available"

    return results





def get_company_info_openai(company_name, additional_info=None, client=None, model=None, max_retries=1):
    """
    Identify and profile a company using OpenAI GPT.
    Steps:
    1. Disambiguate company names, find healthcare-related one if possible.
    2. Check if attending the given trade fair (if context provided).
    3. If confirmed, return structured info (name, country, website, profile).
    4. If not confirmed, return best guesses.
    """
    # Use provided client or fallback to openai module
    if client is None:
        import openai
        client = openai
    # Ensure the client has an API key; if not, fallback to environment variable
    if not getattr(client, 'api_key', None):
        client.api_key = os.getenv("MY_OPENAI_API_KEY")
    # Use default model if not provided
    if model is None:
        model = "gpt-4o-mini"

    system_prompt = """You are a professional business analyst.
Your tasks are:
0. Search for official website. If necessary, refine the search using common domain extensions.
1. Return the top candidate URLs, titles, and snippets.
2. Disambiguate companies that have similar names.
3. Cross-check with official registries and WHOIS data to confirm the company's sector of activity.
4. Prioritize and confirm the company that may work in the healthcare sector and is attending the trade fair (if context provided).
5. If a confirmed company is found, provide the company profile otherwise return the best candidate information.
6 Return always the following schema: Include the following fields in the JSON:
   - ResultType: \"Confirmed\" if a verified company is found, otherwise \"Candidate\".
   - CompanyInformation: a combined object containing key data such as Name, Country, Website, Description, Employees, Revenue, Products and Services catalog, ElectronicsInProducts: Yes/No/Unknown (Yes in case the company has electronics in its products), OutsourceElectronicManufacturing Yes/No/Unknown (yes in case the company has outsourced electronics manufacturing), OutsourceElectronicR&D Yes/No/Unknown (yes in case the company has outsourced electronics R&D), KeyPeople, and ContactInfo.
   - CompleteQueryResponse: the full detailed response from your investigation with evidences for the yes/no/unknown answers.
Do not speculate—if something is unknown, say \"Unknown\".
Output strictly in JSON format using the provided schema."""

    # user_prompt = f"\nCompany to investigate: {company_name}\nTrade fair context: {additional_info or 'Unknown'}\n\nReturn JSON with the following schema:\n{{\n  \"ResultType\": \"Confirmed\" or \"Candidate\",\n  \"MergedInformation\": {{\n      \"Name\": \"...\",\n      \"Country\": \"...\",\n      \"Website\": \"...\",\n      \"Description\": \"...\",\n      \"Employees\": \"...\",\n      \"Revenue\": \"...\",\n      \"ProductsServices\": \"...\",\n      \"ElectronicsInProducts\": \"Yes/No/Unknown\",\n      \"OutsourceElectronicManufacturing\": \"Yes/No/Unknown\",\n      \"OutsourceElectronicR&D\": \"Yes/No/Unknown\",\n      \"KeyPeople\": \"...\",\n      \"ContactInfo\": \"...\"\n  }},\n  \"CompleteQueryResponse\": \"...\"\n}}\nEnsure that the JSON merges all relevant company data into a single structure."
    # user_prompt = f"\nCompany to investigate: {company_name}\nTrade fair context: {additional_info or 'Unknown'}\n\nReturn JSON with the following schema:\n{{\n   \"CompanyInformation\": {{\n  \"ResultType\": \"Confirmed\" or \"Candidate\",\n    \"Name\": \"...\",\n      \"Country\": \"...\",\n      \"Website\": \"...\",\n      \"Description\": \"...\",\n      \"Employees\": \"...\",\n      \"Revenue\": \"...\",\n      \"ProductsServices\": \"...\",\n      \"ElectronicsInProducts\": \"Yes/No/Unknown\",\n      \"OutsourceElectronicManufacturing\": \"Yes/No/Unknown\",\n      \"OutsourceElectronicR&D\": \"Yes/No/Unknown\",\n      \"KeyPeople\": \"...\",\n      \"ContactInfo\": \"...\"\n  }},\n  \"CompleteQueryResponse\": \"...\"\n}}\nEnsure that the JSON includes all relevant company data in the specified structure."

    user_prompt = f"\nCompany to investigate: {company_name}\nTrade fair context: {additional_info or 'Unknown'}\n\nReturn JSON with the following schema: \n{{ \"CompanyInformation\": {{ \"ResultType\": \"<Confirmed|Candidate>\", \"Name\": \"<...>\", \"Country\": \"<...>\", \"Website\": \"<...>\", \"Description\": \"<...>\", \"Employees\": \"<...>\", \"Revenue\": \"<...>\", \"ProductsServices\": \"<...>\", \"ElectronicsInProducts\": \"<Yes|No|Unknown>\", \"OutsourceElectronicManufacturing\": \"<Yes|No|Unknown>\", \"OutsourceElectronic-R&D\": \"<Yes|No|Unknown>\", \"KeyPeople\": \"<...>\", \"ContactInfo\": \"<...>\" }}, \"CompleteQueryResponse\": \"<...>\" }}\nEnsure that the JSON includes all relevant company data in the specified structure."

    retries = 0
    while retries < max_retries:
        try:
            response = client.ChatCompletion.create(
                model=model,
                temperature=0.3,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600
            )
            content = response.choices[0].message.content.strip()
            
            logging.debug(f"HTTP headers for {company_name}: {getattr(response, '_headers', 'No headers available')}")
            logging.info(f"Raw response for {company_name}: {content}")
            if not content:
                raise ValueError("Empty response content received from API")
            content = clean_response(content)  # Clean markdown markers from response
            return json.loads(content)
        except Exception as e:
            retries += 1
            wait = (2 ** retries) + random.random()
            logging.warning(f"Error for {company_name}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return {"error": f"Failed after {max_retries} retries"}

# Helper function to remove markdown code block markers from JSON responses
def clean_response(response_str):
    """Remove markdown code block markers from a JSON response string."""
    if response_str.startswith("```json"):
        response_str = response_str[len("```json"):].strip()
    if response_str.endswith("```"):
        response_str = response_str[:-3].strip()
    return response_str

def check_network():
    """Test basic internet connectivity"""
    test_urls = [
        "https://api.groq.com",
        "https://www.google.com"
    ]
    
    for url in test_urls:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code == 200:
                return True
        except:
            continue
    return False

def main():
    print("=== Company Information Processor ===")
    
    # Get and validate input file
    while True:
        input_file = select_file()
        if not input_file:
            if input("No file selected. Try again? (Y/n): ").lower() == 'n':
                handle_error("No file selected")
            continue
        
        valid, error = validate_excel_file(input_file)
        if valid:
            break
        print(f"Error with file: {error}")
    
    # Get output filename
    while True:
        output_file = get_output_filename(input_file)
        if safe_input("Use default output filename? (Y/n): ", default='y').lower() == 'n':
            while True:
                output_file = safe_input("Enter output filename: ")
                if not output_file.endswith('.xlsx'):
                    print("Filename must end with .xlsx")
                    continue
                # Make path absolute if not already
                if not os.path.isabs(output_file):
                    output_file = os.path.join(os.getcwd(), output_file)
                break
        
        # Validate output path is writable
        valid, error = validate_output_path(output_file)
        if valid:
            break
        print(f"Cannot write to output file: {error}")
        print("Please try a different location")

    print(f"\nOutput will be saved to: {output_file}")
    
    # Load Excel file and get sheet name
    try:
        xl = pd.ExcelFile(input_file)
        if len(xl.sheet_names) == 1:
            sheet_name = xl.sheet_names[0]
        else:
            print("\nAvailable sheets:")
            for idx, name in enumerate(xl.sheet_names, 1):
                print(f"{idx}. {name}")
            sheet_idx = int(safe_input("\nSelect sheet number: ", 
                          validate_func=lambda x: x.isdigit() and 1 <= int(x) <= len(xl.sheet_names))) - 1
            sheet_name = xl.sheet_names[sheet_idx]
        
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        
    except Exception as e:
        handle_error(f"Error loading Excel file: {e}")
    
    # Column selection with validation

    print("\nAvailable columns:")
    for idx, col in enumerate(df.columns, 1):
        print(f"{idx}. {col}")

    # Select main column
    while True:
        try:
            column_idx = int(safe_input("\nSelect main column number (e.g. company_name): ")) - 1
            if 0 <= column_idx < len(df.columns):
                column_name = df.columns[column_idx]
                break
            print("Invalid column number")
        except ValueError:
            print("Please enter a valid number")

    # Select second column
    print("\nSelect a second column for additional info (e.g. website, reference, etc.):")
    for idx, col in enumerate(df.columns, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            second_column_idx = int(safe_input("\nSelect second column number: ")) - 1
            if 0 <= second_column_idx < len(df.columns):
                second_column_name = df.columns[second_column_idx]
                break
            print("Invalid column number")
        except ValueError:
            print("Please enter a valid number")
    
    # Row processing with validation
    total_rows = len(df)
    print(f"\nTotal rows in file: {total_rows}")
    while True:
        try:
            rows_input = safe_input("How many rows to process? (Enter for all): ")
            if not rows_input:
                rows_to_process = total_rows
                break
            rows_to_process = int(rows_input)
            if 0 < rows_to_process <= total_rows:
                break
            print(f"Please enter a number between 1 and {total_rows}")
        except ValueError:
            print("Please enter a valid number")
    
    # Step 1: Check network
    print("\nChecking network connectivity...")
    if not check_network():
        handle_error("No internet connection available")

    # Step 2: Prompt for client selection based on README instructions
    current_client = 0  # Index for current client
    client_choice_input = safe_input("Select client type for processing (1: Groq, 2: OpenAI) [default 1]: ", validate_func=lambda x: (x in ['1','2']) if x else True, default='1').strip()
    if client_choice_input == '2':
        selected_client = "OpenAI"
        # try:
        #     valid_model = pick_OPENAI_valid_model(None)  # Passing None as client since it's not used internally
        # except Exception as e:
        #     handle_error(f"OpenAI model selection failed: {e}")
    else:
        selected_client = "Groq"
    print(f"Selected client: {selected_client} for processing")
    # print(f"Selected client: {selected_client} with model: {valid_model}")
    # # Set clients list based on selection for further processing
    # clients = [(None, valid_model)]
    
    # Step 3: Test all models for each API key and collect rate limit info if necessary
    print("\nTesting API keys and models...")
    groq_client_api_keys = [my_api_key, my_api_key2]
    openai_api_keys = [my_OPENAI_API_KEY, my_OPENAI_API_KEY1]
    api_keys = groq_client_api_keys if selected_client == "Groq" else openai_api_keys
    valid_models = {}
    model_headers = {}
    clients = []  # List of (client, model) tuples

    if selected_client == "OpenAI":
        for idx, api_key in enumerate(api_keys, 1):
            print(f"\nTesting API key {idx} for OpenAI...")
            openai.api_key = api_key
            for model in OPENAI_MODELS:
                try:
                    response = openai.ChatCompletion.create(
                        messages=[{"role": "user", "content": "test"}],
                        model=model,
                        max_tokens=1
                    )
                    if hasattr(response, 'choices'):
                        valid_models.setdefault(model, []).append(idx)
                        # OpenAI responses may not include headers; leaving headers empty
                        model_headers[(idx, model)] = {}
                        clients.append((openai, model))
                        print(f"Valid OpenAI model {model} for API key {idx}")
                except Exception as e:
                    print(f"⚠️ Could not initialize OpenAI client for API key {idx} and model {model}: {e}")
                    continue
    else:
        for idx, api_key in enumerate(api_keys, 1):
            print(f"\nTesting API key {idx} for Groq...")
            try:
                groq_client = Groq(
                    api_key=api_key,
                    timeout=httpx.Timeout(30.0, read=15.0, write=15.0, connect=10.0),
                    max_retries=2
                )
            except Exception as e:
                print(f"⚠️ Could not initialize Groq client for API key {idx}: {e}")
                continue
            for model in GROQ_MODELS:
                try:
                    response = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": "test"}],
                        model=model,
                        max_tokens=1
                    )
                    if hasattr(response, 'choices'):
                        valid_models.setdefault(model, []).append(idx)
                        if hasattr(response, '_headers'):
                            model_headers[(idx, model)] = response._headers
                        clients.append((groq_client, model))
                        print(f"Valid Groq model {model} for API key {idx}")
                except Exception as e:
                    continue

    if not clients:
        handle_error("No valid models found for your API keys.")

    # Step 4: Present menu of valid models with descriptions and rate limit info
    print("\nAvailable models for your API keys:")
    menu_items = []
    model_map = []  # List of (client, model, api_key) tuples for menu
    item_num = 1
    for (client, model) in clients:
        # Get API key value from client, if available
        api_key_val = client.api_key if hasattr(client, 'api_key') else "unknown"
        try:
            api_key_idx = api_keys.index(api_key_val) + 1
        except Exception:
            api_key_idx = "unknown"
        desc = {
            "llama-3.1-8b-instant": "faster, cheaper, high request quota. Best for scaling bulk lookups.",
            "llama-3.1-70b-versatile": "smarter & more accurate reasoning, but quota (1k/day) could bottleneck you quickly.",
            "openai/gpt-oss-120b": "smarter & more accurate reasoning, but quota (1k/day) could bottleneck you quickly.",
            "openai/gpt-oss-20b": "smarter & more accurate reasoning, but quota (1k/day) could bottleneck you quickly.",
            "qwen/qwen3-32b": "balance between quality and high daily token quota (500k/day), but still capped at 1k requests/day.",
            "moonshotai/kimi-k2-instruct": "generous RPM (60) and solid TPD (300k), but again only 1k requests/day."
        }.get(model, "")
        # Try to get live headers if available, else fallback
        headers = model_headers.get((api_key_idx, model), {})
        if headers:
            rpd = headers.get('x-ratelimit-limit-requests', 'unknown')
            rpd_left = headers.get('x-ratelimit-remaining-requests', 'unknown')
            tpm = headers.get('x-ratelimit-limit-tokens', 'unknown')
            tpm_left = headers.get('x-ratelimit-remaining-tokens', 'unknown')
            rpd_reset = headers.get('x-ratelimit-reset-requests', 'unknown')
            tpm_reset = headers.get('x-ratelimit-reset-tokens', 'unknown')
        else:
            limits = MODEL_LIMITS.get(model, {})
            rpd = limits.get('RPD', 'unknown')
            rpd_left = 'unknown'
            tpm = limits.get('TPM', 'unknown')
            tpm_left = 'unknown'
            rpd_reset = 'unknown'
            tpm_reset = 'unknown'
        print(f"{item_num}. {model} (API key {api_key_idx}): {desc}")
        if rpd != 'unknown':
            print(f"   Requests/day: {rpd}, left: {rpd_left}, Tokens/min: {tpm}, left: {tpm_left}, Reset: {rpd_reset} (requests), {tpm_reset} (tokens)")
        menu_items.append(f"{model} (API key {api_key_idx})")
        model_map.append((client, model, api_key_val))
        item_num += 1
    if not menu_items:
        handle_error("No valid models found for your API keys.")

    # Step 5: User selects model
    while True:
        model_choice = safe_input(f"Enter model number (1-{len(menu_items)}): ", validate_func=lambda x: x.isdigit() and 1 <= int(x) <= len(menu_items))
        model_idx = int(model_choice) - 1
        selected_client, selected_model, _ = model_map[model_idx]  # unpack 3-tuple
        # Show summary and confirm
        client_obj = selected_client
        client_type = "OpenAI" if (hasattr(client_obj, "__name__") and client_obj.__name__.lower() == "openai") else "Groq"
        
        print(f"Selected client: {client_type}, model: {selected_model}")
        confirm = safe_input("Proceed with this client and model? (Y/n): ", default='y').lower()
        if confirm == 'y':
            break

    # Step 6: Use selected model/client for all further processing
    model = selected_model
    # current_client = 0  # Index for current client
    clients = [(selected_client, selected_model)]  # Only use selected client/model for processing
    
    # Initialize logging
    files_dir = os.path.join(os.getcwd(), "files")
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
    log_filename = os.path.join(files_dir, f"lookinloop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize progress tracker
    tracker = ProgressTracker(rows_to_process, output_file, sheet_name)
    setup_signal_handlers(tracker)
    
    # Create progress bar
    pbar = tqdm(total=rows_to_process, initial=tracker.current_row)
    
    # Initialize rate limiter or set to None based on selected client
    if (client_type == "Groq"):
        rate_limiter = GroqRateLimiter(selected_model) #valid_model)
    else:
        rate_limiter = None  # For OpenAI, rate limiting is handled internally
    
    # Ensure output DataFrame has required columns
    required_fields = [
        "Identity",
        "Company_name",
        "Country",
        "Website",
        "HasElectronicInhisproducts",
        "DoOutsourceElectronicManufacturing",
        "DoOutsourceElectronicR&D",
        "BriefCompanyDescription"
    ]
    # Ensure output DataFrame has only the required columns (plus original columns)
    for field in required_fields:
        if field not in df.columns:
            df[field] = ""

    # Create initial Excel file with headers
    try:
        print("\nCopying all content from input file to output file...")
        df.to_excel(output_file, sheet_name=sheet_name, index=False)
        print(f"✅ Copied all content to: {output_file}")
    except Exception as e:
        handle_error(f"Failed to copy input file content to output file: {e}")
    
    # After file creation
    success, error = verify_file_creation(output_file)
    if not success:
        handle_error(f"Failed to verify output file: {error}")
    
    # After log file creation
    success, error = verify_file_creation(log_filename)
    if not success:
        print(f"Warning: Could not verify log file: {error}")
    
    # Process rows
    try:
        print("\nStarting processing...")
        print("Press Ctrl+C to stop and save progress.\n")
        client_obj, model = clients[current_client]
        client_type = "OpenAI" if (hasattr(client_obj, "__name__") and client_obj.__name__.lower() == "openai") else "Groq"
        print("client_type:", client_type)
        print("model:", model)

        for index, row in df.iloc[tracker.current_row:rows_to_process].iterrows():
            company_name = row[column_name]
            additional_info = row.get(second_column_name, "")  # Use .get() to avoid KeyError
            logging.info(f"Processing {company_name} ({index + 1}/{rows_to_process}) ")

            try:
                
                if client_type == "OpenAI":
                    result = get_company_info_openai(company_name, additional_info, client=client_obj, model=model)
                else:
                    result = get_company_info(client_obj, company_name, rate_limiter, additional_info, model=model)
                
                if result:
                    logging.info(f"Result for {company_name}: {result}")
                    norm_result = normalize_result(result)
                    primary_result = norm_result  
                    print(f"\r✅ Completed: {company_name}")

                    # Update required columns only using primary_result
                    for field in required_fields:
                        df.at[index, field] = primary_result.get(field, "")

                    # New block: Update any additional keys from the result not already in the DataFrame
                    for key, value in primary_result.items():
                        if key not in df.columns:
                            df[key] = ""
                        df.at[index, key] = json.dumps(value) if isinstance(value, (dict, list)) else value

                    # Save updates to Excel with proper error handling
                    try:
                        output_cols = list(df.columns)
                        df.to_excel(output_file, sheet_name=sheet_name, index=False, columns=output_cols)
                        print(f"💾 Saved updates to {output_file}")
                    except Exception as e:
                        logging.error(f"Failed to save updates: {e}")

                    tracker.update(index + 1)
                    pbar.update(1)

                    # Log rate limit status periodically
                    if (index + 1) % 5 == 0:
                        logging.info(
                            f"Rate limits - Requests: {rate_limiter.remaining_requests}, Tokens: {rate_limiter.remaining_tokens}"
                        )

            except Exception as e:
                print(f"Error processing {company_name} (row {index}): {str(e)}")
                logging.error(f"Error processing {company_name} (row {index}): {str(e)}")
                if len(clients) > 1:
                    current_client = (current_client + 1) % len(clients)
                    print(f"\nSwitching to backup API key {current_client + 1}")
                    continue
                continue

            tracker.update(index + 1)
            pbar.update(1)
    except Exception as e:
        logging.error(f"Processing error: {e}")
    finally:
        print(f"\nOutput file location: {os.path.abspath(output_file)}")
        print(f"Log file location: {os.path.abspath(log_filename)}")
    
    print("\nProcessing complete!")

def verify_file_creation(filepath, expected_type="file"):
    """Verify file was created and is accessible"""
    try:
        if not os.path.exists(filepath):
            return False, f"{expected_type.capitalize()} not found"
        
        if expected_type == "file":
            # Try to open the file
            with open(filepath, 'rb') as f:
                f.read(1)  # Try to read 1 byte
                
        return True, None
    except Exception as e:
        return False, str(e)

class DummyRateLimiter:
    def __init__(self, model_name=None):
        # Dummy initialization for OpenAI; no rate limiting is applied.
        pass

    def should_wait(self):
        # Always return 0 as no wait time is needed for OpenAI.
        return 0

    def update_limits(self, response_headers):
        # No operation needed for updating limits in dummy limiter.
        pass

def normalize_result(result):
    """Flatten the API result so that keys within 'CompanyInformation' become top-level columns."""
    normalized = {}
    if not result:
        return normalized
    # Extract keys from the nested 'CompanyInformation' dictionary
    comp_info = result.get('CompanyInformation', {})
    if isinstance(comp_info, dict):
        for key, value in comp_info.items():
            normalized[key] = value
    # Include additional top-level keys from the result
    for key, value in result.items():
        if key not in ['CompanyInformation']:
            normalized[key] = value
    return normalized

if __name__ == "__main__":
    main()