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


# Load API keys from environment variables
import os
from dotenv import load_dotenv
load_dotenv()
my_api_key = os.getenv("MY_API_KEY")
my_api_key2 = os.getenv("MY_API_KEY2")

# Valid models for Groq (update if they change in docs)
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
Mymodel = "llama-3.1-8b-instant", # cheaper + faster 
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


def test_api_connection(api_key):
    """Test basic API connectivity before attempting model validation"""
    try:
        # Increase timeouts and add SSL verification options
        client = Groq(
            api_key=api_key,
            timeout=httpx.Timeout(30.0, read=15.0, write=15.0, connect=10.0),
            max_retries=2
        )
        
        # Use direct API test instead of health endpoint
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.1-8b-instant",
            max_tokens=1
        )
        
        if hasattr(response, 'choices'):
            return True, client
        return False, "Invalid API response"
            
    except httpx.ConnectError as e:
        return False, f"Connection error: {str(e)} - Check your network/proxy settings"
    except httpx.TimeoutException as e:
        return False, f"Timeout error: {str(e)} - Try increasing timeouts"
    except Exception as e:
        return False, f"API error: {str(e)}"

def pick_valid_model(client):
    """Try available models with better error handling"""
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
            
    raise RuntimeError("No working models found. Please check your API access.")

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
    """Get detailed company information using Groq API with rate limiting and additional info"""
    if model is None:
        model = "llama-3.1-8b-instant"
    system_prompt = """You are a professional business analyst tasked with researching companies in the healthcare sector. 
                    Use all reliable sources of knowledge you have access to, including public websites, press releases, 
                    regulatory filings, and known business databases. 

                    Do not limit your reasoning to any reference context provided — treat that only as a hint for disambiguation. 
                    If information is not available, say "Unknown" or "Not available", and avoid speculation.."""

    results = {}
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
        focus_prompt = f"Given the company name '{company_name}' and the following reference or link: '{additional_info}', extract any relevant information about the company and use it to help identify and focus on the correct company for further research. Summarize what you find and clarify if the reference is useful for distinguishing the company from others with similar names. If there is not an exact match and there are multiple companies with similar names, provide a list of these company names."
        retries = 0
        while retries < max_retries:
            try:
                wait_time = rate_limiter.should_wait()
                if wait_time > 0:
                    print(f"\nRate limit approaching - waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)

                response = client.chat.completions.create(
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

            response = client.chat.completions.create(
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

            response = client.chat.completions.create(
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

            response = client.chat.completions.create(
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

    # Step 2: Test all models for each API key and collect rate limit info
    print("\nTesting API keys and models...")
    api_keys = [my_api_key, my_api_key2]
    valid_models = {}
    model_headers = {}
    clients = []  # List of (client, model) tuples
    for idx, api_key in enumerate(api_keys, 1):
        print(f"\nTesting API key {idx}...")
        try:
            client = Groq(
                api_key=api_key,
                timeout=httpx.Timeout(30.0, read=15.0, write=15.0, connect=10.0),
                max_retries=2
            )
        except Exception as e:
            print(f"⚠️ Could not initialize client for API key {idx}: {e}")
            continue
        for model in GROQ_MODELS:
            try:
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}],
                    model=model,
                    max_tokens=1
                )
                if hasattr(response, 'choices'):
                    valid_models.setdefault(model, []).append(idx)
                    if hasattr(response, '_headers'):
                        model_headers[(idx, model)] = response._headers
                    clients.append((client, model))  # Store tuple for valid (client, model)
            except Exception as e:
                continue

    # Step 3: Present menu of valid models with descriptions and rate limit info
    print("\nAvailable models for your API keys:")
    menu_items = []
    model_map = []  # List of (client, model) tuples for menu
    item_num = 1
    for (client, model) in clients:
        desc = {
            "llama-3.1-8b-instant": "faster, cheaper, high request quota. Best for scaling bulk lookups.",
            "llama-3.1-70b-versatile": "smarter & more accurate reasoning, but quota (1k/day) could bottleneck you quickly.",
            "openai/gpt-oss-120b": "smarter & more accurate reasoning, but quota (1k/day) could bottleneck you quickly.",
            "openai/gpt-oss-20b": "smarter & more accurate reasoning, but quota (1k/day) could bottleneck you quickly.",
            "qwen/qwen3-32b": "balance between quality and high daily token quota (500k/day), but still capped at 1k requests/day.",
            "moonshotai/kimi-k2-instruct": "generous RPM (60) and solid TPD (300k), but again only 1k requests/day."
        }.get(model, "")
        # Find API key index for this client
        api_key_idx = api_keys.index(client.api_key) + 1 if hasattr(client, 'api_key') and client.api_key in api_keys else 1
        # Try to get live headers, else fallback to hardcoded limits
        headers = model_headers.get((api_key_idx, model), {})
        if headers:
            rpd = headers.get('x-ratelimit-limit-requests', 'unknown')
            rpd_left = headers.get('x-ratelimit-remaining-requests', 'unknown')
            tpm = headers.get('x-ratelimit-limit-tokens', 'unknown')
            tpm_left = headers.get('x-ratelimit-remaining-tokens', 'unknown')
            rpd_reset = headers.get('x-ratelimit-reset-requests', 'unknown')
            tpm_reset = headers.get('x-ratelimit-reset-tokens', 'unknown')
        else:
            # Fallback: use hardcoded limits
            limits = MODEL_LIMITS.get(model, {})
            rpd = limits.get('RPD', 'unknown')
            rpd_left = 'unknown'  # Not available
            tpm = limits.get('TPM', 'unknown')
            tpm_left = 'unknown'  # Not available
            rpd_reset = 'unknown'
            tpm_reset = 'unknown'
        print(f"{item_num}. {model} (API key {api_key_idx}): {desc}\n   Requests/day: {rpd}, left: {rpd_left}, Tokens/min: {tpm}, left: {tpm_left}, Reset: {rpd_reset} (requests), {tpm_reset} (tokens)")
        # NOTE: If you want live quota info, check Groq client docs for how to access HTTP response headers.
        menu_items.append(f"{model} (API key {api_key_idx})")
        model_map.append((client, model))
        item_num += 1

    if not menu_items:
        handle_error("No valid models found for your API keys.")

    # Step 4: User selects model
    while True:
        model_choice = safe_input(f"Enter model number (1-{len(menu_items)}): ", validate_func=lambda x: x.isdigit() and 1 <= int(x) <= len(menu_items))
        model_idx = int(model_choice) - 1
        selected_client, selected_model = model_map[model_idx]
        print(f"Selected model: {selected_model}")
        confirm = safe_input("Proceed with this model? (Y/n): ", default='y').lower()
        if confirm == 'y':
            break

    # Step 5: Use selected model/client for all further processing
    model = selected_model
    current_client = 0  # Index for current client
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
    
    # Initialize rate limiter for the selected model
    rate_limiter = GroqRateLimiter(model)
    
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
        for index, row in df.iloc[tracker.current_row:rows_to_process].iterrows():

            try:
                while tracker.paused:
                    time.sleep(0.1)

                company_name = row[column_name]
                additional_info = row[second_column_name]
                logging.info(f"Processing {company_name} ({index + 1}/{rows_to_process}) ")

                try:
                    client, model = clients[current_client]
                    company_info = get_company_info(client, company_name, rate_limiter, additional_info=additional_info, model=model)
                    if company_info:
                        print(f"\r✅ Completed: {company_name}")


                        # Update required columns only
                        for field in required_fields:
                            df.at[index, field] = company_info.get(field, "")

                        # Save updates to Excel with proper error handling
                        try:
                            # Only save required columns plus original columns
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
                                f"Rate limits - Requests: {rate_limiter.remaining_requests}, "
                                f"Tokens: {rate_limiter.remaining_tokens}"
                            )

                except Exception as e:
                    # Log full error details for diagnostics
                    print(f"Error processing {company_name} (row {index}): {str(e)}")
                    logging.error(f"Error processing {company_name} (row {index}): {str(e)}")
                    # Try with backup API key if available
                    if len(clients) > 1:
                        current_client = (current_client + 1) % len(clients)
                        print(f"\nSwitching to backup API key {current_client + 1}")
                        continue
                    continue

            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
                continue
                
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

if __name__ == "__main__":
    main()