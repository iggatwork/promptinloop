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
    "openai/gpt-oss-20b",         # smaller open-source GPT
    "llama-3.1-70b-versatile"    # strong general-purpose
]

# Use the latest supported model, Groq supports Meta LLaMA 3.1, Gemma 2, Mixtral, etc.
# model = "claude-3-opus-20240229"  # That’s an Anthropic Claude model, which Groq does not host.
# Mymodel = "llama-3.1-70b-versatile", # strong general-purpose
Mymodel = "llama-3.1-8b-instant", # cheaper + faster 
# Mymodel = "openai/gpt-oss-120b", # large open-source GPT
# Mymodel = "openai/gpt-oss-20b"   # smaller open-source GPT

class GroqRateLimiter:
    def __init__(self):
        self.requests_per_day = 14400
        self.tokens_per_minute = 18000
        self.remaining_requests = 14400
        self.remaining_tokens = 18000
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
        if self.remaining_requests < 100 or self.remaining_tokens < 1000:
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
        # Create a timestamp to make filename unique
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Use current directory instead of Downloads
        output_path = os.path.join(
            os.getcwd(), 
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

def get_company_info(client, company_name, rate_limiter):
    """Get detailed company information using Groq API with rate limiting"""
    model = "llama-3.1-8b-instant"  # Use the fastest model consistently
    
    # System prompt to improve context
    system_prompt = """You are a business analyst tasked with researching companies in healthcare sector. 
    Provide accurate, concise information based on available data. 
    If information is not available or uncertain, state that clearly."""
    
    prompts = {
    "Company_name": f"What is the full legal name of the company known as {company_name}? Provide the exact registered name only.",
    "Country": f"In which country is {company_name} headquartered? Provide only the country name.",
    "Website": f"What is the official website of {company_name}? Provide the URL only.",
    "HasElectronicInhisproducts": f"Does {company_name} produce, sell, or integrate electronics in their products? Answer only 'Yes' or 'No'. Do not add explanations here.",
    "DoOutsourceElectronicDesign": f"Does {company_name} outsource any electronic design work? Answer only 'Yes' or 'No'. Do not add explanations here.",
    "DoOutsourceElectronicR&D": f"Does {company_name} outsource any electronic R&D activities (research, prototyping, testing)? Answer only 'Yes' or 'No'. Do not add explanations here.",
    "BriefCompanyDescription": f"Summarize {company_name}'s profile in a concise paragraph including: company type (public/private/startup/etc.), size (employees/revenue if available), headquarters and global presence (offices, factories, labs), main products/services, industries/sectors, brands/product lines, manufacturing model (OEM, outsourcing, suppliers, PCBs), R&D capabilities, integration of R&D and manufacturing, key personnel (CEO, CTO, R&D heads), and contact information if available. Also include any evidence you found when answering about electronics in products, outsourcing of electronic design, and outsourcing of electronic R&D. Use only factual information gathered from reliable sources (website, press releases, etc.)."
    }
    results = {}
    max_retries = 3
    base_delay = 2
    
    for field, prompt in prompts.items():
        retries = 0
        while retries < max_retries:
            try:
                # Check rate limits
                wait_time = rate_limiter.should_wait()
                if wait_time > 0:
                    print(f"\nRate limit approaching - waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                
                # Make API call
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=model,
                    temperature=0.5,
                    max_tokens=300,
                    top_p=0.7,
                    stream=False
                )
                
                # Update rate limits from response
                if hasattr(response, '_headers'):
                    rate_limiter.update_limits(response._headers)
                
                content = response.choices[0].message.content.strip()
                if content:
                    results[field] = content
                    break
                    
            except Exception as e:
                retries += 1
                delay = (2 ** retries) + random.uniform(0, 1)
                print(f"Error getting {field} (attempt {retries}/{max_retries}): {str(e)[:200]}")
                print(f"Waiting {delay:.1f} seconds before retry...")
                time.sleep(delay)
                
                if retries == max_retries:
                    results[field] = "Information not available"
    
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
    
    while True:
        try:
            column_idx = int(safe_input("\nSelect column number to process: ")) - 1
            if 0 <= column_idx < len(df.columns):
                column_name = df.columns[column_idx]
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
    
    # Initialize Groq client with better error handling
    try:
        print("\nChecking network connectivity...")
        if not check_network():
            raise Exception("No internet connection available")
            
        print("\nInitializing API clients...")
        clients = []
        
        for idx, api_key in enumerate([my_api_key, my_api_key2], 1):
            print(f"\nTesting API key {idx}...")
            
            # Test basic connectivity first
            connected, result = test_api_connection(api_key)
            if not connected:
                print(f"⚠️ API key {idx} connection failed: {result}")
                continue
                
            try:
                client = result  # Use the client from successful connection test
                model = pick_valid_model(client)
                print(f"✅ API key {idx} validated successfully with model: {model}")
                clients.append(client)  # Store only the client, not a tuple
                
            except Exception as e:
                print(f"⚠️ API key {idx} validation failed: {str(e)[:200]}")
                
        if not clients:
            raise Exception("No working API keys found. Please check:\n" +
                          "1. Your internet connection\n" +
                          "2. API key validity\n" +
                          "3. Groq service status: https://status.groq.com")
            
    except Exception as e:
        handle_error(f"Failed to initialize API clients: {e}")
    
    current_client = 0  # Index for current client
    
    # Initialize logging
    logging.basicConfig(
        filename=f"lookinloop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize progress tracker
    tracker = ProgressTracker(rows_to_process, output_file, sheet_name)
    setup_signal_handlers(tracker)
    
    # Create progress bar
    pbar = tqdm(total=rows_to_process, initial=tracker.current_row)
    
    # Initialize rate limiter
    rate_limiter = GroqRateLimiter()
    
    # Create initial Excel file with headers
    try:
        print("\nCreating output file with headers...")
        df_initial = pd.DataFrame(columns=df.columns)
        df_initial.to_excel(output_file, sheet_name=sheet_name, index=False)
        print(f"✅ Created: {output_file}")
    except Exception as e:
        handle_error(f"Failed to create output file: {e}")
    
    # After file creation
    success, error = verify_file_creation(output_file)
    if not success:
        handle_error(f"Failed to verify output file: {error}")
    
    # After log file creation
    log_file = f"lookinloop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    success, error = verify_file_creation(log_file)
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
                logging.info(f"Processing {company_name} ({index + 1}/{rows_to_process})")
                
                try:
                    company_info = get_company_info(clients[current_client], company_name, rate_limiter)
                    if company_info:
                        print(f"\r✅ Completed: {company_name}")
                        # Update DataFrame
                        for field, value in company_info.items():
                            df.at[index, field] = value
                        
                        # Save updates to Excel with proper error handling
                        try:
                            # Read existing file
                            existing_df = pd.read_excel(output_file, sheet_name=sheet_name)
                            # Update the row
                            for field, value in company_info.items():
                                existing_df.at[index, field] = value
                            # Save entire file
                            existing_df.to_excel(output_file, sheet_name=sheet_name, index=False)
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
                    # Try with backup API key if available
                    if len(clients) > 1:
                        current_client = (current_client + 1) % len(clients)
                        print(f"\nSwitching to backup API key {current_client + 1}")
                        continue
                    logging.error(f"Error processing {company_name}: {str(e)}")
                    continue
    
            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Processing error: {e}")
    finally:
        print(f"\nOutput file location: {os.path.abspath(output_file)}")
        print(f"Log file location: {os.path.abspath(logging.getLoggerClass().root.handlers[0].baseFilename)}")
    
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