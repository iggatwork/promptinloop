# LookinLoop

A Python script for interacting with the Groq API using multiple API keys. This script supports model selection and basic connectivity testing.

## Setup

1. **Clone the repository**
2. **Install dependencies**
   - Run `pip install -r requirements.txt` to install required packages.
   - Ensure `python-dotenv` is installed for environment variable support.
3. **Configure API keys**
   - Create a `.env` file in the project root with the following content:
     ```env
     MY_API_KEY=your_first_groq_api_key
     MY_API_KEY2=your_second_groq_api_key
     ```
   - Never commit your `.env` file with real API keys to version control.
4. **Modify LookinLoop.py**
    - Selecting Groq model list
    - Modify prompt at your convenience

## Usage

Run the script from the command line:

```powershell
python LookinLoop.py
```

The script will:
- Load API keys from the `.env` file
- Test connectivity for each key
- Display an interactive menu with options
- Selecting a Groq model that first works
- Sending prompts to the API
- Viewing results and responses
- Handeling pause/resume and Exiting the program

### Menu Information
When you run `python LookinLoop.py`, a file selection menu will appear in your terminal. Use this menu to select an EXCEL file where a list of company_names exist:
- Choose output filename or use default
- Select the column on the file that states the company_name or the variable you need to include in your prompt
- Determine how many rows to process


### Output Files
When running the script, two main files will be generated or updated:

- **Log File (`lookinloop_YYYYMMDD_HHMMSS.log`)**
  - Contains detailed logs of the script's execution, including API requests, responses, errors, and other relevant events.
  - Useful for troubleshooting and reviewing the history of your interactions.

- **Excel File (`InputFileName-processed-YYYYMMDD_HHMMSS.xlsx`)**
  - Stores processed results from your API queries in a structured spreadsheet format.
  - Allows you to review, analyze, and share the data easily.

Both files are automatically named with a timestamp to help you organize and track your sessions.

## Pause and Resume

During execution, you can pause the script at any time using the menu options provided. When paused:
- The script will save your current progress, including processed rows and output files.
- You can safely exit the program and resume later without losing your work.

To resume:
- Run `python LookinLoop.py` again.
- Select the same input file and output file (or use the default suggested by the script).
- The script will detect previous progress and continue processing from where you left off.

This feature is useful for handling large datasets or long-running tasks, allowing you to split work across multiple sessions.

## Notes
- Update the `GROQ_MODELS` list in `LookinLoop.py` if Groq adds new models.
- Check the `.gitignore` file to ensure sensitive files are not tracked.

## Troubleshooting
- If you see `Import "dotenv" could not be resolved`, ensure you installed `python-dotenv`.
- For other issues, check your Python version and package installations.

## License
MIT
