# LookinLoop - Configurable Company Research Tool

A Python script for AI-powered company research using Groq and OpenAI APIs with configurable research templates for different sectors and use cases. This tool supports multiple API keys, model selection, and customizable research configurations.

## Features

- 🔧 **Configurable Research Templates**: Pre-built configurations for healthcare, defense, and custom sectors
- 🚀 **Multiple AI Providers**: Support for both Groq and OpenAI APIs
- 📊 **Excel Integration**: Seamless input/output with Excel files
- ⏸️ **Pause & Resume**: Save progress and continue later
- 🔄 **Rate Limiting**: Built-in rate limit management
- 📝 **Detailed Logging**: Comprehensive logging for troubleshooting

## Quick Start

1. **Run the tool**: `python LookinLoop.py`
2. **Select your configuration** when prompted (e.g., Healthcare Research, Defense Industry Research)
3. **Choose your input file** and columns with sensible information
4. **The tool will use the selected configuration** for prompts and output fields

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   - Ensure `python-dotenv` is installed for environment variable support.

3. **Configure API keys**
   - Create a `.env` file in the project root with the following content:
     ```env
     MY_API_KEY=your_first_groq_api_key
     MY_API_KEY2=your_second_groq_api_key
     MY_OPENAI_API_KEY=your_first_openAI_key
     MY_OPENAI_API_KEY1=your_second_openAI_key
     ```
   - Never commit your `.env` file with real API keys to version control.

## Usage

Run the script from the command line:

```powershell
python LookinLoop.py
```

### Workflow

The script will guide you through:

1. **Configuration Selection**: Choose from available research templates
2. **File Selection**: Select your Excel input file
3. **Column Mapping**: Choose company name and context columns
4. **Processing Options**: Set number of rows to process
5. **AI Provider Selection**: Choose between Groq or OpenAI
6. **Model Selection**: Pick the best model for your needs
7. **Processing**: Automated research with progress tracking

### Menu Information

When you run the script, an interactive menu will appear:
- Choose output filename or use default
- Select the column containing company names
- Select a secondary column for additional context
- Determine how many rows to process
- Select from available AI providers and models
- Monitor real-time progress

### Output Files

Two main files are generated:

- **Log File (`lookinloop_YYYYMMDD_HHMMSS.log`)**
  - Detailed execution logs, API requests/responses, errors
  - Useful for troubleshooting and session review

- **Excel File (`InputFileName-processed-YYYYMMDD_HHMMSS.xlsx`)**
  - Structured spreadsheet with research results
  - Ready for analysis and sharing

Both files use timestamps for easy organization.

## Configuration System

### Available Configurations

#### Healthcare Research (`healthcare_research.json`)
- **Focus**: Medical devices, healthcare companies at trade fairs
- **Fields**: Electronics in products, outsourcing practices
- **Use case**: Trade fair exhibitor research

#### Defense Research (`defense_research.json`)
- **Focus**: Defense contractors, military suppliers
- **Fields**: Defense contracts, security clearances, ITAR status
- **Use case**: Defense industry analysis

### Creating Custom Configurations

Create a new configuration file in the `config/` directory:

**Example**: `config/biotech_research.json`

```json
{
  "name": "Biotech Research",
  "description": "Research biotechnology and pharmaceutical companies",
  "system_prompt": "You are a biotech industry analyst specializing in pharmaceutical research...",
  "user_prompt_template": "Company to investigate: {company_name}\nContext: {additional_info}\n\nReturn JSON with biotech-specific fields...",
  "output_fields": [
    {"field": "Company_name", "source": "Name", "default": ""},
    {"field": "Country", "source": "Country", "default": ""},
    {"field": "DrugPipeline", "source": "DrugPipeline", "default": ""},
    {"field": "ClinicalTrials", "source": "ClinicalTrials", "default": ""},
    {"field": "FDA_Approvals", "source": "FDA_Approvals", "default": ""}
  ],
  "api_settings": {
    "temperature": 0.2,
    "max_tokens": 800,
    "max_retries": 2
  }
}
```

### Configuration Structure

- **name**: Display name for the configuration
- **description**: Brief description of what this config does
- **system_prompt**: Instructions for the AI about the research domain
- **user_prompt_template**: Template with `{company_name}` and `{additional_info}` placeholders
- **output_fields**: Array of field mappings:
  - `field`: Column name in the output Excel
  - `source`: Field name from the AI response JSON
  - `default`: Default value if field is missing
- **api_settings**: 
  - `temperature`: AI creativity (0.0-1.0)
  - `max_tokens`: Maximum response length
  - `max_retries`: Number of retry attempts

## Model Selection Guidelines

### Factors to Consider

**Daily Request Allowance (RPD):**
- `llama-3.1-8b-instant`: 14,400/day → highest by far 🚀
- Most others: only 1,000/day

**Token Capacity (TPM / TPD):**
- `llama-3.1-8b-instant`: 6k/min, 500k/day
- `qwen/qwen3-32b`: 6k/min, 500k/day
- `gpt-oss-120b`: 8k/min, but only 200k/day
- `llama-3.1-70b-versatile`: 12k/min, but only 100k/day
- `moonshotai/kimi-k2-instruct`: 10k/min, 300k/day

**Model Quality vs Cost:**
- **8B models** (`llama-3.1-8b-instant`): Faster, cheaper, high request quota. Best for bulk lookups.
- **70B/120B models**: Smarter, more accurate reasoning, but quota limits (1k/day) may bottleneck large datasets.
- **32B models** (`qwen3-32b`): Balance between quality and high daily token quota.

## Advanced Features

### Pause and Resume

During execution, you can pause the script at any time:
- Current progress is automatically saved
- Output files are preserved
- Resume by running the script again with the same input file

### Custom Normalization
The tool automatically maps AI response fields to your specified output columns based on the configuration.

### Multiple Configurations
Create as many configurations as needed for different research scenarios.

### API Settings per Configuration
Each configuration can have its own optimal API settings (temperature, tokens, retries).

## Benefits of the Configuration System

1. **Reusability**: Create once, use multiple times
2. **Customization**: Easy to modify prompts and output fields
3. **Consistency**: Standardized approach across research types
4. **Maintainability**: No code changes needed for new domains
5. **Sharing**: Configuration files can be shared between team members

## Tips and Best Practices

1. **Test your prompts** with a small dataset first
2. **Use specific field names** in your JSON schema
3. **Include fallback values** for missing information
4. **Adjust temperature** based on creativity needs (lower = more factual)
5. **Monitor token usage** for cost optimization
6. **Use appropriate models** for your dataset size and quality needs

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **API key issues**: Verify your `.env` file contains valid API keys
- **Rate limits**: The tool includes automatic rate limiting, but monitor your usage
- **File permissions**: Ensure write permissions for output directories

## License

MIT