import json
import os
from typing import Dict, List, Any

class ConfigManager:
    """Manages different research configurations for reusable prompts and output fields."""
    
    def __init__(self, config_dir="config"):
        self.config_dir = config_dir
        self.configs = {}
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files from the config directory."""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            return
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]  # Remove .json extension
                config_path = os.path.join(self.config_dir, filename)
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.configs[config_name] = json.load(f)
                except Exception as e:
                    print(f"Error loading config {filename}: {e}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a specific configuration by name."""
        return self.configs.get(config_name, {})
    
    def list_configs(self) -> List[str]:
        """List all available configuration names."""
        return list(self.configs.keys())
    
    def get_system_prompt(self, config_name: str) -> str:
        """Get the system prompt for a configuration."""
        config = self.get_config(config_name)
        return config.get('system_prompt', '')
    
    def get_user_prompt_template(self, config_name: str) -> str:
        """Get the user prompt template for a configuration."""
        config = self.get_config(config_name)
        return config.get('user_prompt_template', '')
    
    def format_user_prompt(self, config_name: str, company_name: str, additional_info: str = "Unknown") -> str:
        """Format the user prompt with company information."""
        template = self.get_user_prompt_template(config_name)
        return template.format(company_name=company_name, additional_info=additional_info or "Unknown")
    
    def get_output_fields(self, config_name: str) -> List[Dict[str, str]]:
        """Get the output field mappings for a configuration."""
        config = self.get_config(config_name)
        return config.get('output_fields', [])
    
    def get_required_columns(self, config_name: str) -> List[str]:
        """Get list of required column names for output."""
        output_fields = self.get_output_fields(config_name)
        return [field['field'] for field in output_fields]
    
    def get_api_settings(self, config_name: str) -> Dict[str, Any]:
        """Get API settings for a configuration."""
        config = self.get_config(config_name)
        return config.get('api_settings', {
            'temperature': 0.3,
            'max_tokens': 600,
            'max_retries': 1
        })
    
    def map_result_to_output(self, config_name: str, result: Dict[str, Any]) -> Dict[str, str]:
        """Map API result to output columns based on configuration."""
        output_fields = self.get_output_fields(config_name)
        mapped_result = {}
        
        # Get CompanyInformation if it exists, otherwise use result directly
        company_info = result.get('CompanyInformation', result)
        
        for field_config in output_fields:
            field_name = field_config['field']
            source_field = field_config['source']
            default_value = field_config['default']
            
            # Try to get value from company_info, fallback to result, then default
            value = company_info.get(source_field, result.get(source_field, default_value))
            mapped_result[field_name] = value
        
        return mapped_result

def select_config(config_manager: ConfigManager) -> str:
    """Interactive configuration selection."""
    configs = config_manager.list_configs()
    
    if not configs:
        print("No configurations found. Please create configuration files in the 'config' directory.")
        return None
    
    print("\nAvailable research configurations:")
    for idx, config_name in enumerate(configs, 1):
        config = config_manager.get_config(config_name)
        name = config.get('name', config_name)
        description = config.get('description', 'No description')
        print(f"{idx}. {name} - {description}")
    
    while True:
        try:
            choice = input(f"\nSelect configuration (1-{len(configs)}): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(configs):
                return configs[int(choice) - 1]
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please try again.")
    
    return None