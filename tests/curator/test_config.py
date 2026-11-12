"""
Test module - configuration class tests
"""
import pytest
import os
import tempfile
import yaml
from curator.core.config.config import (
    GlobalConfig,
    OperationConfig,
    PipelineConfig,
    DataConfig,
    ResourcesConfig,
    PipelineConfiguration
)


class TestGlobalConfig:
    """Test GlobalConfig class"""

    def test_default_config(self):
        """Test default config"""
        config = GlobalConfig()
        
        assert config.logging_level == "info"
        assert config.temp_dir == "/tmp"
        assert config.cache_dir == ".cache"

    def test_custom_config(self):
        """Test custom config"""
        config = GlobalConfig(
            logging_level="debug",
            temp_dir="/custom/temp",
            cache_dir="/custom/cache"
        )
        
        assert config.logging_level == "debug"
        assert config.temp_dir == "/custom/temp"
        assert config.cache_dir == "/custom/cache"


class TestOperationConfig:
    """Test OperationConfig class"""

    def test_valid_operation(self):
        """Test valid operation config"""
        config = OperationConfig(name="text_loading", params={"data_paths": ["data.json"]})
        
        assert config.name == "text_loading"
        assert config.params == {"data_paths": ["data.json"]}

    def test_empty_name_raises_error(self):
        """Test that an empty name raises"""
        with pytest.raises(ValueError):
            OperationConfig(name="")

    def test_invalid_name_type_raises_error(self):
        """Test that an invalid name type raises"""
        with pytest.raises(ValueError):
            OperationConfig(name=123)


class TestPipelineConfig:
    """Test PipelineConfig class"""

    def test_with_operations(self):
        """Test a config containing operations"""
        config = PipelineConfig(
            operations=[
                OperationConfig(name="load", params={}),
                OperationConfig(name="filter", params={})
            ]
        )
        
        assert len(config.operations) == 2

    def test_empty_operations(self):
        """Test an empty operation list"""
        config = PipelineConfig(operations=[])
        
        assert len(config.operations) == 0


class TestDataConfig:
    """Test DataConfig class"""

    def test_valid_config(self):
        """Test valid data config"""
        config = DataConfig(
            input_path="/input/data.json",
            output_path="/output/result.json",
            modalities=["text", "image"]
        )
        
        assert config.input_path == "/input/data.json"
        assert config.output_path == "/output/result.json"
        assert config.modalities == ["text", "image"]


class TestResourcesConfig:
    """Test ResourcesConfig class"""

    def test_default_config(self):
        """Test default resource config"""
        config = ResourcesConfig()
        
        assert config.gpu == 0
        assert config.cpu == 1
        assert config.memory == "4GB"

    def test_custom_config(self):
        """Test custom resource config"""
        config = ResourcesConfig(gpu=2, cpu=8, memory="16GB")
        
        assert config.gpu == 2
        assert config.cpu == 8
        assert config.memory == "16GB"


class TestPipelineConfiguration:
    """Test PipelineConfiguration class"""

    def test_from_dict(self):
        """Test loading from a dictionary"""
        config_dict = {
            "pipeline": {
                "operations": [
                    {"name": "load", "params": {}}
                ]
            }
        }
        config = PipelineConfiguration.from_dict(config_dict)
        
        assert config is not None
        assert len(config.pipeline.operations) == 1

    def test_from_yaml(self, tmp_path):
        """Test loading from a YAML file"""
        yaml_content = {
            "pipeline": {
                "operations": [
                    {"name": "text_loading", "params": {"data_paths": ["data.json"]}}
                ]
            }
        }
        
        yaml_file = tmp_path / "config.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f)
        
        config = PipelineConfiguration.from_yaml(yaml_file)
        
        assert config is not None
        assert len(config.pipeline.operations) == 1

    def test_to_yaml(self, tmp_path):
        """Test saving to a YAML file"""
        config = PipelineConfiguration(
            pipeline=PipelineConfig(
                operations=[OperationConfig(name="load", params={})]
            )
        )
        
        yaml_file = tmp_path / "output.yaml"
        config.to_yaml(yaml_file)
        
        assert os.path.exists(yaml_file)

    def test_with_all_sections(self):
        """Test complete configuration"""
        config_dict = {
            "global": {
                "logging_level": "debug",
                "temp_dir": "/tmp/test",
                "cache_dir": "/cache"
            },
            "pipeline": {
                "operations": [
                    {"name": "load", "params": {}},
                    {"name": "process", "params": {"batch_size": 32}}
                ]
            },
            "data": {
                "input_path": "/input",
                "output_path": "/output",
                "modalities": ["text"]
            },
            "resources": {
                "gpu": 1,
                "cpu": 4,
                "memory": "8GB"
            }
        }
        
        config = PipelineConfiguration.from_dict(config_dict)
        
        assert config.global_config.logging_level == "debug"
        assert config.global_config.temp_dir == "/tmp/test"
        assert config.data.input_path == "/input"
        assert config.resources.gpu == 1

    def test_with_global_config_alias(self):
        """Test the global config alias"""
        config_dict = {
            "global": {"logging_level": "warning"},
            "pipeline": {"operations": []}
        }
        
        config = PipelineConfiguration.from_dict(config_dict)
        
        assert config.global_config.logging_level == "warning"
