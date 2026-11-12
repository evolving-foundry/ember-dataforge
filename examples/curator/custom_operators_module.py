"""
Create an example custom operator module demonstrating how to add new operators without modifying core code
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from curator.core.pipeline import register_operation
from curator.core.operation import Operation
from typing import Any, Dict, Optional
import ray


@register_operation('custom_word_count')
class WordCountOperation(Operation):
    """Word count operation that counts words in a text field"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.text_field = self.params.get('text_field', 'text')
        self.count_field = self.params.get('count_field', 'word_count')
    
    def execute(self, dataset: ray.data.Dataset):
        """
        Count the number of words in the text field of each record

        Args:
            dataset: Input dataset

        Returns:
            Dataset with a word count field added
        """
        def count_words(row):
            row = row.copy()  # Avoid mutating the original data
            if self.text_field in row and isinstance(row[self.text_field], str):
                row[self.count_field] = len(row[self.text_field].split())
            else:
                row[self.count_field] = 0
            return row
            
        return dataset.map(count_words)


@register_operation('custom_prefix_adder')
class PrefixAdderOperation(Operation):
    """Prefix adder operation that adds a specified prefix to a text field"""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.text_field = self.params.get('text_field', 'text')
        self.prefix = self.params.get('prefix', '[PREFIX] ')
    
    def execute(self, dataset: ray.data.Dataset):
        """
        Add a prefix to the text in the specified field

        Args:
            dataset: Input dataset

        Returns:
            Dataset with the prefix added
        """
        def add_prefix(row):
            row = row.copy()  # Avoid mutating the original data
            if self.text_field in row and isinstance(row[self.text_field], str):
                row[self.text_field] = self.prefix + row[self.text_field]
            return row
            
        return dataset.map(add_prefix)
