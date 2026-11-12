"""
Tracking report generator for producing detailed reports of the data processing flow
"""
from typing import List, Dict, Any
from datetime import datetime
import json
from .tracking import TrackingMetadata, OperationTrackingInfo, DataChangeRecord, ChangeType


class TrackingReportGenerator:
    """Tracking report generator"""
    
    def __init__(self, tracking_metadata: TrackingMetadata):
        """
        Initialize the report generator
        
        Args:
            tracking_metadata: Tracking metadata
        """
        self.tracking_metadata = tracking_metadata
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report"""
        operations = self.tracking_metadata.operations
        
        summary = {
            'pipeline_id': self.tracking_metadata.pipeline_id,
            'session_id': self.tracking_metadata.session_id,
            'total_operations': len(operations),
            'total_input_records': self.tracking_metadata.total_input_records,
            'total_output_records': self.tracking_metadata.total_output_records,
            'net_change': self.tracking_metadata.total_output_records - self.tracking_metadata.total_input_records,
            'execution_start_time': operations[0].start_time.isoformat() if operations else None,
            'execution_end_time': operations[-1].end_time.isoformat() if operations else None,
            'total_execution_time_ms': self._calculate_total_execution_time(),
            'successful_operations': len([op for op in operations if op.success]),
            'failed_operations': len([op for op in operations if not op.success])
        }
        
        return summary
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """Generate a detailed report"""
        detailed_report = {
            'summary': self.generate_summary_report(),
            'operations': []
        }
        
        for op in self.tracking_metadata.operations:
            op_report = {
                'operation_id': op.operation_id,
                'operation_name': op.operation_name,
                'operation_type': op.operation_type.value,
                'start_time': op.start_time.isoformat(),
                'end_time': op.end_time.isoformat() if op.end_time else None,
                'execution_time_ms': (op.end_time - op.start_time).total_seconds() * 1000 if op.end_time else None,
                'success': op.success,
                'error_message': op.error_message,
                'input_stats': op.input_stats,
                'output_stats': op.output_stats,
                'parameters': op.parameters,
                'changes': []
            }
            
            for change in op.changes:
                change_report = {
                    'change_type': change.change_type.value,
                    'timestamp': change.timestamp.isoformat(),
                    'count': change.count,
                    'record_ids': change.record_ids,
                    'columns_affected': change.columns_affected,
                    'metadata': change.metadata
                }
                op_report['changes'].append(change_report)
            
            detailed_report['operations'].append(op_report)
        
        return detailed_report
    
    def generate_changes_summary(self) -> Dict[str, Any]:
        """Generate a change summary"""
        changes_summary = {
            'total_changes': 0,
            'by_type': {},
            'by_operation': {}
        }
        
        for op in self.tracking_metadata.operations:
            op_changes = {'total': 0, 'by_type': {}}
            
            for change in op.changes:
                changes_summary['total_changes'] += 1
                op_changes['total'] += 1
                
                # Count by change type
                change_type = change.change_type.value
                if change_type not in changes_summary['by_type']:
                    changes_summary['by_type'][change_type] = 0
                changes_summary['by_type'][change_type] += change.count or 1
                
                if change_type not in op_changes['by_type']:
                    op_changes['by_type'][change_type] = 0
                op_changes['by_type'][change_type] += change.count or 1
            
            changes_summary['by_operation'][op.operation_name] = op_changes
        
        return changes_summary
    
    def generate_data_quality_report(self) -> Dict[str, Any]:
        """Generate a data quality report"""
        quality_report = {
            'data_reduction_analysis': self._analyze_data_reduction(),
            'duplicate_analysis': self._analyze_duplicates(),
            'schema_changes': self._analyze_schema_changes()
        }
        
        return quality_report
    
    def _calculate_total_execution_time(self) -> float:
        """Compute total execution time"""
        operations = self.tracking_metadata.operations
        if not operations or not operations[0].start_time or not operations[-1].end_time:
            return 0
        
        total_time = (operations[-1].end_time - operations[0].start_time).total_seconds() * 1000
        return round(total_time, 2)
    
    def _analyze_data_reduction(self) -> Dict[str, Any]:
        """Analyze data reduction"""
        operations = self.tracking_metadata.operations
        
        reduction_analysis = {
            'initial_records': operations[0].input_stats['num_rows'] if operations and operations[0].input_stats else 0,
            'final_records': operations[-1].output_stats['num_rows'] if operations and operations[-1].output_stats else 0,
            'operations_analysis': []
        }
        
        for op in operations:
            if op.input_stats and op.output_stats:
                input_count = op.input_stats['num_rows']
                output_count = op.output_stats['num_rows']
                change = output_count - input_count
                
                reduction_analysis['operations_analysis'].append({
                    'operation': op.operation_name,
                    'input_count': input_count,
                    'output_count': output_count,
                    'change': change,
                    'percentage_change': round(((output_count - input_count) / input_count * 100), 2) if input_count > 0 else 0
                })
        
        return reduction_analysis
    
    def _analyze_duplicates(self) -> Dict[str, Any]:
        """Analyze duplicate data"""
        duplicate_changes = []
        
        for op in self.tracking_metadata.operations:
            for change in op.changes:
                if change.change_type in [ChangeType.DUPLICATE_FOUND, ChangeType.DUPLICATE_REMOVED]:
                    duplicate_changes.append({
                        'operation': op.operation_name,
                        'change_type': change.change_type.value,
                        'count': change.count,
                        'timestamp': change.timestamp.isoformat(),
                        'metadata': change.metadata
                    })
        
        return {
            'total_duplicate_operations': len(duplicate_changes),
            'details': duplicate_changes
        }
    
    def _analyze_schema_changes(self) -> Dict[str, Any]:
        """Analyze schema changes"""
        schema_changes = []
        
        for op in self.tracking_metadata.operations:
            for change in op.changes:
                if change.change_type in [ChangeType.COLUMN_ADDED, ChangeType.COLUMN_REMOVED, ChangeType.COLUMN_MODIFIED]:
                    schema_changes.append({
                        'operation': op.operation_name,
                        'change_type': change.change_type.value,
                        'columns_affected': change.columns_affected,
                        'count': change.count,
                        'timestamp': change.timestamp.isoformat(),
                        'metadata': change.metadata
                    })
        
        return {
            'total_schema_changes': len(schema_changes),
            'details': schema_changes
        }
    
    def export_report(self, format_type: str = 'json', filepath: str = None) -> str:
        """Export the report"""
        report_data = self.generate_detailed_report()
        
        if format_type.lower() == 'json':
            content = json.dumps(report_data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Report exported to {filepath}"
        else:
            return content


def print_readable_report(tracking_metadata: TrackingMetadata):
    """Print a readable tracking report"""
    generator = TrackingReportGenerator(tracking_metadata)
    
    print("="*60)
    print("Data processing tracking report")
    print("="*60)
    
    # Print summary
    summary = generator.generate_summary_report()
    print(f"\nPipeline ID: {summary['pipeline_id']}")
    print(f"Session ID: {summary['session_id']}")
    print(f"Total operations: {summary['total_operations']}")
    print(f"Input records: {summary['total_input_records']:,}")
    print(f"Output records: {summary['total_output_records']:,}")
    print(f"Net change: {summary['net_change']:+,}")
    print(f"Total execution time: {summary['total_execution_time_ms']} ms")
    print(f"Successful operations: {summary['successful_operations']}")
    print(f"Failed operations: {summary['failed_operations']}")
    
    # Print change summary
    changes_summary = generator.generate_changes_summary()
    print(f"\nTotal changes: {changes_summary['total_changes']}")
    print("By change type:")
    for change_type, count in changes_summary['by_type'].items():
        print(f"  - {change_type}: {count}")
    
    # Print operation details
    print(f"\nOperation details:")
    for op in tracking_metadata.operations:
        print(f"\n  Operation: {op.operation_name} ({op.operation_type.value})")
        print(f"    Time: {op.start_time.strftime('%H:%M:%S.%f')[:-3]} - {op.end_time.strftime('%H:%M:%S.%f')[:-3] if op.end_time else 'Running'}")
        print(f"    Success: {'Yes' if op.success else 'No'}")
        if op.input_stats:
            print(f"    Input: {op.input_stats['num_rows']:,} rows")
        if op.output_stats:
            print(f"    Output: {op.output_stats['num_rows']:,} rows")
        if op.changes:
            print(f"    Changed records: {len(op.changes)} entries")
            for change in op.changes[:3]:  # Show only the first 3 entries
                print(f"      - {change.change_type.value}: {change.count} items")
            if len(op.changes) > 3:
                print(f"      ... and {len(op.changes) - 3} records")
    
    print("\n" + "="*60)