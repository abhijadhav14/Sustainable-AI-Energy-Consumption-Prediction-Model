"""
Sustainability Metrics Module

This module provides tools for measuring and monitoring the sustainability aspects of AI systems,
including energy consumption, carbon footprint, and other sustainability metrics.
"""

import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Union, Optional, Tuple, Any
import matplotlib.pyplot as plt

class EnergyTracker:
    """
    Tracks energy consumption of AI model training and inference.
    """
    
    def __init__(self, device_type: str = 'cpu', power_consumption_watts: Optional[float] = None):
        """
        Initialize the energy tracker.
        
        Args:
            device_type: Type of computing device ('cpu', 'gpu', 'tpu')
            power_consumption_watts: Optional override for device power consumption
        """
        self.device_type = device_type.lower()
        self.start_time = None
        self.end_time = None
        self.duration_seconds = 0
        self.total_energy_kwh = 0
        
        # Set default power consumption based on device type if not provided
        if power_consumption_watts is None:
            if self.device_type == 'cpu':
                self.power_consumption_watts = 65  # Average CPU power consumption
            elif self.device_type == 'gpu':
                self.power_consumption_watts = 250  # Average GPU power consumption
            elif self.device_type == 'tpu':
                self.power_consumption_watts = 200  # Estimated TPU power consumption
            else:
                self.power_consumption_watts = 100  # Default for unknown devices
        else:
            self.power_consumption_watts = power_consumption_watts
        
        # Initialize tracking variables
        self.tracking_history = []
    
    def start_tracking(self) -> None:
        """Start tracking energy consumption."""
        self.start_time = time.time()
    
    def stop_tracking(self) -> Dict[str, float]:
        """
        Stop tracking energy consumption and calculate metrics.
        
        Returns:
            Dictionary with energy consumption metrics
        """
        if self.start_time is None:
            raise ValueError("Tracking was not started. Call start_tracking() first.")
        
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time
        
        # Calculate energy consumption in kilowatt-hours
        # Power (W) * Time (h) / 1000 = Energy (kWh)
        self.total_energy_kwh = (self.power_consumption_watts * self.duration_seconds / 3600) / 1000
        
        # Record tracking session
        session_data = {
            'device_type': self.device_type,
            'power_consumption_watts': self.power_consumption_watts,
            'duration_seconds': self.duration_seconds,
            'energy_consumption_kwh': self.total_energy_kwh,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self.tracking_history.append(session_data)
        
        return session_data
    
    def get_total_energy_consumption(self) -> float:
        """
        Get the total energy consumption in kilowatt-hours.
        
        Returns:
            Total energy consumption in kWh
        """
        return self.total_energy_kwh
    
    def get_tracking_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all tracking sessions.
        
        Returns:
            List of tracking session data
        """
        return self.tracking_history
    
    def estimate_energy_for_operations(self, num_operations: int, operations_per_second: float) -> float:
        """
        Estimate energy consumption for a given number of operations.
        
        Args:
            num_operations: Number of operations to perform
            operations_per_second: Operations per second the device can perform
            
        Returns:
            Estimated energy consumption in kWh
        """
        estimated_seconds = num_operations / operations_per_second
        estimated_kwh = (self.power_consumption_watts * estimated_seconds / 3600) / 1000
        return estimated_kwh
    
    def plot_energy_consumption(self, output_path: Optional[str] = None) -> None:
        """
        Plot energy consumption history.
        
        Args:
            output_path: Optional path to save the plot
        """
        if not self.tracking_history:
            raise ValueError("No tracking history available.")
        
        # Extract data for plotting
        timestamps = [session['timestamp'] for session in self.tracking_history]
        energy_values = [session['energy_consumption_kwh'] for session in self.tracking_history]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(energy_values)), energy_values)
        plt.xticks(range(len(timestamps)), timestamps, rotation=45)
        plt.xlabel('Tracking Session')
        plt.ylabel('Energy Consumption (kWh)')
        plt.title('Energy Consumption History')
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def save_tracking_data(self, filepath: str) -> None:
        """
        Save tracking data to a JSON file.
        
        Args:
            filepath: Path to save the data
        """
        with open(filepath, 'w') as f:
            json.dump({
                'device_type': self.device_type,
                'power_consumption_watts': self.power_consumption_watts,
                'tracking_history': self.tracking_history
            }, f, indent=2)
    
    @classmethod
    def load_tracking_data(cls, filepath: str) -> 'EnergyTracker':
        """
        Load tracking data from a JSON file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            EnergyTracker instance with loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls(
            device_type=data['device_type'],
            power_consumption_watts=data['power_consumption_watts']
        )
        
        tracker.tracking_history = data['tracking_history']
        
        # Calculate total energy consumption
        tracker.total_energy_kwh = sum(session['energy_consumption_kwh'] for session in tracker.tracking_history)
        
        return tracker


class CarbonFootprintCalculator:
    """
    Calculates carbon footprint of AI model training and inference.
    """
    
    # Carbon intensity of electricity by region (kg CO2 per kWh)
    # Source: IEA 2020 data (simplified)
    CARBON_INTENSITY = {
        'global': 0.475,
        'us': 0.417,
        'eu': 0.275,
        'china': 0.555,
        'india': 0.708,
        'canada': 0.135,
        'australia': 0.656,
        'uk': 0.233,
        'france': 0.056,
        'germany': 0.338,
        'japan': 0.474,
        'brazil': 0.074,
        'south_africa': 0.912,
        'russia': 0.374,
        'mexico': 0.454,
        'south_korea': 0.415,
        'indonesia': 0.761,
        'saudi_arabia': 0.709
    }
    
    def __init__(self, region: str = 'global', renewable_percentage: float = 0):
        """
        Initialize the carbon footprint calculator.
        
        Args:
            region: Geographic region for carbon intensity calculation
            renewable_percentage: Percentage of energy from renewable sources (0-100)
        """
        self.region = region.lower()
        
        # Validate region
        if self.region not in self.CARBON_INTENSITY:
            raise ValueError(f"Region '{region}' not supported. Available regions: {list(self.CARBON_INTENSITY.keys())}")
        
        # Validate renewable percentage
        if not 0 <= renewable_percentage <= 100:
            raise ValueError("Renewable percentage must be between 0 and 100")
        
        self.renewable_percentage = renewable_percentage
        
        # Calculate effective carbon intensity based on renewable percentage
        self.effective_carbon_intensity = self.CARBON_INTENSITY[self.region] * (1 - renewable_percentage / 100)
    
    def calculate_carbon_footprint(self, energy_consumption_kwh: float) -> float:
        """
        Calculate carbon footprint based on energy consumption.
        
        Args:
            energy_consumption_kwh: Energy consumption in kilowatt-hours
            
        Returns:
            Carbon footprint in kg CO2 equivalent
        """
        return energy_consumption_kwh * self.effective_carbon_intensity
    
    def get_carbon_intensity(self) -> float:
        """
        Get the effective carbon intensity.
        
        Returns:
            Effective carbon intensity in kg CO2 per kWh
        """
        return self.effective_carbon_intensity
    
    def compare_regions(self, energy_consumption_kwh: float) -> Dict[str, float]:
        """
        Compare carbon footprint across different regions.
        
        Args:
            energy_consumption_kwh: Energy consumption in kilowatt-hours
            
        Returns:
            Dictionary mapping regions to carbon footprints
        """
        return {
            region: energy_consumption_kwh * intensity * (1 - self.renewable_percentage / 100)
            for region, intensity in self.CARBON_INTENSITY.items()
        }
    
    def calculate_offset_cost(self, carbon_footprint_kg: float, cost_per_ton: float = 15) -> float:
        """
        Calculate the cost to offset carbon emissions.
        
        Args:
            carbon_footprint_kg: Carbon footprint in kg CO2 equivalent
            cost_per_ton: Cost per ton of CO2 for carbon offsets
            
        Returns:
            Cost to offset carbon emissions in the same currency as cost_per_ton
        """
        # Convert kg to tons and calculate cost
        return (carbon_footprint_kg / 1000) * cost_per_ton
    
    def get_equivalent_activities(self, carbon_footprint_kg: float) -> Dict[str, float]:
        """
        Get equivalent activities for the carbon footprint.
        
        Args:
            carbon_footprint_kg: Carbon footprint in kg CO2 equivalent
            
        Returns:
            Dictionary mapping activities to equivalent values
        """
        # Average emissions for various activities
        car_emissions_per_km = 0.12  # kg CO2 per km
        flight_emissions_per_km = 0.1  # kg CO2 per passenger km
        beef_emissions_per_kg = 27  # kg CO2 per kg
        tree_absorption_per_year = 21  # kg CO2 per tree per year
        
        return {
            'car_km': carbon_footprint_kg / car_emissions_per_km,
            'flight_km': carbon_footprint_kg / flight_emissions_per_km,
            'beef_kg': carbon_footprint_kg / beef_emissions_per_kg,
            'trees_for_year': carbon_footprint_kg / tree_absorption_per_year
        }
    
    def plot_carbon_comparison(self, energy_consumption_kwh: float, 
                             regions_to_compare: Optional[List[str]] = None,
                             output_path: Optional[str] = None) -> None:
        """
        Plot carbon footprint comparison across regions.
        
        Args:
            energy_consumption_kwh: Energy consumption in kilowatt-hours
            regions_to_compare: Optional list of regions to compare
            output_path: Optional path to save the plot
        """
        if regions_to_compare is None:
            # Select a subset of regions for clarity
            regions_to_compare = ['global', 'us', 'eu', 'china', 'india', 'france', 'brazil']
        
        # Calculate carbon footprint for each region
        footprints = {
            region: energy_consumption_kwh * self.CARBON_INTENSITY[region] * (1 - self.renewable_percentage / 100)
            for region in regions_to_compare if region in self.CARBON_INTENSITY
        }
        
        # Sort by footprint
        sorted_regions = sorted(footprints.items(), key=lambda x: x[1])
        regions = [item[0] for item in sorted_regions]
        values = [item[1] for item in sorted_regions]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(regions, values)
        
        # Highlight current region
        if self.region in regions:
            idx = regions.index(self.region)
            bars[idx].set_color('red')
        
        plt.xlabel('Region')
        plt.ylabel('Carbon Footprint (kg CO2eq)')
        plt.title(f'Carbon Footprint Comparison for {energy_consumption_kwh:.2f} kWh')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()


class SustainabilityEvaluator:
    """
    Evaluates overall sustainability of AI models based on multiple metrics.
    """
    
    def __init__(self):
        """Initialize the sustainability evaluator."""
        self.energy_tracker = None
        self.carbon_calculator = None
        self.evaluation_results = {}
    
    def set_energy_tracker(self, energy_tracker: EnergyTracker) -> None:
        """
        Set the energy tracker.
        
        Args:
            energy_tracker: EnergyTracker instance
        """
        self.energy_tracker = energy_tracker
    
    def set_carbon_calculator(self, carbon_calculator: CarbonFootprintCalculator) -> None:
        """
        Set the carbon footprint calculator.
        
        Args:
            carbon_calculator: CarbonFootprintCalculator instance
        """
        self.carbon_calculator = carbon_calculator
    
    def evaluate_model(self, model_info: Dict[str, Any], 
                      energy_data: Dict[str, float],
                      carbon_data: Optional[Dict[str, float]] = None,
                      fairness_metrics: Optional[Dict[str, Any]] = None,
                      interpretability_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate model sustainability based on multiple metrics.
        
        Args:
            model_info: Information about the model
            energy_data: Energy consumption data
            carbon_data: Optional carbon footprint data
            fairness_metrics: Optional fairness metrics
            interpretability_metrics: Optional interpretability metrics
            
        Returns:
            Dictionary with sustainability evaluation results
        """
        # Initialize results dictionary
        results = {
            'model_name': model_info.get('model_name', 'Unknown Model'),
            'model_type': model_info.get('model_type', 'Unknown Type'),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'sustainability_metrics': {}
        }
        
        # Add model complexity metrics if available
        if 'complexity' in model_info:
            results['model_complexity'] = model_info['complexity']
        
        # Evaluate energy efficiency
        energy_score = self._evaluate_energy_efficiency(energy_data)
        results['sustainability_metrics']['energy_efficiency'] = energy_score
        
        # Evaluate carbon efficiency if data is provided
        if carbon_data:
            carbon_score = self._evaluate_carbon_efficiency(carbon_data)
            results['sustainability_metrics']['carbon_efficiency'] = carbon_score
        
        # Evaluate fairness if metrics are provided
        if fairness_metrics:
            fairness_score = self._evaluate_fairness(fairness_metrics)
            results['sustainability_metrics']['fairness'] = fairness_score
        
        # Evaluate interpretability if metrics are provided
        if interpretability_metrics:
            interpretability_score = self._evaluate_interpretability(interpretability_metrics)
            results['sustainability_metrics']['interpretability'] = interpretability_score
        
        # Calculate overall sustainability score
        metrics = results['sustainability_metrics']
        overall_score = sum(metrics.values()) / len(metrics)
        results['overall_sustainability_score'] = overall_score
        
        # Store evaluation results
        self.evaluation_results = results
        
        return results
    
    def _evaluate_energy_efficiency(self, energy_data: Dict[str, float]) -> float:
        """
        Evaluate energy efficiency on a scale of 0-100.
        
        Args:
            energy_data: Energy consumption data
            
        Returns:
            Energy efficiency score (0-100)
        """
        # Extract energy consumption
        energy_kwh = energy_data.get('estimated_energy_kwh', 0)
        
        # Define thresholds for scoring
        # These thresholds should be calibrated based on model type and task
        excellent_threshold = 0.1  # kWh
        good_threshold = 1.0  # kWh
        moderate_threshold = 5.0  # kWh
        poor_threshold = 10.0  # kWh
        
        # Calculate score
        if energy_kwh <= excellent_threshold:
            # Linear scale from 90-100 for excellent efficiency
            score = 90 + 10 * (1 - energy_kwh / excellent_threshold)
        elif energy_kwh <= good_threshold:
            # Linear scale from 75-90 for good efficiency
            score = 75 + 15 * (good_threshold - energy_kwh) / (good_threshold - excellent_threshold)
        elif energy_kwh <= moderate_threshold:
            # Linear scale from 50-75 for moderate efficiency
            score = 50 + 25 * (moderate_threshold - energy_kwh) / (moderate_threshold - good_threshold)
        elif energy_kwh <= poor_threshold:
            # Linear scale from 25-50 for poor efficiency
            score = 25 + 25 * (poor_threshold - energy_kwh) / (poor_threshold - moderate_threshold)
        else:
            # Linear scale from 0-25 for very poor efficiency
            score = max(0, 25 * (1 - (energy_kwh - poor_threshold) / poor_threshold))
        
        return score
    
    def _evaluate_carbon_efficiency(self, carbon_data: Dict[str, float]) -> float:
        """
        Evaluate carbon efficiency on a scale of 0-100.
        
        Args:
            carbon_data: Carbon footprint data
            
        Returns:
            Carbon efficiency score (0-100)
        """
        # Extract carbon footprint
        carbon_kg = carbon_data.get('carbon_footprint_kg', 0)
        
        # Define thresholds for scoring
        excellent_threshold = 0.05  # kg CO2
        good_threshold = 0.5  # kg CO2
        moderate_threshold = 2.5  # kg CO2
        poor_threshold = 5.0  # kg CO2
        
        # Calculate score
        if carbon_kg <= excellent_threshold:
            # Linear scale from 90-100 for excellent efficiency
            score = 90 + 10 * (1 - carbon_kg / excellent_threshold)
        elif carbon_kg <= good_threshold:
            # Linear scale from 75-90 for good efficiency
            score = 75 + 15 * (good_threshold - carbon_kg) / (good_threshold - excellent_threshold)
        elif carbon_kg <= moderate_threshold:
            # Linear scale from 50-75 for moderate efficiency
            score = 50 + 25 * (moderate_threshold - carbon_kg) / (moderate_threshold - good_threshold)
        elif carbon_kg <= poor_threshold:
            # Linear scale from 25-50 for poor efficiency
            score = 25 + 25 * (poor_threshold - carbon_kg) / (poor_threshold - moderate_threshold)
        else:
            # Linear scale from 0-25 for very poor efficiency
            score = max(0, 25 * (1 - (carbon_kg - poor_threshold) / poor_threshold))
        
        return score
    
    def _evaluate_fairness(self, fairness_metrics: Dict[str, Any]) -> float:
        """
        Evaluate fairness on a scale of 0-100.
        
        Args:
            fairness_metrics: Fairness metrics data
            
        Returns:
            Fairness score (0-100)
        """
        # Extract fairness metrics
        # We'll focus on demographic parity difference and equal opportunity difference
        fairness_scores = []
        
        for attribute, metrics in fairness_metrics.items():
            # Demographic parity difference (lower is better)
            if 'demographic_parity_difference' in metrics:
                dpd = metrics['demographic_parity_difference']
                # Convert to score (0-100)
                dpd_score = 100 * max(0, min(1, 1 - dpd))
                fairness_scores.append(dpd_score)
            
            # Equal opportunity difference (lower is better)
            if 'equal_opportunity_difference' in metrics:
                eod = metrics['equal_opportunity_difference']
                if eod is not None:
                    # Convert to score (0-100)
                    eod_score = 100 * max(0, min(1, 1 - eod))
                    fairness_scores.append(eod_score)
        
        # Calculate average fairness score
        if fairness_scores:
            return sum(fairness_scores) / len(fairness_scores)
        else:
            return 50  # Default score if no metrics are provided
    
    def _evaluate_interpretability(self, interpretability_metrics: Dict[str, Any]) -> float:
        """
        Evaluate interpretability on a scale of 0-100.
        
        Args:
            interpretability_metrics: Interpretability metrics data
            
        Returns:
            Interpretability score (0-100)
        """
        # Extract interpretability metrics
        model_type = interpretability_metrics.get('model_type', 'unknown')
        
        # Base score on model type
        if model_type in ['linear_regression', 'logistic_regression', 'decision_tree']:
            # Highly interpretable models
            base_score = 85
        elif model_type in ['random_forest', 'gradient_boosting', 'gbm']:
            # Moderately interpretable models
            base_score = 70
        elif model_type in ['neural_network', 'deep_learning', 'lstm', 'transformer']:
            # Less interpretable models
            base_score = 50
        else:
            # Unknown model type
            base_score = 60
        
        # Adjust score based on additional metrics
        adjustments = 0
        
        # Feature importance available
        if interpretability_metrics.get('feature_importance_available', False):
            adjustments += 10
        
        # Local explanations available
        if interpretability_metrics.get('local_explanations_available', False):
            adjustments += 10
        
        # Global explanations available
        if interpretability_metrics.get('global_explanations_available', False):
            adjustments += 5
        
        # Example-based explanations available
        if interpretability_metrics.get('example_explanations_available', False):
            adjustments += 5
        
        # Calculate final score
        final_score = min(100, base_score + adjustments)
        
        return final_score
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """
        Get the latest evaluation results.
        
        Returns:
            Dictionary with evaluation results
        """
        return self.evaluation_results
    
    def generate_sustainability_report(self, output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Generate a sustainability report.
        
        Args:
            output_format: Format of the report ('dict', 'json', 'text')
            
        Returns:
            Sustainability report in the specified format
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        # Create report dictionary
        report = {
            'model_info': {
                'name': self.evaluation_results.get('model_name', 'Unknown Model'),
                'type': self.evaluation_results.get('model_type', 'Unknown Type')
            },
            'sustainability_scores': self.evaluation_results['sustainability_metrics'],
            'overall_sustainability_score': self.evaluation_results['overall_sustainability_score'],
            'timestamp': self.evaluation_results.get('timestamp', time.strftime('%Y-%m-%d %H:%M:%S')),
            'recommendations': self._generate_recommendations()
        }
        
        # Add model complexity if available
        if 'model_complexity' in self.evaluation_results:
            report['model_complexity'] = self.evaluation_results['model_complexity']
        
        # Return report in the specified format
        if output_format == 'dict':
            return report
        elif output_format == 'json':
            return json.dumps(report, indent=2)
        elif output_format == 'text':
            return self._format_report_as_text(report)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate recommendations for improving sustainability.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        metrics = self.evaluation_results['sustainability_metrics']
        
        # Energy efficiency recommendations
        if 'energy_efficiency' in metrics:
            score = metrics['energy_efficiency']
            if score < 50:
                recommendations.append("Consider using a more energy-efficient model architecture.")
                recommendations.append("Optimize hyperparameters for energy efficiency.")
                recommendations.append("Use quantization or pruning to reduce model size and energy consumption.")
            elif score < 75:
                recommendations.append("Further optimize model for energy efficiency.")
                recommendations.append("Consider using early stopping to reduce training time.")
            
        # Carbon efficiency recommendations
        if 'carbon_efficiency' in metrics:
            score = metrics['carbon_efficiency']
            if score < 60:
                recommendations.append("Train models in regions with lower carbon intensity.")
                recommendations.append("Use renewable energy sources for training.")
            
        # Fairness recommendations
        if 'fairness' in metrics:
            score = metrics['fairness']
            if score < 70:
                recommendations.append("Implement bias mitigation techniques.")
                recommendations.append("Increase diversity in training data.")
                recommendations.append("Use fairness constraints during model training.")
            
        # Interpretability recommendations
        if 'interpretability' in metrics:
            score = metrics['interpretability']
            if score < 60:
                recommendations.append("Use more interpretable model architectures.")
                recommendations.append("Implement feature importance analysis.")
                recommendations.append("Add local explanation capabilities.")
        
        # Add general recommendations
        recommendations.append("Regularly monitor and report sustainability metrics.")
        recommendations.append("Consider the full lifecycle impact of AI models.")
        
        return recommendations
    
    def _format_report_as_text(self, report: Dict[str, Any]) -> str:
        """
        Format report as text.
        
        Args:
            report: Report dictionary
            
        Returns:
            Formatted text report
        """
        text = []
        
        # Add header
        text.append("=" * 50)
        text.append(f"SUSTAINABILITY REPORT: {report['model_info']['name']}")
        text.append("=" * 50)
        text.append(f"Model Type: {report['model_info']['type']}")
        text.append(f"Timestamp: {report['timestamp']}")
        text.append("")
        
        # Add sustainability scores
        text.append("-" * 50)
        text.append("SUSTAINABILITY SCORES")
        text.append("-" * 50)
        for metric, score in report['sustainability_scores'].items():
            text.append(f"{metric.replace('_', ' ').title()}: {score:.2f}/100")
        text.append("")
        text.append(f"Overall Sustainability Score: {report['overall_sustainability_score']:.2f}/100")
        text.append("")
        
        # Add model complexity if available
        if 'model_complexity' in report:
            text.append("-" * 50)
            text.append("MODEL COMPLEXITY")
            text.append("-" * 50)
            for key, value in report['model_complexity'].items():
                text.append(f"{key.replace('_', ' ').title()}: {value}")
            text.append("")
        
        # Add recommendations
        text.append("-" * 50)
        text.append("RECOMMENDATIONS")
        text.append("-" * 50)
        for i, recommendation in enumerate(report['recommendations'], 1):
            text.append(f"{i}. {recommendation}")
        
        return "\n".join(text)
    
    def plot_sustainability_scores(self, output_path: Optional[str] = None) -> None:
        """
        Plot sustainability scores.
        
        Args:
            output_path: Optional path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model() first.")
        
        metrics = self.evaluation_results['sustainability_metrics']
        
        # Create plot
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Convert category names to title case
        categories = [cat.replace('_', ' ').title() for cat in categories]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, values, color=['#2C8ECF', '#2CA02C', '#D62728', '#9467BD'][:len(categories)])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Add reference lines
        plt.axhline(y=80, color='green', linestyle='--', alpha=0.7)
        plt.text(0, 81, 'Good', color='green', ha='center')
        
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
        plt.text(0, 61, 'Moderate', color='orange', ha='center')
        
        plt.axhline(y=40, color='red', linestyle='--', alpha=0.7)
        plt.text(0, 41, 'Poor', color='red', ha='center')
        
        # Set y-axis limits
        plt.ylim(0, 105)
        
        # Add labels and title
        plt.ylabel('Score (0-100)')
        plt.title('Sustainability Metrics')
        
        # Add overall score
        overall_score = self.evaluation_results['overall_sustainability_score']
        plt.figtext(0.5, 0.01, f'Overall Sustainability Score: {overall_score:.1f}',
                  ha='center', fontsize=12, bbox=dict(facecolor='#E6E6E6', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save or show plot
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def save_evaluation_results(self, filepath: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
    
    @classmethod
    def load_evaluation_results(cls, filepath: str) -> 'SustainabilityEvaluator':
        """
        Load evaluation results from a JSON file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            SustainabilityEvaluator instance with loaded results
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        evaluator = cls()
        evaluator.evaluation_results = results
        
        return evaluator


