"""
Main Dashboard Application

This module provides a Flask application for the interactive visualization dashboard
that demonstrates sustainable AI principles and metrics.
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Import core modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.core.sustainability_metrics import SustainabilityEvaluator, EnergyTracker, CarbonFootprintCalculator
from src.core.ethical_validator import EthicalValidator

app = Flask(__name__)

# Sample data for demonstration
SAMPLE_DATA = {
    'climate': {
        'model_info': {
            'model_type': 'lstm_climate_model',
            'energy_efficiency_constraint': 0.7,
            'complexity': {
                'total_parameters': 28353,
                'memory_footprint_mb': 0.108,
                'layer_counts': {'LSTM': 2, 'Dropout': 2, 'Dense': 2}
            }
        },
        'energy_metrics': {
            'training_energy_kwh': 0.85,
            'inference_energy_kwh': 0.002,
            'carbon_footprint_kg': 0.404,
            'renewable_percentage': 60
        },
        'performance_metrics': {
            'rmse': 0.32,
            'mae': 0.25,
            'r_squared': 0.87
        }
    },
    'energy': {
        'model_info': {
            'model_type': 'gbm_energy_forecaster',
            'interpretability_constraint': 0.6,
            'complexity': {
                'n_estimators': 110,
                'max_depth': 5,
                'max_total_nodes': 3850
            }
        },
        'energy_metrics': {
            'training_energy_kwh': 0.42,
            'inference_energy_kwh': 0.001,
            'carbon_footprint_kg': 0.200,
            'renewable_percentage': 60
        },
        'performance_metrics': {
            'rmse': 0.28,
            'mae': 0.22,
            'r_squared': 0.91
        },
        'feature_importances': {
            'temperature': 0.35,
            'time_of_day': 0.25,
            'day_of_week': 0.15,
            'holiday': 0.10,
            'previous_consumption': 0.08,
            'humidity': 0.07
        }
    },
    'healthcare': {
        'model_info': {
            'model_type': 'fairness_aware_classifier',
            'fairness_constraint': 0.8,
            'interpretability_constraint': 0.7,
            'complexity': {
                'model_type': 'logistic_regression',
                'n_coefficients': 24
            }
        },
        'energy_metrics': {
            'training_energy_kwh': 0.15,
            'inference_energy_kwh': 0.0005,
            'carbon_footprint_kg': 0.071,
            'renewable_percentage': 60
        },
        'performance_metrics': {
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.81,
            'f1_score': 0.80
        },
        'fairness_metrics': {
            'gender': {
                'demographic_parity_difference': 0.05,
                'equal_opportunity_difference': 0.04,
                'group_0_acceptance_rate': 0.48,
                'group_1_acceptance_rate': 0.53
            },
            'age': {
                'demographic_parity_difference': 0.08,
                'equal_opportunity_difference': 0.07,
                'group_0_acceptance_rate': 0.45,
                'group_1_acceptance_rate': 0.53
            }
        }
    }
}

# Initialize sustainability evaluator and ethical validator
sustainability_evaluator = SustainabilityEvaluator()
ethical_validator = EthicalValidator()

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/sustainability')
def sustainability():
    """Render the sustainability metrics page."""
    return render_template('sustainability.html')

@app.route('/ethics')
def ethics():
    """Render the ethical AI page."""
    return render_template('ethics.html')

@app.route('/models')
def models():
    """Render the models showcase page."""
    return render_template('models.html')

@app.route('/documentation')
def documentation():
    """Render the documentation page."""
    return render_template('documentation.html')

@app.route('/api/sustainability_metrics')
def api_sustainability_metrics():
    """API endpoint for sustainability metrics."""
    domain = request.args.get('domain', 'climate')
    
    if domain not in SAMPLE_DATA:
        return jsonify({'error': 'Invalid domain'}), 400
    
    # Get data for the requested domain
    domain_data = SAMPLE_DATA[domain]
    
    # Calculate sustainability score
    energy_data = {
        'estimated_energy_kwh': domain_data['energy_metrics']['training_energy_kwh']
    }
    
    carbon_data = {
        'carbon_footprint_kg': domain_data['energy_metrics']['carbon_footprint_kg']
    }
    
    # Create a simplified sustainability score
    energy_score = 100 * max(0, min(1, 1 - (energy_data['estimated_energy_kwh'] - 0.1) / 9.9))
    carbon_score = 100 * max(0, min(1, 1 - (carbon_data['carbon_footprint_kg'] - 0.05) / 4.95))
    
    # Add interpretability score based on model type
    if domain_data['model_info']['model_type'] == 'fairness_aware_classifier' and domain_data['model_info']['complexity']['model_type'] == 'logistic_regression':
        interpretability_score = 85
    elif domain_data['model_info']['model_type'] == 'gbm_energy_forecaster':
        interpretability_score = 70
    else:
        interpretability_score = 60
    
    # Add fairness score if available
    if 'fairness_metrics' in domain_data:
        fairness_scores = []
        for attr, metrics in domain_data['fairness_metrics'].items():
            dpd_score = 100 * max(0, min(1, 1 - metrics['demographic_parity_difference']))
            eod_score = 100 * max(0, min(1, 1 - metrics['equal_opportunity_difference']))
            attr_score = (dpd_score + eod_score) / 2
            fairness_scores.append(attr_score)
        
        fairness_score = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0
    else:
        fairness_score = None
    
    # Calculate overall score
    scores = [energy_score, carbon_score, interpretability_score]
    if fairness_score is not None:
        scores.append(fairness_score)
    
    overall_score = sum(scores) / len(scores)
    
    # Prepare response
    response = {
        'domain': domain,
        'model_type': domain_data['model_info']['model_type'],
        'energy_metrics': domain_data['energy_metrics'],
        'performance_metrics': domain_data['performance_metrics'],
        'sustainability_scores': {
            'energy_efficiency': energy_score,
            'carbon_efficiency': carbon_score,
            'interpretability': interpretability_score
        },
        'overall_sustainability_score': overall_score
    }
    
    if fairness_score is not None:
        response['sustainability_scores']['fairness'] = fairness_score
    
    if 'feature_importances' in domain_data:
        response['feature_importances'] = domain_data['feature_importances']
    
    if 'fairness_metrics' in domain_data:
        response['fairness_metrics'] = domain_data['fairness_metrics']
    
    return jsonify(response)

@app.route('/api/model_comparison')
def api_model_comparison():
    """API endpoint for model comparison."""
    # Compare sustainability metrics across domains
    domains = list(SAMPLE_DATA.keys())
    
    # Extract metrics for comparison
    energy_values = [data['energy_metrics']['training_energy_kwh'] for data in SAMPLE_DATA.values()]
    carbon_values = [data['energy_metrics']['carbon_footprint_kg'] for data in SAMPLE_DATA.values()]
    
    # Calculate sustainability scores
    energy_scores = [100 * max(0, min(1, 1 - (e - 0.1) / 9.9)) for e in energy_values]
    carbon_scores = [100 * max(0, min(1, 1 - (c - 0.05) / 4.95)) for c in carbon_values]
    
    # Interpretability scores
    interpretability_scores = []
    for domain in domains:
        if SAMPLE_DATA[domain]['model_info']['model_type'] == 'fairness_aware_classifier' and SAMPLE_DATA[domain]['model_info']['complexity']['model_type'] == 'logistic_regression':
            interpretability_scores.append(85)
        elif SAMPLE_DATA[domain]['model_info']['model_type'] == 'gbm_energy_forecaster':
            interpretability_scores.append(70)
        else:
            interpretability_scores.append(60)
    
    # Performance metrics (normalized for comparison)
    performance_scores = []
    for domain in domains:
        metrics = SAMPLE_DATA[domain]['performance_metrics']
        if 'r_squared' in metrics:
            # For regression models
            performance_scores.append(metrics['r_squared'] * 100)
        else:
            # For classification models
            performance_scores.append(metrics['f1_score'] * 100)
    
    # Prepare response
    response = {
        'domains': domains,
        'metrics': {
            'energy_consumption': {
                'values': energy_values,
                'unit': 'kWh'
            },
            'carbon_footprint': {
                'values': carbon_values,
                'unit': 'kg CO2eq'
            },
            'energy_efficiency_score': {
                'values': energy_scores,
                'unit': 'score (0-100)'
            },
            'carbon_efficiency_score': {
                'values': carbon_scores,
                'unit': 'score (0-100)'
            },
            'interpretability_score': {
                'values': interpretability_scores,
                'unit': 'score (0-100)'
            },
            'performance_score': {
                'values': performance_scores,
                'unit': 'score (0-100)'
            }
        }
    }
    
    return jsonify(response)

@app.route('/api/ethical_assessment')
def api_ethical_assessment():
    """API endpoint for ethical assessment."""
    domain = request.args.get('domain', 'healthcare')
    
    if domain not in SAMPLE_DATA:
        return jsonify({'error': 'Invalid domain'}), 400
    
    # Get data for the requested domain
    domain_data = SAMPLE_DATA[domain]
    
    # Prepare model info for ethical validation
    model_info = {
        'model_type': domain_data['model_info']['model_type'],
        'feature_importances': domain_data.get('feature_importances'),
        'model_parameters': domain_data['model_info']['complexity'],
        'documentation': {
            'purpose': 'Sustainable AI demonstration',
            'limitations': 'This is a simplified model for demonstration purposes',
            'intended_use': f'Educational demonstration of sustainable AI in {domain}',
            'performance_metrics': domain_data['performance_metrics']
        }
    }
    
    # Prepare fairness metrics if available
    fairness_metrics = domain_data.get('fairness_metrics')
    
    # Prepare accountability info
    accountability_info = {
        'data_provenance': {
            'sources': 'Synthetic data for demonstration',
            'collection_methods': 'Generated for educational purposes',
            'preprocessing_steps': 'Standardization and normalization',
            'limitations': 'Not based on real-world data'
        },
        'model_lineage': {
            'development_process': 'Developed for sustainable AI demonstration',
            'training_data': 'Synthetic data',
            'validation_process': 'Cross-validation',
            'version_history': 'v1.0'
        },
        'monitoring_plan': {
            'performance_monitoring': True,
            'drift_detection': True,
            'fairness_monitoring': True,
            'incident_response': True
        },
        'human_oversight': {
            'review_process': 'Regular review by domain experts',
            'override_mechanisms': 'Human-in-the-loop decision making',
            'roles_responsibilities': 'Clearly defined roles for oversight'
        },
        'feedback_mechanisms': {
            'user_feedback': True,
            'stakeholder_input': True,
            'continuous_improvement': True
        }
    }
    
    # Prepare sustainability info
    sustainability_info = {
        'energy_metrics': {
            'training_energy': domain_data['energy_metrics']['training_energy_kwh'],
            'inference_energy': domain_data['energy_metrics']['inference_energy_kwh']
        },
        'carbon_metrics': {
            'carbon_footprint': domain_data['energy_metrics']['carbon_footprint_kg'],
            'renewable_energy_percentage': domain_data['energy_metrics']['renewable_percentage']
        },
        'resource_usage': {
            'memory_efficiency': 0.7,
            'computation_efficiency': 0.8
        },
        'lifecycle_assessment': {
            'model_lifespan': 24,  # months
            'reusability': 0.8
        }
    }
    
    # Perform ethical validation
    validation_results = ethical_validator.validate_model(
        model_info=model_info,
        fairness_metrics=fairness_metrics,
        accountability_info=accountability_info,
        sustainability_info=sustainability_info
    )
    
    # Generate ethical report
    ethical_report = ethical_validator.generate_ethical_report()
    
    # Combine results
    response = {
        'domain': domain,
        'validation_results': validation_results,
        'ethical_report': ethical_report
    }
    
    return jsonify(response)

@app.route('/api/generate_plot')
def api_generate_plot():
    """API endpoint for generating plots."""
    plot_type = request.args.get('type', 'sustainability')
    domain = request.args.get('domain', 'climate')
    
    if domain not in SAMPLE_DATA:
        return jsonify({'error': 'Invalid domain'}), 400
    
    # Create plot based on type
    if plot_type == 'sustainability':
        img_data = generate_sustainability_plot(domain)
    elif plot_type == 'performance':
        img_data = generate_performance_plot(domain)
    elif plot_type == 'feature_importance':
        img_data = generate_feature_importance_plot(domain)
    elif plot_type == 'fairness':
        img_data = generate_fairness_plot(domain)
    else:
        return jsonify({'error': 'Invalid plot type'}), 400
    
    return jsonify({'image_data': img_data})

def generate_sustainability_plot(domain):
    """Generate sustainability metrics plot."""
    domain_data = SAMPLE_DATA[domain]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate sustainability scores
    energy_kwh = domain_data['energy_metrics']['training_energy_kwh']
    carbon_kg = domain_data['energy_metrics']['carbon_footprint_kg']
    
    energy_score = 100 * max(0, min(1, 1 - (energy_kwh - 0.1) / 9.9))
    carbon_score = 100 * max(0, min(1, 1 - (carbon_kg - 0.05) / 4.95))
    
    # Add interpretability score based on model type
    if domain_data['model_info']['model_type'] == 'fairness_aware_classifier' and domain_data['model_info']['complexity']['model_type'] == 'logistic_regression':
        interpretability_score = 85
    elif domain_data['model_info']['model_type'] == 'gbm_energy_forecaster':
        interpretability_score = 70
    else:
        interpretability_score = 60
    
    # Add fairness score if available
    if 'fairness_metrics' in domain_data:
        fairness_scores = []
        for attr, metrics in domain_data['fairness_metrics'].items():
            dpd_score = 100 * max(0, min(1, 1 - metrics['demographic_parity_difference']))
            eod_score = 100 * max(0, min(1, 1 - metrics['equal_opportunity_difference']))
            attr_score = (dpd_score + eod_score) / 2
            fairness_scores.append(attr_score)
        
        fairness_score = sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0
        
        # Plot all four scores
        categories = ['Energy\nEfficiency', 'Carbon\nEfficiency', 'Interpretability', 'Fairness']
        values = [energy_score, carbon_score, interpretability_score, fairness_score]
    else:
        # Plot three scores
        categories = ['Energy\nEfficiency', 'Carbon\nEfficiency', 'Interpretability']
        values = [energy_score, carbon_score, interpretability_score]
    
    # Create bar chart
    bars = ax.bar(categories, values, color=['#2C8ECF', '#2CA02C', '#D62728', '#9467BD'][:len(categories)])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Add reference line at 80 (good score threshold)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.7)
    ax.text(0, 81, 'Good', color='green', ha='center')
    
    # Add reference line at 60 (moderate score threshold)
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
    ax.text(0, 61, 'Moderate', color='orange', ha='center')
    
    # Add reference line at 40 (poor score threshold)
    ax.axhline(y=40, color='red', linestyle='--', alpha=0.7)
    ax.text(0, 41, 'Poor', color='red', ha='center')
    
    # Set y-axis limits
    ax.set_ylim(0, 105)
    
    # Add labels and title
    ax.set_ylabel('Score (0-100)')
    ax.set_title(f'Sustainability Metrics for {domain.capitalize()} Model')
    
    # Add overall score
    overall_score = sum(values) / len(values)
    ax.text(0.5, 0.05, f'Overall Sustainability Score: {overall_score:.1f}',
            ha='center', va='center', transform=fig.transFigure,
            bbox=dict(facecolor='#E6E6E6', alpha=0.8))
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data

def generate_performance_plot(domain):
    """Generate performance metrics plot."""
    domain_data = SAMPLE_DATA[domain]
    metrics = domain_data['performance_metrics']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metrics based on domain
    if domain in ['climate', 'energy']:
        # Regression metrics
        categories = ['RMSE', 'MAE', 'R²']
        values = [metrics['rmse'], metrics['mae'], metrics['r_squared']]
        
        # Create bar chart for RMSE and MAE
        ax.bar([0, 1], values[:2], color=['#2C8ECF', '#2CA02C'])
        
        # Create a second y-axis for R²
        ax2 = ax.twinx()
        ax2.bar(2, values[2], color='#D62728')
        
        # Set axis labels
        ax.set_ylabel('Error (RMSE, MAE)')
        ax2.set_ylabel('R² Score')
        
        # Set axis limits
        ax.set_ylim(0, max(values[:2]) * 1.2)
        ax2.set_ylim(0, 1.1)
        
        # Set x-tick labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(categories)
        
        # Add value labels
        for i, v in enumerate(values[:2]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center')
        ax2.text(2, values[2] + 0.02, f'{values[2]:.3f}', ha='center')
        
    else:
        # Classification metrics
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        
        # Create bar chart
        bars = ax.bar(categories, values, color=['#2C8ECF', '#2CA02C', '#D62728', '#9467BD'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Set y-axis limits
        ax.set_ylim(0, 1.1)
        
        # Set y-axis label
        ax.set_ylabel('Score')
    
    # Add title
    ax.set_title(f'Performance Metrics for {domain.capitalize()} Model')
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data

def generate_feature_importance_plot(domain):
    """Generate feature importance plot."""
    domain_data = SAMPLE_DATA[domain]
    
    # Check if feature importances are available
    if 'feature_importances' not in domain_data:
        # Create a placeholder plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Feature importances not available for this model',
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_data
    
    # Get feature importances
    importances = domain_data['feature_importances']
    
    # Sort importances
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    feature_names = [item[0] for item in sorted_importances]
    importance_values = [item[1] for item in sorted_importances]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(importance_values)), importance_values, align='center')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, i, f'{width:.3f}', va='center')
    
    # Set y-tick labels
    ax.set_yticks(range(len(importance_values)))
    ax.set_yticklabels(feature_names)
    
    # Add labels and title
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importances for {domain.capitalize()} Model')
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data

def generate_fairness_plot(domain):
    """Generate fairness metrics plot."""
    domain_data = SAMPLE_DATA[domain]
    
    # Check if fairness metrics are available
    if 'fairness_metrics' not in domain_data:
        # Create a placeholder plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Fairness metrics not available for this model',
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_data
    
    # Get fairness metrics
    fairness_metrics = domain_data['fairness_metrics']
    
    # Create a figure with multiple subplots
    n_attributes = len(fairness_metrics)
    fig, axes = plt.subplots(n_attributes, 2, figsize=(12, 4 * n_attributes))
    
    # If there's only one attribute, wrap axes in a list
    if n_attributes == 1:
        axes = [axes]
    
    for i, (attr_name, metrics) in enumerate(fairness_metrics.items()):
        # Plot acceptance rates
        axes[i][0].bar([0, 1], [metrics['group_0_acceptance_rate'], metrics['group_1_acceptance_rate']])
        axes[i][0].set_xticks([0, 1])
        axes[i][0].set_xticklabels(['Group 0', 'Group 1'])
        axes[i][0].set_ylabel('Acceptance Rate')
        axes[i][0].set_title(f'{attr_name.capitalize()} - Acceptance Rates')
        
        # Add demographic parity difference
        axes[i][0].text(0.5, 0.9, f"DPD: {metrics['demographic_parity_difference']:.4f}",
                     horizontalalignment='center', transform=axes[i][0].transAxes)
        
        # Plot fairness metrics
        metric_names = ['demographic_parity_difference', 'equal_opportunity_difference']
        metric_values = [metrics['demographic_parity_difference'], 
                       metrics['equal_opportunity_difference'] if metrics['equal_opportunity_difference'] is not None else 0]
        
        axes[i][1].bar(range(len(metric_values)), metric_values)
        axes[i][1].set_xticks(range(len(metric_values)))
        axes[i][1].set_xticklabels(['DPD', 'EOD'])
        axes[i][1].set_ylabel('Difference')
        axes[i][1].set_title(f'{attr_name.capitalize()} - Fairness Metrics')
        
        # Add reference line at 0.1 (common threshold for fairness)
        axes[i][1].axhline(y=0.1, color='r', linestyle='--')
    
    # Add overall title
    fig.suptitle(f'Fairness Metrics for {domain.capitalize()} Model', fontsize=16)
    
    # Convert plot to base64 image
    buf = io.BytesIO()
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return img_data

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5004)