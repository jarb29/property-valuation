"""
Main Pipeline Module.

This module serves as the main entry point for the property valuation pipeline.
It orchestrates the data and model pipelines to create a complete workflow.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, List

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.model_pipeline import ModelTrainingPipeline
from src.config import MODEL_DIR, OUTPUT_DIR, PIPELINE_LOGS_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT
from src.utils.logging import (
    configure_logging, ProcessingLogger, ContextLogger,
    PerformanceLoggerContext, LogSummary, log_memory_usage, log_exception
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> str:
    """
    Set up comprehensive logging for the pipeline.

    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (Optional[str]): Path to log file. If None, a timestamped file in PIPELINE_LOGS_DIR is used.

    Returns:
        str: Path to the log file.
    """
    # Use the enhanced centralized logging configuration
    return configure_logging(
        app_name='PropertyValuationPipeline',
        log_level=log_level,
        log_to_console=True,
        log_to_file=True,
        log_file=log_file,
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        date_format='%Y-%m-%d %H:%M:%S,%f',
        use_colors=True,
        max_bytes=LOG_MAX_BYTES,
        backup_count=LOG_BACKUP_COUNT,
        # Exclude some noisy modules from logs
        exclude_modules=['matplotlib.*', 'PIL.*']
    )


def run_pipeline(data_type: str = 'train',
                use_latest_data: bool = False,
                validate_data: bool = True,
                model_type: str = 'gradient_boosting',
                model_params: Optional[Dict[str, Any]] = None,
                slice_features: Optional[List[str]] = None,
                save_model: bool = True,
                model_base_name: str = 'property_valuation',
                output_dir: str = OUTPUT_DIR,
                model_dir: str = MODEL_DIR) -> Dict[str, Any]:
    """
    Run the complete pipeline from data loading to model training and evaluation.

    Args:
        data_type (str): Type of data to process ('train' or 'test').
        use_latest_data (bool): Whether to use the latest processed data or the original data.
        validate_data (bool): Whether to validate the data against a schema.
        model_type (str): Type of model to train.
        model_params (Optional[Dict[str, Any]]): Parameters for the model.
        slice_features (Optional[List[str]]): Features to slice the data by for evaluation.
        save_model (bool): Whether to save the trained model.
        model_base_name (str): Base name for the model file.
        output_dir (str): Directory to save output files.
        model_dir (str): Directory to save model files.

    Returns:
        Dict[str, Any]: Results of the pipeline run.
    """
    # Create a log summary to collect important events
    summary = LogSummary("Pipeline Execution Summary")
    summary.add_info("Pipeline started")

    # Create a context logger for tracking the pipeline execution
    ctx_logger = ContextLogger(logger, {
        'model_type': model_type,
        'data_type': data_type,
        'use_latest_data': str(use_latest_data),
        'validate_data': str(validate_data)
    })

    # Log initial memory usage
    log_memory_usage(logger)

    pipeline_start_time = time.time()
    ctx_logger.info("Starting the property valuation pipeline")

    # Create timing dictionary to track performance
    timings = {}

    # Step 1: Run the data pipeline
    ctx_logger.info("=" * 80)
    ctx_logger.info("STEP 1: RUNNING THE DATA PIPELINE")
    ctx_logger.info("=" * 80)

    # Create a child context logger for the data pipeline step
    data_ctx = ctx_logger.create_child('step', 'data_pipeline')

    # Use PerformanceLoggerContext to measure execution time and memory usage
    with PerformanceLoggerContext(logger, "Data Pipeline"):
        data_pipeline = DataPipeline(output_dir=output_dir)

        # Add to summary
        summary.add_info("Data pipeline started")

        # Train data processing
        data_ctx.info("Processing training data...")
        train_start = time.time()

        try:
            train_data = data_pipeline.run(data_type='train', use_latest=use_latest_data, validate=validate_data)
            train_time = time.time() - train_start
            data_ctx.info(f"Training data processed in {train_time:.2f}s - Shape: {train_data.shape}")
            summary.add_info(f"Training data processed: {train_data.shape[0]} samples, {train_data.shape[1]} features")
        except Exception as e:
            summary.add_error(f"Failed to process training data: {str(e)}")
            log_exception(logger, "Error processing training data", e)
            raise

        # Test data processing
        data_ctx.info("Processing test data...")
        test_start = time.time()

        try:
            test_data = data_pipeline.run(data_type='test', use_latest=use_latest_data, validate=validate_data)
            test_time = time.time() - test_start
            data_ctx.info(f"Test data processed in {test_time:.2f}s - Shape: {test_data.shape}")
            summary.add_info(f"Test data processed: {test_data.shape[0]} samples, {test_data.shape[1]} features")
        except Exception as e:
            summary.add_error(f"Failed to process test data: {str(e)}")
            log_exception(logger, "Error processing test data", e)
            raise

        # Record timing
        step1_time = time.time() - pipeline_start_time
        timings['data_pipeline'] = step1_time
        data_ctx.info(f"Data pipeline completed in {step1_time:.2f}s")
        summary.add_info(f"Data pipeline completed in {step1_time:.2f}s")

    # Step 2: Prepare features and target
    ctx_logger.info("\n" + "=" * 80)
    ctx_logger.info("STEP 2: PREPARING FEATURES AND TARGET")
    ctx_logger.info("=" * 80)

    # Create a child context logger for the feature preparation step
    feature_ctx = ctx_logger.create_child('step', 'feature_preparation')

    # Use PerformanceLoggerContext to measure execution time and memory usage
    with PerformanceLoggerContext(logger, "Feature Preparation"):
        # Add to summary
        summary.add_info("Feature preparation started")

        try:
            features = [col for col in train_data.columns if col not in ['id', 'price']]
            target = 'price'

            X_train = train_data[features]
            y_train = train_data[target].values
            X_test = test_data[features]
            y_test = test_data[target].values

            feature_ctx.info(f"Number of features: {len(features)}")
            feature_ctx.info(f"Target variable: {target}")
            feature_ctx.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            feature_ctx.info(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

            # Add to summary
            summary.add_info(f"Selected {len(features)} features for modeling")
            summary.add_info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

            # Record timing
            step2_time = time.time() - (pipeline_start_time + timings.get('data_pipeline', 0))
            timings['feature_preparation'] = step2_time
            feature_ctx.info(f"Feature preparation completed in {step2_time:.2f}s")
            summary.add_info(f"Feature preparation completed in {step2_time:.2f}s")
        except Exception as e:
            summary.add_error(f"Failed to prepare features: {str(e)}")
            log_exception(logger, "Error preparing features", e)
            raise

    # Step 3: Run the model training pipeline
    ctx_logger.info("\n" + "=" * 80)
    ctx_logger.info("STEP 3: RUNNING THE MODEL TRAINING PIPELINE")
    ctx_logger.info("=" * 80)

    # Create a child context logger for the model training step
    model_ctx = ctx_logger.create_child('step', 'model_training')
    model_ctx.add_context('model_type', model_type)

    # Use PerformanceLoggerContext to measure execution time and memory usage
    with PerformanceLoggerContext(logger, "Model Training Pipeline"):
        # Add to summary
        summary.add_info(f"Model training started with model type: {model_type}")

        # Log model details
        model_ctx.info(f"Model type: {model_type}")
        if model_params:
            model_ctx.info(f"Model parameters: {model_params}")
        if slice_features:
            model_ctx.info(f"Slice features for evaluation: {slice_features}")

        try:
            # Create model pipeline
            model_pipeline = ModelTrainingPipeline(model_dir=model_dir, output_dir=output_dir)

            # Log memory usage before model training
            log_memory_usage(logger, "Memory usage before model training")

            # Run the model pipeline with timing
            model_start = time.time()
            results = model_pipeline.run(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_type=model_type,
                model_params=model_params,
                slice_features=slice_features,
                is_regression=True,
                save_model=save_model,
                base_name=model_base_name
            )
            model_time = time.time() - model_start

            # Log memory usage after model training
            log_memory_usage(logger, "Memory usage after model training")

            # Add evaluation results to summary
            if 'evaluation_results' in results and 'evaluation_results' in results['evaluation_results']:
                eval_results = results['evaluation_results']['evaluation_results']
                for metric, value in eval_results.items():
                    if isinstance(value, float):
                        summary.add_info(f"Model {metric}: {value:.4f}")
                    else:
                        summary.add_info(f"Model {metric}: {value}")

            # Record timing
            step3_time = time.time() - (pipeline_start_time + timings.get('data_pipeline', 0) + timings.get('feature_preparation', 0))
            timings['model_pipeline'] = step3_time
            model_ctx.info(f"Model training pipeline completed in {step3_time:.2f}s")
            summary.add_info(f"Model training completed in {step3_time:.2f}s")
        except Exception as e:
            summary.add_error(f"Failed to train model: {str(e)}")
            log_exception(logger, "Error training model", e)
            raise

    # Step 4: Log the results
    ctx_logger.info("\n" + "=" * 80)
    ctx_logger.info("STEP 4: LOGGING THE RESULTS")
    ctx_logger.info("=" * 80)

    # Create a child context logger for the results logging step
    results_ctx = ctx_logger.create_child('step', 'results_logging')

    # Use PerformanceLoggerContext to measure execution time and memory usage
    with PerformanceLoggerContext(logger, "Results Logging"):
        # Add to summary
        summary.add_info("Results logging started")

        try:
            step4_start = time.time()

            # Log evaluation metrics with formatting
            results_ctx.info("Model evaluation results:")
            for metric, value in results['evaluation_results']['evaluation_results'].items():
                if isinstance(value, float):
                    results_ctx.info(f"  {metric}: {value:.4f}")
                else:
                    results_ctx.info(f"  {metric}: {value}")

            # Log model paths if saved
            if results['model_path']:
                results_ctx.info(f"Model saved to: {results['model_path']}")
                summary.add_info(f"Model saved to: {os.path.basename(results['model_path'])}")
            if results['metadata_path']:
                results_ctx.info(f"Model metadata saved to: {results['metadata_path']}")

            # Log timing information
            results_ctx.info("\nPerformance summary:")
            for step, duration in timings.items():
                results_ctx.info(f"  {step}: {duration:.2f}s")

            total_time = time.time() - pipeline_start_time
            results_ctx.info(f"Total pipeline execution time: {total_time:.2f}s")

            step4_time = time.time() - step4_start
            timings['results_logging'] = step4_time

            # Add final timing to summary
            summary.add_info(f"Total pipeline execution time: {total_time:.2f}s")
        except Exception as e:
            summary.add_error(f"Failed to log results: {str(e)}")
            log_exception(logger, "Error logging results", e)
            raise

    # Log final memory usage
    log_memory_usage(logger, "Final memory usage")

    # Log the summary
    ctx_logger.info("\n" + "=" * 80)
    ctx_logger.info("PIPELINE EXECUTION SUMMARY")
    ctx_logger.info("=" * 80)
    summary.log_summary(logger)

    ctx_logger.info("\n" + "=" * 80)
    ctx_logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    ctx_logger.info("=" * 80)

    return {
        "train_data": train_data,
        "test_data": test_data,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "model_results": results,
        "timings": timings,
        "total_time": total_time,
        "summary": summary
    }


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Property Valuation Pipeline')

    # Data pipeline arguments
    parser.add_argument('--data-type', type=str, default='train', choices=['train', 'test'],
                       help='Type of data to process')
    parser.add_argument('--use-latest-data', action='store_true',
                       help='Use the latest processed data instead of the original data')
    parser.add_argument('--validate-data', action='store_true',
                       help='Validate the data against a schema')

    # Model pipeline arguments
    parser.add_argument('--model-type', type=str, default='gradient_boosting',
                       choices=['gradient_boosting', 'random_forest', 'linear_regression'],
                       help='Type of model to train')
    parser.add_argument('--slice-features', type=str, nargs='+',
                       default=['type', 'sector'],
                       help='Features to slice the data by for evaluation')
    parser.add_argument('--no-save-model', action='store_true',
                       help='Do not save the trained model')
    parser.add_argument('--model-base-name', type=str, default='property_valuation',
                       help='Base name for the model file')

    # Directory arguments
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Directory to save output files')
    parser.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory to save model files')

    # Logging arguments
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Path to log file')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Set up logging with enhanced configuration
    log_file = setup_logging(args.log_level, args.log_file)

    # Create a context logger for the main execution
    main_ctx = ContextLogger(logger, {
        'execution_mode': 'command_line',
        'data_type': args.data_type,
        'model_type': args.model_type
    })

    # Log initial memory usage
    log_memory_usage(logger, "Initial memory usage")

    # Create a processing logger for detailed progress tracking
    proc_logger = ProcessingLogger("pipeline", total_files=0)

    # Log the arguments with both loggers
    proc_logger.log_arguments(
        data_type=args.data_type,
        use_latest_data=args.use_latest_data,
        validate_data=args.validate_data,
        model_type=args.model_type,
        slice_features=args.slice_features,
        save_model=not args.no_save_model,
        model_base_name=args.model_base_name
    )

    main_ctx.info("Starting pipeline execution with command line arguments")

    # Create a log summary for the main execution
    main_summary = LogSummary("Main Execution Summary")
    main_summary.add_info(f"Pipeline started with model type: {args.model_type}")

    # Record start time for performance tracking
    start_time = time.time()

    try:
        # Run the pipeline with performance tracking
        with PerformanceLoggerContext(logger, "Complete Pipeline Execution"):
            results = run_pipeline(
                data_type=args.data_type,
                use_latest_data=args.use_latest_data,
                validate_data=args.validate_data,
                model_type=args.model_type,
                slice_features=args.slice_features,
                save_model=not args.no_save_model,
                model_base_name=args.model_base_name,
                output_dir=args.output_dir,
                model_dir=args.model_dir
            )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Add success information to the main summary
        main_summary.add_info(f"Pipeline completed successfully in {elapsed_time:.2f}s")
        main_summary.add_info(f"Processed {len(results['train_data'])} training samples and {len(results['test_data'])} test samples")

        # Log final memory usage
        log_memory_usage(logger, "Final memory usage after pipeline execution")

        # Log summary with context logger
        main_ctx.info("\nPipeline completed successfully!")
        main_ctx.info(f"Processed {len(results['train_data'])} training samples and {len(results['test_data'])} test samples")
        main_ctx.info(f"Total execution time: {elapsed_time:.2f} seconds")
        main_ctx.info(f"Log file: {log_file}")

        # Print summary to console as well
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Processed {len(results['train_data'])} training samples and {len(results['test_data'])} test samples")

        # Print timing information
        print("\nPerformance summary:")
        if 'timings' in results:
            for step, duration in results['timings'].items():
                print(f"  {step}: {duration:.2f}s")
        print(f"  Total execution time: {elapsed_time:.2f}s")

        # Print model evaluation results
        print("\nModel evaluation results:")
        for metric, value in results['model_results']['evaluation_results']['evaluation_results'].items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

        # Log the main execution summary
        print("\n" + "=" * 80)
        print("MAIN EXECUTION SUMMARY")
        print("=" * 80)
        main_summary.log_summary(logger)

        print(f"\nLog file: {log_file}")

    except Exception as e:
        # Log the error with detailed information
        main_summary.add_error(f"Pipeline execution failed: {str(e)}")
        log_exception(logger, "Pipeline execution failed", e)

        # Log the main execution summary even in case of error
        main_ctx.error("\n" + "=" * 80)
        main_ctx.error("PIPELINE EXECUTION FAILED")
        main_ctx.error("=" * 80)
        main_summary.log_summary(logger)

        # Re-raise the exception
        raise
