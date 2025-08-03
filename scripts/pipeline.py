#!/usr/bin/env python
"""
Simplified Property Valuation Pipeline.
"""

import os
import sys
import logging
import argparse
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.model_pipeline import ModelTrainingPipeline
from src.config import MODEL_DIR, OUTPUT_DIR, PIPELINE_LOGS_DIR, LOG_MAX_BYTES, LOG_BACKUP_COUNT
from src.utils.logging import (
    configure_logging, ContextLogger, PerformanceLoggerContext,
    LogSummary, log_memory_usage, log_exception
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> str:
    """
    Set up comprehensive logging for the pipeline.
    """
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
        exclude_modules=['matplotlib.*', 'PIL.*']
    )


def run_pipeline(use_latest_data=False, validate_data=True, model_type='gradient_boosting', save_model=True):
    """Run the complete ML pipeline."""
    # Create log summary and context logger
    summary = LogSummary("Pipeline Execution Summary")
    summary.add_info("Pipeline started")

    ctx_logger = ContextLogger(logger, {
        'model_type': model_type,
        'use_latest_data': str(use_latest_data),
        'validate_data': str(validate_data)
    })

    log_memory_usage(logger)
    start_time = time.time()
    ctx_logger.info("Starting the property valuation pipeline")

    try:
        # Step 1: Process data
        ctx_logger.info("STEP 1: RUNNING THE DATA PIPELINE")
        with PerformanceLoggerContext(logger, "Data Pipeline"):
            data_pipeline = DataPipeline()

            train_data = data_pipeline.run('train', use_latest_data, validate_data)
            test_data = data_pipeline.run('test', use_latest_data, validate_data)

            ctx_logger.info(f"Data processed - Train: {train_data.shape}, Test: {test_data.shape}")
            summary.add_info(f"Data processed: {train_data.shape[0]} train, {test_data.shape[0]} test samples")

        # Step 2: Prepare features
        ctx_logger.info("STEP 2: PREPARING FEATURES")
        with PerformanceLoggerContext(logger, "Feature Preparation"):
            features = [col for col in train_data.columns if col not in ['id', 'price']]
            X_train, y_train = train_data[features], train_data['price'].values
            X_test, y_test = test_data[features], test_data['price'].values

            ctx_logger.info(f"Features prepared: {len(features)} features selected")
            summary.add_info(f"Selected {len(features)} features for modeling")

        # Step 3: Train model
        ctx_logger.info(f"STEP 3: TRAINING {model_type.upper()} MODEL")
        with PerformanceLoggerContext(logger, "Model Training"):
            model_pipeline = ModelTrainingPipeline()
            results = model_pipeline.run(X_train, y_train, X_test, y_test, model_type,
                                       save_model=save_model)

            # Log results
            metrics = results['evaluation_results']['evaluation_results']
            ctx_logger.info("Model evaluation results:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    ctx_logger.info(f"  {metric}: {value:.4f}")
                    summary.add_info(f"Model {metric}: {value:.4f}")
                else:
                    ctx_logger.info(f"  {metric}: {value}")

            if save_model and results.get('model_path'):
                ctx_logger.info(f"Model saved to: {results['model_path']}")
                summary.add_info(f"Model saved to: {os.path.basename(results['model_path'])}")

        total_time = time.time() - start_time
        ctx_logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
        summary.add_info(f"Pipeline completed in {total_time:.2f}s")

        # Log summary
        summary.log_summary(logger)
        log_memory_usage(logger, "Final memory usage")

        return results

    except Exception as e:
        summary.add_error(f"Pipeline failed: {str(e)}")
        log_exception(logger, "Pipeline execution failed", e)
        summary.log_summary(logger)
        raise


def main():
    parser = argparse.ArgumentParser(description='Property Valuation Pipeline')
    parser.add_argument('--use-latest-data', action='store_true', help='Use latest processed data')
    parser.add_argument('--validate-data', action='store_true', help='Validate data against schema')
    parser.add_argument('--model-type', default='gradient_boosting',
                       choices=['gradient_boosting', 'random_forest', 'linear_regression'],
                       help='Type of model to train')
    parser.add_argument('--no-save-model', action='store_true', help='Do not save model')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str, default=None, help='Path to log file')

    args = parser.parse_args()

    # Set up logging with proper configuration
    log_file = setup_logging(args.log_level, args.log_file)

    # Create main context logger
    main_ctx = ContextLogger(logger, {
        'execution_mode': 'command_line',
        'model_type': args.model_type
    })

    main_ctx.info("Starting pipeline execution with command line arguments")
    log_memory_usage(logger, "Initial memory usage")

    try:
        with PerformanceLoggerContext(logger, "Complete Pipeline Execution"):
            results = run_pipeline(
                use_latest_data=args.use_latest_data,
                validate_data=args.validate_data,
                model_type=args.model_type,
                save_model=not args.no_save_model
            )

        # Print console summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        metrics = results['evaluation_results']['evaluation_results']
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        print(f"\nLog file: {log_file}")
        print("=" * 60)

    except Exception as e:
        main_ctx.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()