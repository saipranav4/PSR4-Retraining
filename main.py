import time

import subprocess

import sys

import inquirer  # Ensure 'pip install inquirer' is run
 
def get_user_confirmation():

    """Interactive prompt to confirm pipeline execution."""

    print("\n" + "█" * 65)

    print(" 🛠️  UNIFIED CLASSIFICATION PIPELINE ORCHESTRATOR")

    print("█" * 65)

    print("  This will sequentially execute:")

    print("  1. Data Ingestion (Pickle -> S3)")

    print("  2. Data Asset Check")

    print("  3. Feature Engineering")

    print("  4. Model Building & Tuning")

    print("  5. Model Metrics Comparison")

    print("-" * 65)
 
    confirm_q = [

        inquirer.List(

            'confirm',

            message="Ready to begin execution?",

            choices=['START PIPELINE', 'ABORT'],

            default='START PIPELINE'

        )

    ]

    conf_ans = inquirer.prompt(confirm_q)

    if not conf_ans or conf_ans['confirm'] == 'ABORT':

        print("❌ Pipeline cancelled by user.")

        sys.exit(0)
 
def run_pipeline():

    get_user_confirmation()

    # Define the exact execution order of your scripts

    steps = [

        ("STEP 1: Data Ingestion to S3", "input_data.py"),

        ("STEP 2: Data Asset Check / EDA", "data_check_main.py"),

        ("STEP 3: Feature Engineering", "feature_engineering.py"),

        ("STEP 4: Model Build & Tuning", "model_build.py"),

        ("STEP 5: Model Metrics & PDF Generation", "model_comparison.py")

    ]

    print(f"\n▶️ Starting execution of {len(steps)} pipeline steps...\n")

    for step_name, script_name in steps:

        print("=" * 60)

        print(f"⚡ Executing: {step_name} [{script_name}]")

        print("=" * 60)

        start_time = time.time()

        try:

            # subprocess.run ensures memory is freed after each script finishes

            subprocess.run([sys.executable, script_name], check=True)

            print(f"\n✅ {step_name} completed successfully | Time: {time.time() - start_time:.2f}s\n")

        except subprocess.CalledProcessError as e:

            print(f"\n❌ CRITICAL ERROR: '{script_name}' failed to execute.")

            print("⚠️ Halting pipeline to protect data integrity.")

            sys.exit(1)

        except FileNotFoundError:

             print(f"\n❌ CRITICAL ERROR: Could not find script '{script_name}' in the current directory.")

             print("⚠️ Halting pipeline.")

             sys.exit(1)

        time.sleep(1) # Brief pause between steps for terminal readability

    print("\n" + "⭐ " * 20)

    print(" UNIFIED CLASSIFICATION PIPELINE FULLY COMPLETED ")

    print("⭐ " * 20 + "\n")
 
if __name__ == "__main__":

    run_pipeline()
 