#python pipeline_automation.py script1.py script2.py script3.py

import subprocess
import sys
import os
import datetime

def log_message(log_file, message):
    with open(log_file, 'a') as log:
        log.write(message + "\n")

def execute_script(script_path, log_file):
    start_time = datetime.datetime.now()
    log_message(log_file, f"Execution of {script_path} started at {start_time}\n")

    try:
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        log_message(log_file, f"Execution of {script_path} completed at {end_time}")
        log_message(log_file, f"Duration: {duration}\n")
        
        log_message(log_file, "Script Output:\n")
        log_message(log_file, result.stdout)

        log_message(log_file, "Script Errors (if any):\n")
        log_message(log_file, result.stderr)
        
        if result.returncode == 0:
            log_message(log_file, f"Script {script_path} executed successfully.\n")
        else:
            log_message(log_file, f"Script {script_path} failed with return code {result.returncode}.\n")
        
    except Exception as e:
        log_message(log_file, f"An error occurred while executing the script {script_path}: {str(e)}\n")

def main(scripts):
    log_file = 'pipeline_execution_log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_message(log_file, "Starting pipeline execution.\n")

    for script in scripts:
        if not os.path.isfile(script):
            log_message(log_file, f"Error: The script {script} does not exist.\n")
            print(f"Error: The script {script} does not exist.")
            continue
        execute_script(script, log_file)
    
    log_message(log_file, "Pipeline execution completed.\n")
    print("Pipeline execution completed. Check the log file: pipeline_execution_log.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline_executor.py <script1> <script2> ... <scriptN>")
        sys.exit(1)

    scripts = sys.argv[1:]
    main(scripts)
