#python src/code_execution.py path_to_your_script.py

import subprocess
import sys
import os
import datetime

def log_message(log_file, message):
    with open(log_file, 'a') as log:
        log.write(message + "\n")

def execute_script(script_path):
    log_file = 'execution_log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    start_time = datetime.datetime.now()
    log_message(log_file, f"Execution started at {start_time}\n")

    try:
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        end_time = datetime.datetime.now()
        duration = end_time - start_time

        log_message(log_file, f"Execution completed at {end_time}")
        log_message(log_file, f"Duration: {duration}\n")
        
        log_message(log_file, "Script Output:\n")
        log_message(log_file, result.stdout)

        log_message(log_file, "Script Errors (if any):\n")
        log_message(log_file, result.stderr)
        
        if result.returncode == 0:
            log_message(log_file, "Script executed successfully.")
        else:
            log_message(log_file, f"Script failed with return code {result.returncode}.")
        
    except Exception as e:
        log_message(log_file, f"An error occurred while executing the script: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python execute_script.py <path_to_script>")
        sys.exit(1)

    script_path = sys.argv[1]
    if not os.path.isfile(script_path):
        print(f"Error: The script {script_path} does not exist.")
        sys.exit(1)

    execute_script(script_path)
    print("Execution completed. Check the log file: execution_log.txt")
