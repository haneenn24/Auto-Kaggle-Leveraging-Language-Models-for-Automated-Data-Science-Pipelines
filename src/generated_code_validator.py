import ast
import subprocess
import os

def log_message(log_file, check, description, result):
    with open(log_file, 'a') as log:
        log.write(f"Check: {check}\n")
        log.write(f"Description: {description}\n")
        log.write(f"Result: {result}\n\n")

def check_syntax(script, log_file):
    try:
        with open(script, 'r') as file:
            source_code = file.read()
        compile(source_code, script, 'exec')
        log_message(log_file, "Syntax Check", "Check if the script has valid Python syntax.", "Pass")
    except SyntaxError as e:
        log_message(log_file, "Syntax Check", f"Check if the script has valid Python syntax. Error: {e}", "Fail")

def check_formatting(script, log_file):
    try:
        result = subprocess.run(['flake8', script], capture_output=True, text=True)
        if result.returncode == 0:
            log_message(log_file, "Code Formatting", "Check for PEP 8 compliance using flake8.", "Pass")
        else:
            log_message(log_file, "Code Formatting", f"Check for PEP 8 compliance using flake8. Issues:\n{result.stdout}", "Fail")
    except Exception as e:
        log_message(log_file, "Code Formatting", f"Check for PEP 8 compliance using flake8. Error: {e}", "Fail")

def check_api_calls(script, log_file):
    try:
        with open(script, 'r') as file:
            source_code = file.read()
        tree = ast.parse(source_code)
        api_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if api_calls:
            log_message(log_file, "API Calls Check", "Check for presence of API calls in the script.", "Pass")
        else:
            log_message(log_file, "API Calls Check", "Check for presence of API calls in the script. No API calls found.", "Fail")
    except Exception as e:
        log_message(log_file, "API Calls Check", f"Check for presence of API calls in the script. Error: {e}", "Fail")

def check_comments(script, log_file):
    try:
        with open(script, 'r') as file:
            lines = file.readlines()
        comment_lines = [line for line in lines if line.strip().startswith('#')]
        if comment_lines:
            log_message(log_file, "Comments Check", "Check for presence of comments in the script.", "Pass")
        else:
            log_message(log_file, "Comments Check", "Check for presence of comments in the script. No comments found.", "Fail")
    except Exception as e:
        log_message(log_file, "Comments Check", f"Check for presence of comments in the script. Error: {e}", "Fail")

def check_function_docstrings(script, log_file):
    try:
        with open(script, 'r') as file:
            source_code = file.read()
        tree = ast.parse(source_code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        undocumented_functions = [func.name for func in functions if not ast.get_docstring(func)]
        if undocumented_functions:
            log_message(log_file, "Function Docstrings Check", f"Check for presence of docstrings in functions. Functions missing docstrings: {undocumented_functions}", "Fail")
        else:
            log_message(log_file, "Function Docstrings Check", "Check for presence of docstrings in functions.", "Pass")
    except Exception as e:
        log_message(log_file, "Function Docstrings Check", f"Check for presence of docstrings in functions. Error: {e}", "Fail")

def validate_script(script):
    log_file = 'validation_log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    check_syntax(script, log_file)
    check_formatting(script, log_file)
    check_api_calls(script, log_file)
    check_comments(script, log_file)
    check_function_docstrings(script, log_file)
    
    print(f"Validation completed. Check the log file: {log_file}")

if __name__ == "__main__":
    script_to_validate = 'script_to_validate.py'  # replace with your script file name
    validate_script(script_to_validate)
