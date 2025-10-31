# /// script
# dependencies = [
#   "smolagents @ git+https://github.com/huggingface/smolagents.git","numpy"]
# ///

# Your actual code here
from smolagents.local_python_executor import LocalPythonExecutor

custom_executor = LocalPythonExecutor(["numpy"])

def run_capture_exception(command: str):
    try:
        custom_executor(harmful_command)
    except Exception as e:
        print("ERROR:\n", e)

# Example 1: non-defined command
# In Jupyter it works
# !echo Bad command

# In our interpreter, it does not.
harmful_command="!echo Bad command"
run_capture_exception(harmful_command)

# Example 2: os not imported
harmful_command="""
import os
exit_code = os.system("echo Bad command")
"""
run_capture_exception(harmful_command)

# Example 3: random._os.system not imported
harmful_command="""
import random
random._os.system('echo Bad command')
"""
run_capture_exception(harmful_command)

# Example 4: infinite loop
harmful_command="""
while True:
    pass
"""
run_capture_exception(harmful_command)

