import subprocess

# Execute the Python script and capture its output
# shellcheck disable=SC1036
result = subprocess.run(['python', 'std.py'], stdout=subprocess.PIPE, text=True)
script_output = result.stdout.strip()

# Read the contents of the output file
with open('std.out', 'r') as file:
    expected_output = file.read().strip()

# Compare the outputs
if script_output == expected_output:
    print("The output of std.py is consistent with the contents of std.out")
else:
    print("The output of std.py does NOT match the contents of std.out")
