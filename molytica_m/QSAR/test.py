import subprocess

# Command to open gnome-terminal, execute 'pwd', and then wait for user input
command = 'gnome-terminal -- /bin/bash -c "pwd; echo Press enter to exit; read"'
subprocess.run(command, shell=True)
