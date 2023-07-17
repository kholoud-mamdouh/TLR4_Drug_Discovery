import subprocess

subprocess.check_call(['pip', 'install', 'chembl_webresource_client'])
subprocess.check_call(['pip', 'install', 'rdkit'])


# Display first 5 lines of the molecule.smi file
subprocess.run(['cat', 'molecule.smi'], check=True, text=True, capture_output=True)


# Count the number of lines in the molecule.smi file
subprocess.run(['wc', '-l', 'molecule.smi'], check=True, text=True, capture_output=True)


# Display the contents of the padel.sh file
subprocess.run(['cat', 'padel.sh'], check=True, text=True, capture_output=True)


# Execute the padel.sh script
subprocess.run(['bash', 'padel.sh'], check=True, text=True, capture_output=True)
