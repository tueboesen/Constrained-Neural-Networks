import os
import subprocess
"""
A small python script to run a bunch of different simulations sequentially.
The script could easily be parallelized if you want to run several simultaneously.
"""


base_path = os.getcwd()
print('base_path', base_path)
path_to_search = f"{base_path}/./../scripts_to_run/"

python_executable = 'python'
print('python_executable:', python_executable)

py_file_list = []
for dir_path, _, file_name_list in os.walk(path_to_search):
    for file_name in file_name_list:
        if file_name.endswith('.py'):
            py_file_list.append(
                os.path.join(dir_path, file_name))

print(f'Found {len(py_file_list)} python files.')
for i, file_path in enumerate(py_file_list):
    print('Running   {:3d} {}'.format(i, file_path))
    subprocess.run([python_executable, file_path])