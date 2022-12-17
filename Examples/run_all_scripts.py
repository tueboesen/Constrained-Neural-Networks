import os
import subprocess

base_path = os.getcwd()
print('base_path', base_path)
path_to_search = f"{base_path}/./../scripts_to_run/"

# TODO: this might need to be 'python3' in some cases
python_executable = 'python'
print('python_executable:', python_executable)

py_file_list = []
for dir_path, _, file_name_list in os.walk(path_to_search):
    for file_name in file_name_list:
        if file_name.endswith('.py'):
            # add full path, not just file_name
            py_file_list.append(
                os.path.join(dir_path, file_name))

print(f'Found {len(py_file_list)} python files.')
for i, file_path in enumerate(py_file_list):
    print('Running   {:3d} {}'.format(i, file_path))
    subprocess.run([python_executable, file_path])