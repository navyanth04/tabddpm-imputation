import os

def print_folder_structure(path, indent=""):
    for item in sorted(os.listdir(path)):
        full_path = os.path.join(path, item)
        if os.path.isdir(full_path):
            print(f"{indent}[DIR]  {item}/")
            print_folder_structure(full_path, indent + "    ")
        else:
            print(f"{indent}[FILE] {item}")

# Example usage
root_folder = "/"  # change this to the folder you want to inspect
print_folder_structure(root_folder)
