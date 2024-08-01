import os

def count_files_in_subdirs(base_dir):
    file_counts = {}
    for subdir, _, files in os.walk(base_dir):
        if os.path.dirname(subdir) == base_dir:
            file_counts[os.path.basename(subdir)] = len(files)
    return file_counts

def main():
    base_dir = input("Enter the directory path: ").strip()
    
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(__file__), '..', base_dir)
    
    base_dir = os.path.abspath(base_dir)
    
    file_counts = count_files_in_subdirs(base_dir)
    print(file_counts)

if __name__ == '__main__':
    main()
