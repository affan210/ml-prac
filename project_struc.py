import os


def create_directory(path):
    if not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path)


def create_file(path, content=""):
    print(f"Creating file: {path}")
    with open(path, "w") as f:
        f.write(content)


def create_ml_project_structure(project_name):
    # Define project structure
    directories = [
        f"{project_name}/data/raw",
        f"{project_name}/data/interim",
        f"{project_name}/data/processed",
        f"{project_name}/data/external",
        f"{project_name}/notebooks",
        f"{project_name}/src/data",
        f"{project_name}/src/features",
        f"{project_name}/src/models",
        f"{project_name}/src/evaluation",
        f"{project_name}/src/visualization",
        f"{project_name}/src/utils",
        f"{project_name}/scripts",
        f"{project_name}/tests",
        f"{project_name}/configs",
        f"{project_name}/logs",
        f"{project_name}/results/figures",
        f"{project_name}/results/reports",
        f"{project_name}/environment",
    ]

    files = {
        f"{project_name}/data/README.md": "Description of data folders and files.",
        f"{project_name}/notebooks/README.md": "Description of notebooks.",
        f"{project_name}/notebooks/01_exploration.ipynb": "",
        f"{project_name}/notebooks/02_preprocessing.ipynb": "",
        f"{project_name}/notebooks/03_modeling.ipynb": "",
        f"{project_name}/src/__init__.py": "",
        f"{project_name}/src/data/__init__.py": "",
        f"{project_name}/src/data/load_data.py": "",
        f"{project_name}/src/data/process_data.py": "",
        f"{project_name}/src/features/__init__.py": "",
        f"{project_name}/src/features/build_features.py": "",
        f"{project_name}/src/models/__init__.py": "",
        f"{project_name}/src/models/train_model.py": "",
        f"{project_name}/src/models/model.py": "",
        f"{project_name}/src/evaluation/__init__.py": "",
        f"{project_name}/src/evaluation/evaluate_model.py": "",
        f"{project_name}/src/visualization/__init__.py": "",
        f"{project_name}/src/visualization/visualize.py": "",
        f"{project_name}/src/utils/__init__.py": "",
        f"{project_name}/src/utils/utils.py": "",
        f"{project_name}/scripts/run_train.py": "",
        f"{project_name}/scripts/run_evaluate.py": "",
        f"{project_name}/scripts/run_visualize.py": "",
        f"{project_name}/tests/__init__.py": "",
        f"{project_name}/tests/test_data.py": "",
        f"{project_name}/tests/test_features.py": "",
        f"{project_name}/tests/test_models.py": "",
        f"{project_name}/tests/test_evaluation.py": "",
        f"{project_name}/tests/test_utils.py": "",
        f"{project_name}/configs/config.yaml": "",
        f"{project_name}/configs/README.md": "Explanation of the configuration files.",
        f"{project_name}/logs/README.md": "Explanation of the log files.",
        f"{project_name}/results/README.md": "Explanation of the results and structure.",
        f"{project_name}/environment/environment.yml": "",
        f"{project_name}/environment/requirements.txt": "",
        f"{project_name}/environment/README.md": "Explanation of environment setup.",
        f"{project_name}/README.md": f"# {project_name}\n\nProject overview and instructions go here.",
        f"{project_name}/.gitignore": """# Ignore virtual environments
venv/

# Ignore Python byte code
*.pyc

# Ignore system-specific files
.DS_Store
""",
    }

    # Create directories
    for directory in directories:
        create_directory(directory)

    # Create files
    for filepath, content in files.items():
        create_file(filepath, content)

    print(f"Created project structure for '{project_name}'.")


def print_directory_structure(root_dir, indent=""):
    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(f"{indent}{item}/")
            print_directory_structure(path, indent + "    ")
        else:
            print(f"{indent}{item}")


if __name__ == "__main__":
    project_name = input("Enter the name of your machine learning project: ").strip()
    create_ml_project_structure(project_name)
    print_directory_structure(project_name)
