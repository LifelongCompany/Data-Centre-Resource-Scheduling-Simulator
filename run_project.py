import os
import sys
import subprocess

def install_dependencies():
    packages = ["pandas", "numpy", "simpy", "matplotlib", "seaborn"]
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def setup_directories():
    os.makedirs("outputs/plots", exist_ok=True)
    print("Directories outputs/ and outputs/plots/ created/verified.")

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    try:
        subprocess.check_call([sys.executable, script_name])
        print(f"--- Successfully finished {script_name} ---")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}. Exiting.")
        sys.exit(e.returncode)

if __name__ == "__main__":
    install_dependencies()
    setup_directories()
    run_script("main_simulation.py")
    run_script("visualizer.py")
    print("\nProject run completed successfully.")
