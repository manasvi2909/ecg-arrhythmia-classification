import os
import subprocess
import sys

def run_script(script_path, args=None):
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        print(f"Completed: {script_path}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Failed: {script_path} (Exit code: {e.returncode})")
        return e.returncode

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(root)
    if not os.path.exists("data/processed/X.npy"):
        print("Preprocessing data first...")
        run_script("src/data/preprocess.py", ["-m"])
    
    # 1. Generate Dataset Analysis Visuals
    print("\n[1/3] Generating Dataset Analysis Visuals...")
    run_script("src/data/analyze.py", ["-m"])
    
    # 2. Generate Model Performance Visuals (Confusion Matrix, Loss Curve)
    if os.path.exists("results/models/best_cnn.pth"):
        print("\n[2/3] Generating Model Performance Visuals (from existing best model)...")
        run_script("src/train/demo_cnn.py")
    else:
        print("\n[2/3] No best_cnn.pth found. Running brief training to generate visuals...")
        run_script("src/train/train_cnn.py", ["-m"])
    
    # Summary
    fig_dir = "results/figures"
    print("\n" + "="*50)
    print("ALL VISUALS GENERATED")
    print("="*50)
    print(f"Location: {os.path.abspath(fig_dir)}")
    files = os.listdir(fig_dir)
    for f in sorted(files):
        print(f"  • {f}")
    print("="*50)

if __name__ == "__main__":
    main()
