import os

def run_cli():
    print("\n🧪 Running CLI Predictor...\n")
    os.system("python coffee.py")

def run_gui():
    print("\n Launching GUI Predictor...\n")
    os.system("python gui.py")

def main():
    print("===================================")
    print("☕ Coffee Quality Prediction System")
    print("===================================\n")
    print("Choose an option:")
    print("1. CLI Version")
    print("2. GUI Version")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice == '1':
            run_cli()
            break
        elif choice == '2':
            run_gui()
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
