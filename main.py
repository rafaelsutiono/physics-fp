import easygui
from basketball_simulation import basketball_shot_simulation
from free_shot_simulation import free_shot_simulation

def run_simulation():
    while True:
        choice = easygui.buttonbox("Choose a simulation:", "Simulation Choice", ("Basketball Shot", "Free Shot", "Exit"))

        if choice == "Basketball Shot":
            if not basketball_shot_simulation():
                break
        elif choice == "Free Shot":
            if not free_shot_simulation():
                break
        elif choice == "Exit":
            break

# Main entry point
run_simulation()