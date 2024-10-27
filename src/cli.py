def get_user_theme():
    print("Please select a theme:")
    print("1. Fantasy")
    print("2. Sci-Fi")
    print("3. Mystery")
    return input ("Enter your choice: ")

def get_user_choice(options):
    print("What would you like to do?")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    return input("Enter your choice: ")

def get_free_text_input(prompt):
    return input(f"\n{prompt}: ")