# src/main.py
from cli import get_free_text_input
from nlp_engine import extract_entities
from context_manager import ContextManager
from story_generator import generate_story_segment
from scenario_tracker import ScenarioTracker

def main():
    context_manager = ContextManager()
    scenario_tracker = ScenarioTracker()

    # start conversation with an initial theme or story setup
    initial_theme = "You are in an Enchanted Forest with mysterious creatures."
    scenario_tracker.set_scene(initial_theme)
    context_manager.add_to_history(initial_theme)
    print(f"\n{initial_theme} - A new adventure begins!")

    # story progression loop
    while True:
        # generate story and choices
        current_context = context_manager.get_context()
        story_segment, choices = generate_story_segment(current_context)
        
        # print the generated story
        print(f"\nStory: {story_segment}")

        # update context with the generated story
        context_manager.add_to_history(story_segment)

        # print choices
        print("\nWhat would you like to do next?")
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")
        
        user_choice_index = input(f"Enter choice number (1-{len(choices)}): ")
        try:
            user_choice = choices[int(user_choice_index) - 1]
        except (IndexError, ValueError):
            print("Invalid choice. Please try again.")
            continue
        
        # update scenario based on choice
        scenario_tracker.set_scene(user_choice)
        context_manager.add_to_history(user_choice)

        # extract entities (temp))
        free_text = get_free_text_input("Any additional thoughts or actions?")
        entities = extract_entities(free_text)
        for entity in entities:
            context_manager.add_entity(*entity)

        # check for story end condition
        if "End the story" in choices:
            print("\nThe adventure concludes. Thank you for playing!")
            break
