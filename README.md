python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd src
python main.py


cli.py
handles interation with the user throw the command line

nlp_engine.py
implement function to extract entities using SpaCy

context_manager.py
stores user choices and entities for continuity

story_generator.py
use hugging face transformers library to initialise and use t5 model

scenario_tracking.py
track scenario state, make adjustments based on user decisions

main.py
start the program by connecting all components
