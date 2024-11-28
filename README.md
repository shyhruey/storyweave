# CS425 - G1T2  
## StoryWeave

### Overview:
StoryWeave is a project developed for the CS425 course, aimed at generating context-based short stories. The project includes a set of trained models, a quantitative evaluation script, a proof of concept for both frontend and backend, and a final dataset that combines multiple story datasets for improved story generation.

### Included Documents and Files:

1. **Trained Models:**
    - **Model 1:** Trained on Context Short Stories dataset.
    - **Model 2:** Trained on Context Short Stories dataset, augmented with Writing Prompts.
    - **Model 3:** Trained on Context Short Stories dataset, augmented with ROCStories.

2. **Proof of Concept:**
    - **Frontend:**
        1. Navigate to the `POC` directory:
            ```bash
            cd POC
            ```
        2. Install the necessary dependencies:
            ```bash
            npm install
            ```
        3. Run the frontend development server:
            ```bash
            npm run dev
            ```

    - **Backend:**
        1. Navigate to the `POC/server` directory:
            ```bash
            cd POC
            cd server
            ```
        2. Run the backend application:
            ```bash
            python app.py
            ```

3. **metrics.py:** A Python script that provides quantitative evaluation metrics for the performance of all three models on the test dataset.

4. **final_dataset.csv:** A cleaned and combined dataset consisting of Context Short Stories and ROCStories, used to train and evaluate the models.
