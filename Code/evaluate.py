# evaluate.py

import time
import csv
from rag_engine import RAG_Engine

# --- Test Case Definition ---
# A list of dictionaries, each representing a test case.
# 'expected_keywords': A list of substrings that MUST appear in the bot's answer for it to be considered "accurate".
test_cases = [
    {
        "type": "Simple Fact",
        "question": "Who is the project lead for Project Nova?",
        "expected_keywords": ["evelyn reed"]
    },
    {
        "type": "Simple Fact",
        "question": "What is the codename for the backend?",
        "expected_keywords": ["orion"]
    },
    {
        "type": "Simple Fact",
        "question": "What is the project's go-live target date?",
        "expected_keywords": ["december 1", "2025"]
    },
    {
        "type": "Complex Fact",
        "question": "What technologies are being used for the frontend development, also known as Lyra?",
        "expected_keywords": ["react", "typescript"]
    },
    {
        "type": "Complex Fact",
        "question": "Which phase is scheduled for November 15, 2025?",
        "expected_keywords": ["uat", "deployment"]
    },
    {
        "type": "Complex Fact",
        "question": "What is the primary goal of Project Nova?",
        "expected_keywords": ["dashboard", "real-time", "visualization"]
    },
    {
        "type": "Inferential",
        "question": "Who is responsible for the Python development?",
        "expected_keywords": ["david chen"] # Because Python is a backend tech
    },
    {
        "type": "Inferential",
        "question": "What is the next major phase after the backend is completed on September 30?",
        "expected_keywords": ["frontend", "lyra"]
    },
    {
        "type": "Negative Test",
        "question": "What is the total budget for Project Nova?",
        "expected_keywords": ["do not have information", "not available"] # Bot should state it doesn't know
    },
    {
        "type": "Negative Test",
        "question": "Who is the chief designer for the project?",
        "expected_keywords": ["do not have information", "not available"]
    },
    {
        "type": "Negative Test",
        "question": "Is the project currently ahead of schedule?",
        "expected_keywords": ["do not have information", "not available"]
    },
    {
        "type": "Boundary Test",
        "question": "What is the start date?",
        "expected_keywords": ["august 1", "2025"]
    }
]

def check_accuracy(response, keywords):
    """
    Checks if all keywords are present in the response (case-insensitive).
    This is a simple proxy for factual accuracy.
    """
    response_lower = response.lower()
    for keyword in keywords:
        if keyword.lower() not in response_lower:
            return 0  # Not Accurate
    return 1  # Accurate

def run_evaluation():
    """
    Runs the full evaluation suite.
    """
    print("--- Starting Chatbot Evaluation ---")
    
    # Initialize the RAG Engine (this will take a moment)
    try:
        engine = RAG_Engine("project_nova_brief.pdf")
    except Exception as e:
        print(f"Failed to initialize RAG Engine: {e}")
        return

    results = []
    total_time = 0
    total_accurate = 0

    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, test in enumerate(test_cases):
        question = test["question"]
        print(f"[{i+1}/{len(test_cases)}] Testing Question: {question}")
        
        start_time = time.time()
        response = engine.query(question)
        end_time = time.time()
        
        response_time = end_time - start_time
        accuracy = check_accuracy(response, test["expected_keywords"])

        results.append({
            "ID": i + 1,
            "Type": test["type"],
            "Question": question,
            "Bot Response": response.strip(),
            "Accuracy (Automated)": "Correct" if accuracy == 1 else "Incorrect",
            "Response Time (s)": round(response_time, 2)
        })

        total_time += response_time
        total_accurate += accuracy
        print(f"  -> Accuracy: {'Correct' if accuracy == 1 else 'Incorrect'}, Time: {response_time:.2f}s")

    # --- Summary and Report Generation ---
    print("\n--- Evaluation Complete ---")
    
    overall_accuracy = (total_accurate / len(test_cases)) * 100
    average_time = total_time / len(test_cases)
    
    print("\n** Summary **")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_accurate}/{len(test_cases)})")
    print(f"Average Response Time: {average_time:.2f}s")

    # Save results to CSV
    csv_file = "evaluation_results.csv"
    try:
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults have been saved to '{csv_file}'")
    except IOError as e:
        print(f"\nError saving results to CSV: {e}")

if __name__ == "__main__":
    run_evaluation()