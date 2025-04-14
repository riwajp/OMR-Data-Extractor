import csv

import numpy as np

from ast import literal_eval



def generateResults(SESSION,test_type):
    # Function to read the first CSV (answers from images)
    def readSheetsData(filename):
        with open(filename, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # Skip header
            data = [row for row in csv_reader]
        return data

    # Function to read the second CSV (question data with options and correct answers)
    def readQuestionsData(filename):
        with open(filename, mode='r') as file:
            csv_reader = csv.reader(file)
            headers = next(csv_reader)  # Skip header
            data = [row for row in csv_reader]
        return data


    def extractRollNo(text: str):
        before_dot = text.rsplit('.', 1)[0]      # Everything before the last '.'
        after_underscore = before_dot.rsplit('_', 1)[-1]  # Everything after the last '_' in the trimmed text
        return after_underscore


    # Function to match answers from first CSV with the correct ones from the second CSV
    def compare_answers(sheets_data, questions_data):
        results = []
        
        for row in sheets_data:
           
            file_name = row[0]
            roll_no=extractRollNo(file_name)
          
            set_number = (row[1])
            user_answers = [None] *5  # QN1-QN5 answers

            

            # Fill values at the correct positions
            for d in literal_eval(row[2]):
                user_answers[d[0] - 1] = d[1]

            test_type=row[0].split("_")[1]
            session=row[0].split("_")[0]

        
            if(set_number=="") :
                continue
            if(session=="normalization"):
                break

            # Filter data for this set in the second CSV
            set_questions = [q for q in questions_data if (int(q[0][-1]) == int(set_number) and test_type in q[0] and session in q[0])]
            
          
            
          
            
            # Compare user answers with correct answers
            correct_answers = [q[6] for q in set_questions]  # Correct options from second CSV
            
            correct_flags = [(user_answers[i] == correct_answers[i]) for i in range(5)]
            print("==============")
            print(correct_answers)
            print(row[2])
            print(correct_flags)
            print("==============")
            
            # Add the result for this file
            results.append([row[0], set_number] + correct_flags)
        
        return results

    # Write the final results to a new CSV
    def write_results_to_csv(results, output_filename):
        with open(output_filename, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['image', 'set', 'q1', 'q2', 'q3', 'q4','q5'])
            csv_writer.writerows(results)

    # File paths
    first_csv_file = f"./answers-data.csv"  # This file contains the answers from images
    second_csv_file = './questions.csv'  # This file contains the questions with correct answers
    output_csv_file = f'checked-data.csv'  # Final output file with correctness of answers

    # Read both CSVs
    sheets_data = readSheetsData(first_csv_file)
    questions_data = readQuestionsData(second_csv_file)

    # Compare answers and generate results
    comparison_results = compare_answers(sheets_data, questions_data)

    # Write results to the output CSV
    write_results_to_csv(comparison_results, output_csv_file)

    print(f"Comparison results have been written to {output_csv_file}.")

generateResults("er_pretest_lab","pretest")
generateResults("er_posttest_lab","posttest")
generateResults("er_pretest_watrin","pretest")
generateResults("er_posttest_watrin","posttest")