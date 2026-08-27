import csv

grade_count = {}

with open("students.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        grade = row["grade"]

        if grade in grade_count:
            grade_count[grade] += 1
        else:
            grade_count[grade] = 1


print("Grade Frequency:")

for grade, count in grade_count.items():
    print(f"Grade {grade}: {count} students")


with open("output.txt", "w") as file:
    file.write("Grade Frequency:\n")

    for grade, count in grade_count.items():
        file.write(f"Grade {grade}: {count} students\n")


print("\nSummary saved to output.txt")