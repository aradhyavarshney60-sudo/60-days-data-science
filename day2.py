def calculate_average(marks):
    return sum(marks) / len(marks)


def classify_grade(average):
    if average >= 75:
        return "Distinction"
    elif average >= 40:
        return "Pass"
    else:
        return "Fail"


print("Student Marks Analyzer")

marks = []

for i in range(5):
    mark = float(input(f"Enter marks for subject {i + 1}: "))
    marks.append(mark)

average = calculate_average(marks)
result = classify_grade(average)

print("\n----- Result -----")
print("Marks:", marks)
print("Average Marks:", round(average, 2))
print("Result:", result)