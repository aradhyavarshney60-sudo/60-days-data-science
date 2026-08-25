name = input("Enter your name: ")
age = int(input("Enter your age: "))
salary = float(input("Enter your monthly salary: "))

yearly_salary = salary * 12

user_data = {
    "name": name,
    "age": age,
    "monthly_salary": salary,
    "yearly_salary": yearly_salary
}

print("\nUser Details:")
print(user_data)