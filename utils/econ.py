import random from datetime import date

class Person:
    def init(self, name):
        self.name = name

class Government(Person):
    def init(self, name):
        super().init(name)

    def tax_rate(self, rate):
        return rate

class CentralBank(Person):
    def init(self, name):
        super().init(name)
        self.rate_of_return = 0.02
        self.reserve_ratio = 0.2

def raise_interest_rates(self, amount):
    if amount > 0:
        print(f"Central bank raised interest rates by {amount} percent")
    else:
        print("Central bank lowered interest rates")

class CommercialBank(Person):
    def init(self, name, capital):
        super().init(name)
        self.capital = capital
        self.interest_rate = 0

def deposit(self, amount):
    if amount > 0:
        self.capital += amount
        print(f"Deposited {amount} to the bank")
    else:
        print("Invalid deposit amount")

class Business(Person):
    def init(self, name, capital, employees):
        super().init(name)
        self.capital = capital,
        self.employees = employees

def invest(self, amount):
    if amount > 0:
        self.capital += amount
        print(f"Invested {amount} in the business")
    else:
        print("Invalid investment amount")

class Public(Person): def init(self, name): super().init(name)

def main():
    government = Government("Government")
    central_bank = CentralBank("Central Bank")
    commercial_banks = [CommercialBank(f"Bank {i}", random.randint(1000000, 2000000)) for i in range(5)]
    businesses = [Business(f"Company {i}", random.randint(50000, 100000), random.randint(10, 50)) for i in range(3)]
    public = Public("Public")

    current_month = date.today().month
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    while True:
        print("\nCurrent month: ", months[current_month - 1])
        print("Government tax rate: ", government.tax_rate(10) + "%")
        print("Central bank interest rate: ", central_bank.rate_of_return * 100, "%")
 
        if current_month == 12:
            break
 
        action = input("\nDo you want to (G)overn, (C)entral bank, (B)usiness, or (P)ublic? ")
 
        if action.upper() == "G":
            government_tax_rate = float(input("Enter tax rate: "))
            current_month += 1
            print(f"Tax rate increased by {government_tax_rate}%")
 
        elif action.upper() == "C":
            interest_rate_increase = random.randint(-10, 10)
            if interest_rate_increase > 0:
                central_bank.raise_interest_rates(interest_rate_increase)
            else:
                central_bank.rate_of_return += abs(interest_rate_increase) / 100
            print(f"Interest rate increased by {interest_rate_increase}%")
 
        elif action.upper() == "B":
            business_name = input("Enter business name: ")
            business_capital = float(input("Enter capital: "))
            business_employees = int(input("Enter number of employees: "))
            businesses.append(Business(business_name, business_capital, business_employees))
            print(f"{businesses[-1].name} created")
 
        elif action.upper() == "P":
            public_name = input("Enter public name: ")
            public.capital += random.randint(1000, 5000)
            print(f"{public_name} received {random.randint(1000, 5000)}")

if __name__ == "__main__":
    main()