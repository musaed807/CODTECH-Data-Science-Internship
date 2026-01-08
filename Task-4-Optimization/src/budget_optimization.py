from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus


def main():
    # -----------------------------
    # 1. Create Optimization Model
    # -----------------------------
    model = LpProblem(
        name="bank-branch-budget-optimization",
        sense=LpMaximize
    )

    # -----------------------------
    # 2. Decision Variables
    # -----------------------------
    x_A = LpVariable("Budget_Branch_A", lowBound=10, upBound=50)
    x_B = LpVariable("Budget_Branch_B", lowBound=20, upBound=60)
    x_C = LpVariable("Budget_Branch_C", lowBound=15, upBound=40)

    # -----------------------------
    # 3. Objective Function
    # -----------------------------
    model += (
        5 * x_A +
        4 * x_B +
        6 * x_C,
        "Total_Profit"
    )

    # -----------------------------
    # 4. Constraints
    # -----------------------------
    model += (
        x_A + x_B + x_C <= 100,
        "Total_Budget_Constraint"
    )

    # -----------------------------
    # 5. Solve Model
    # -----------------------------
    model.solve()

    # -----------------------------
    # 6. Results
    # -----------------------------
    print("Optimization Status:", LpStatus[model.status])
    print("\nOptimal Budget Allocation:")

    for variable in model.variables():
        print(f"{variable.name} = {variable.value()}")

    print("\nMaximum Profit:", model.objective.value())


if __name__ == "__main__":
    main()
