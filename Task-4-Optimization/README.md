# Task 4: Bank Branch Budget Optimization

## Project Overview
This project solves a real-world business optimization problem using
**Linear Programming** as part of the CODTECH Data Science Internship.

The objective is to optimally allocate a limited budget across multiple
bank branches to maximize overall profit while satisfying operational
constraints.

---

## Business Problem
A bank has a fixed monthly budget and must distribute it among different
branches. Each branch has:
- A minimum budget requirement
- A maximum budget limit
- A known profit per unit of budget

---

## Objective
Maximize total profit generated from all branches.

---

## Constraints
- Total allocated budget must not exceed available budget
- Each branch must receive budget within its minimum and maximum limits
- Budget allocations must be non-negative

---

## Optimization Technique
- Linear Programming
- Solved using the PuLP library in Python

---

## Results

### Optimal Budget Allocation
| Branch | Budget Allocated |
|------|------------------|
| Branch A | 40 |
| Branch B | 20 |
| Branch C | 40 |

### Maximum Profit
**520 units**

---

## Business Insights
- Branch C receives the highest allocation due to maximum profit return
- Branch B is limited to minimum allocation due to lower profitability
- All constraints are satisfied, ensuring a feasible solution

---

## Technologies Used
- Python
- PuLP (Linear Programming)

---

## How to Run
```bash
cd Task-4-Optimization/src
python budget_optimization.py
