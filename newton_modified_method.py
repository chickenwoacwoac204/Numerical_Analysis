import numpy as np
from sympy import symbols, Matrix, lambdify
from sympy import cos, sin, exp

def modified_simple_iteration(F_expressions, variables, X0, tol=1e-6, max_iter=50):
    n = len(variables)
    F = Matrix(F_expressions)
    J = F.jacobian(variables)  # Ma trận Jacobian
    
    # Chuyển đổi sang hàm lambda để tính toán số
    F_func = lambdify(variables, F, 'numpy')
    J_func = lambdify(variables, J, 'numpy')
    
    # Tính ma trận xấp xỉ B (nghịch đảo J tại X0)
    J_inv = np.linalg.inv(J_func(*X0))
    
    X = np.array(X0, dtype=float)
    print("Iteration |", "X values", "| Error")
    print("-"*50)
    
    for k in range(max_iter):
        F_val = np.array(F_func(*X), dtype=float).flatten()
        X_new = X - J_inv @ F_val  # Phương pháp lặp cải tiến
        error = np.abs(X_new - X)
        
        print(f"{k+1:^9} | {X_new} | {error.max()}")
        
        if np.all(error < tol):  # Kiểm tra hội tụ
            print("\nConverged!")
            return X_new
        
        X = X_new  # Cập nhật giá trị mới
    
    print("\nMax iterations reached.")
    return X  # Trả về giá trị cuối cùng

# Nhập số biến n
n = int(input("Enter the number of variables: "))
vars_list = symbols(f'x1:{n+1}')  # Tạo danh sách biến x1, x2, ..., xn

# Nhập hệ phương trình phi tuyến
F_exprs = []
print("Enter the equations (in terms of x1, x2, ..., xn):")
for i in range(n):
    F_exprs.append(eval(input(f"Equation {i+1}: "), {**{str(v): v for v in vars_list}, **globals()}))


# Nhập giá trị khởi tạo
X0 = list(map(float, input("Enter initial values (space-separated): ").split()))

# Nhập sai số hoặc số lần lặp tối đa
choice = input("Enter tolerance (default 1e-6) or max iterations: ")
if choice.isdigit():
    max_iter = int(choice)
    tol = 1e-6
else:
    max_iter = 50
    tol = float(choice) if choice else 1e-6

# Gọi hàm giải hệ phương trình
solution = modified_simple_iteration(F_exprs, vars_list, X0, tol, max_iter)
print("\nFinal Solution:", solution)
