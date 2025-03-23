# LƯU Ý: CHỌN MÔI TRƯỜNG PYTHON ĐÚNG ĐỂ DÙNG ĐƯỢC THƯ VIỆN SYMPY: CTRL + SHIFT + P => SELECT INTERPRETER => PYTHON 3.12.9 ('BASE') LƯU Ở MINICONDA

"""
    Giải hệ phương trình phi tuyến bằng phương pháp Newton cổ điển hoặc Newton Modified.
    
    Args:
        F_expressions (list): Danh sách các biểu thức phi tuyến (SymPy).
        variables (list): Danh sách biến tượng trưng (x1, x2, ...).
        X0 (list): Vectơ giá trị khởi tạo.
        method (str): "newton" (cập nhật Jacobian mỗi bước) hoặc "modified" (Jacobian cố định), mặc định "newton".
        tol (float): Sai số cho phép, mặc định 1e-6.
        max_iter (int): Số lần lặp tối đa, mặc định 50.
    
    Returns:
        numpy.ndarray: Nghiệm cuối cùng nếu hội tụ, hoặc giá trị cuối cùng nếu không hội tụ.
    
    Raises:
        ValueError: Nếu biến vượt miền xác định, phương pháp không hợp lệ, Jacobian suy biến, hoặc nghiệm phát tán.
"""
# ------------- Cú pháp SymPy và miền xác định ----------------------------------------------
# Hàm đa thức: 
# - Ví dụ: x**5 - 0.2*x + 15.0           [x**5 = x^5, dùng ** thay vì ^]
# - Miền xác định: Toàn R (số thực), không có hạn chế trừ khi có mẫu số.

# Hàm lượng giác:
# - Hàm sin: sin(x)                      [Miền: R, kết quả ∈ [-1, 1]]
# - Hàm cos: cos(x)                      [Miền: R, kết quả ∈ [-1, 1]]
# - Hàm tan: tan(x)                      [Miền: R, không xác định tại x = (k + 1/2)π, k ∈ Z (bội số lẻ của π/2)]
# - Hàm cot: cot(x) hoặc 1/tan(x)        [Miền: R, không xác định tại x = kπ, k ∈ Z (tan(x) = 0)]
# - Hàm sec: 1/cos(x)                    [Miền: R, không xác định tại x = (k + 1/2)π, k ∈ Z]
# - Hàm csc: 1/sin(x)                    [Miền: R, không xác định tại x = kπ, k ∈ Z]

# Hàm lượng giác ngược:
# - Hàm arcsin: asin(x) hoặc arcsin(x)   [Miền: x ∈ [-1, 1], kết quả ∈ [-π/2, π/2]]
# - Hàm arccos: acos(x) hoặc arccos(x)   [Miền: x ∈ [-1, 1], kết quả ∈ [0, π]]
# - Hàm arctan: atan(x) hoặc arctan(x)   [Miền: R, kết quả ∈ (-π/2, π/2)]
# - Hàm arccot: acot(x) hoặc 1/atan(x)   [Miền: R, kết quả ∈ (0, π), lưu ý arccot không chuẩn hóa trong SymPy]
# - Hàm arcsec: asec(x)                  [Miền: x ∈ (-∞, -1] ∪ [1, ∞), kết quả ∈ [0, π/2) ∪ (π/2, π]]
# - Hàm arccsc: acsc(x)                  [Miền: x ∈ (-∞, -1] ∪ [1, ∞), kết quả ∈ [-π/2, 0) ∪ (0, π/2]]

# Hàm căn:
# - Hàm căn bậc 2: sqrt(x)               [Miền: x ≥ 0, kết quả ∈ [0, ∞)]
# - Hàm căn bậc n: x**(1/n)              [Miền: x ≥ 0 nếu n chẵn, x ∈ R nếu n lẻ; n là số nguyên dương]
# - Lưu ý: Với n chẵn và x < 0, SymPy trả về số phức, nhưng numpy trả về nan trong chế độ thực.

# Hàm mũ và logarit:
# - Hàm e mũ: exp(x)                     [Miền: R, kết quả ∈ (0, ∞)]
# - Hàm logarit tự nhiên (ln): log(x)    [Miền: x > 0, kết quả ∈ R]
# - Hàm log với cơ số bất kỳ: log(x, base) [Miền: x > 0, base > 0 và base ≠ 1, kết quả ∈ R]
# - Ví dụ: log(x, 10)                    [log cơ số 10]
# - Lưu ý: Nếu x <= 0, SymPy trả về số phức, nhưng numpy trả về nan trong chế độ thực.

# Hàm hyperbolic (tuỳ chọn):
# - Hàm sinh: sinh(x)                    [Miền: R, kết quả ∈ R]
# - Hàm cosh: cosh(x)                    [Miền: R, kết quả ∈ [1, ∞)]
# - Hàm tanh: tanh(x)                    [Miền: R, kết quả ∈ (-1, 1)]

# Lưu ý chung:
# - Các hàm cần được nhập đúng cú pháp SymPy (sin, cos, exp, ...), không dùng math.sin hay np.sin.
# - Khi dùng với Newton Modified, đảm bảo Jacobian khả nghịch và X0 nằm trong miền xác định.
# - Nếu giá trị vượt ra ngoài miền xác định (VD: sqrt(-1), asin(2)), có thể gây lỗi nan hoặc ValueError.
# ------------------------------------------------------------------------------------------------------------

import numpy as np
from sympy import symbols, Matrix, lambdify
from sympy import cos, sin, exp

def newton_solver(F_expressions, variables, X0, method="newton", tol=1e-6, max_iter=50):
    n = len(variables)
    F = Matrix(F_expressions)
    J = F.jacobian(variables)
    
    F_func = lambdify(variables, F, 'numpy')
    J_func = lambdify(variables, J, 'numpy')
    
    X = np.array(X0, dtype=float)
    
    if method == "modified":
        J_at_X0 = J_func(*X0)
        if np.abs(np.linalg.det(J_at_X0)) < 1e-10:
            raise ValueError("Jacobian at X0 is singular (not invertible). Try a different initial guess or use Newton method.")
        J_inv_fixed = np.linalg.inv(J_at_X0)
    
    # Định dạng tiêu đề bảng
    headers = ["Iteration"] + [f"x{i+1}" for i in range(n)] + [f"F{i+1}" for i in range(n)] + ["Max Error", "det(J(X_n))"]
    col_width_iter = 10
    col_width_val = 15
    col_width_error = 15
    col_width_det = 15  # Độ rộng cột det(J(X_n))
    total_width = col_width_iter + 2 * n * col_width_val + col_width_error + col_width_det + (2 * n + 3)
    print("-" * total_width)
    print(f"|{headers[0]:^{col_width_iter}}|", end="")
    for i in range(1, n+1):
        print(f"{headers[i]:^{col_width_val}}|", end="")
    for i in range(n+1, 2*n+1):
        print(f"{headers[i]:^{col_width_val}}|", end="")
    print(f"{headers[2*n+1]:^{col_width_error}}|", end="")
    print(f"{headers[2*n+2]:^{col_width_det}}|")
    print("-" * total_width)
    
    for k in range(max_iter):
        # Kiểm tra miền xác định cho exp
        if any('exp' in str(expr) for expr in F_expressions):
            for i, expr in enumerate(F_expressions):
                if 'exp' in str(expr):
                    if 'x1*x2' in str(expr):
                        arg = -X[0] * X[1]
                        if arg < -100:
                            raise ValueError(f"Iteration {k+1}: Argument of exp in equation {i+1} too negative ({arg}), may cause divergence.")

        F_val = np.array(F_func(*X), dtype=float).flatten()
        
        if method == "newton":
            J_current = J_func(*X)
            if np.abs(np.linalg.det(J_current)) < 1e-10:
                raise ValueError(f"Jacobian at iteration {k+1} is singular (not invertible).")
            J_inv = np.linalg.inv(J_current)
        elif method == "modified":
            J_inv = J_inv_fixed
            J_current = J_func(*X)  # Vẫn tính J(X_n) để hiển thị det(J(X_n))
        else:
            raise ValueError("Method must be 'newton' or 'modified'")
        
        X_new = X - J_inv @ F_val
        error = np.abs(X_new - X)
        
        # Kiểm tra nan hoặc inf
        if np.any(np.isnan(X_new)) or np.any(np.isinf(X_new)):
            print("-" * total_width)
            raise ValueError(f"Iteration {k+1}: Solution diverged (nan or inf encountered).")
        
        # Kiểm tra phát tán sớm
        if error.max() > 1e3:
            print("-" * total_width)
            raise ValueError(f"Iteration {k+1}: Solution diverging (Max Error = {error.max():.6e} too large).")
        
        # Tính F(X_new) và det(J(X_n))
        F_new = np.array(F_func(*X_new), dtype=float).flatten()
        det_J = np.linalg.det(J_current)
        
        print(f"|{k+1:^{col_width_iter}}|", end="")
        for x in X_new:
            print(f"{x:^{col_width_val}.10f}|", end="")
        for f in F_new:
            print(f"{f:^{col_width_val}.6e}|", end="")
        print(f"{error.max():^{col_width_error}.6e}|", end="")
        print(f"{det_J:^{col_width_det}.6e}|")
        
        if np.all(error < tol):
            print("-" * total_width)
            print(f"\nConverged using {method} method!")
            return X_new
        
        X = X_new
    
    print("-" * total_width)
    print(f"\nMax iterations reached with {method} method.")
    return X

if __name__ == "__main__":
    n = int(input("Enter the number of variables: "))
    vars_list = symbols(f'x1:{n+1}')

    F_exprs = []
    print("Enter the equations (in terms of x1, x2, ..., xn):")
    for i in range(n):
        F_exprs.append(eval(input(f"Equation {i+1}: "), {**{str(v): v for v in vars_list}, **globals()}))

    X0 = list(map(float, input("Enter initial values (space-separated): ").split()))

    choice = input("Enter tolerance (default 1e-6) or max iterations: ")
    if choice.isdigit():
        max_iter = int(choice)
        tol = 1e-6
    else:
        max_iter = 50
        tol = float(choice) if choice else 1e-6

    method = input("Choose method ('newton' or 'modified'): ").strip().lower()
    if method not in ["newton", "modified"]:
        print("Invalid method! Defaulting to 'newton'.")
        method = "newton"

    solution = newton_solver(F_exprs, vars_list, X0, method, tol, max_iter)
    print("\nFinal Solution:", solution)
    
# input ví dụ
# Enter the number of variables: 3
# Enter the equations (in terms of x1, x2, ..., xn):
# Equation 1: 3*x1 - cos(x2*x3) - 0.5
# Equation 2: 4*x1**2 - 625*x2**2 + 2*x2 - 1
# Equation 3: exp(-x1*x2) + 20*x3 + (10*3.1415926536)/3 - 1
# Enter initial values (space-separated): 0 0 0
# Enter tolerance (default 1e-6) or max iterations: 6
# Choose method ('newton' or 'modified'): newton