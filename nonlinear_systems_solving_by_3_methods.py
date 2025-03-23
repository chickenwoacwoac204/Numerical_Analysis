# LƯU Ý: CHỌN MÔI TRƯỜNG PYTHON ĐÚNG ĐỂ DÙNG ĐƯỢC THƯ VIỆN SYMPY: CTRL + SHIFT + P => SELECT INTERPRETER => PYTHON 3.12.9 ('BASE') LƯU Ở MINICONDA

"""
    Giải hệ phương trình phi tuyến bằng các phương pháp Newton, Newton Modified, hoặc Lặp đơn.
    
    Args:
        F_expressions (list): Danh sách các biểu thức phi tuyến (SymPy) của F(X) = 0.
        variables (list): Danh sách biến tượng trưng (x1, x2, ...).
        X0 (list): Vectơ giá trị khởi tạo.
        mode (int): Chế độ tính toán (0-5):
            - 0: Newton với sai số tuyệt đối
            - 1: Newton với sai số tương đối
            - 2: Newton Modified với sai số tuyệt đối
            - 3: Newton Modified với sai số tương đối
            - 4: Lặp đơn với sai số tuyệt đối
            - 5: Lặp đơn với sai số tương đối
        tol (float): Sai số cho phép, mặc định 1e-6.
        max_iter (int): Số lần lặp tối đa, mặc định 50.
        G_expressions (list, optional): Danh sách biểu thức G(X) cho lặp đơn (X = G(X)). Nếu None, cố gắng tự động biến đổi.
    
    Returns:
        numpy.ndarray: Nghiệm cuối cùng nếu hội tụ, hoặc giá trị cuối cùng nếu không hội tụ.
    
    Raises:
        ValueError: Nếu biến vượt miền xác định, mode không hợp lệ, Jacobian suy biến, hoặc nghiệm phát tán.
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
from sympy import symbols, Matrix, lambdify, sqrt
from sympy import cos, sin, exp

def newton_solver(F_expressions, variables, X0, mode=0, tol=1e-6, max_iter=50, G_expressions=None):
    n = len(variables)
    F = Matrix(F_expressions)
    J = F.jacobian(variables)
    
    F_func = lambdify(variables, F, 'numpy')
    J_func = lambdify(variables, J, 'numpy')
    
    # Xác định phương pháp và loại sai số từ mode
    if mode not in range(6):
        raise ValueError("Mode must be an integer from 0 to 5.")
    
    method_dict = {0: "newton", 1: "newton", 2: "modified", 3: "modified", 4: "fixed_point", 5: "fixed_point"}
    error_type_dict = {0: "absolute", 1: "relative", 2: "absolute", 3: "relative", 4: "absolute", 5: "relative"}
    
    method = method_dict[mode]
    error_type = error_type_dict[mode]
    
    # Chuẩn bị cho lặp đơn
    if method == "fixed_point":
        if G_expressions is None:
            G_expressions = []
            for i, expr in enumerate(F_expressions):
                try:
                    g_expr = (expr + variables[i]).simplify()
                    G_expressions.append(g_expr)
                except:
                    raise ValueError(f"Cannot automatically transform equation {i+1} into X = G(X) form. Please provide G_expressions.")
        G = Matrix(G_expressions)
        G_func = lambdify(variables, G, 'numpy')
    
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
    col_width_det = 15
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
        
        if method in ["newton", "modified"]:
            if method == "newton":
                J_current = J_func(*X)
                if np.abs(np.linalg.det(J_current)) < 1e-10:
                    raise ValueError(f"Jacobian at iteration {k+1} is singular (not invertible).")
                J_inv = np.linalg.inv(J_current)
            else:
                J_inv = J_inv_fixed
                J_current = J_func(*X)
            X_new = X - J_inv @ F_val
        elif method == "fixed_point":
            X_new = np.array(G_func(*X), dtype=float).flatten()
            J_current = J_func(*X)
        else:
            raise ValueError("Method must be 'newton', 'modified', or 'fixed_point'")
        
        # Tính sai số
        abs_error = np.abs(X_new - X)
        if error_type == "absolute":
            error = abs_error
        else:  # relative error
            # Tránh chia cho 0: nếu |X_new| quá nhỏ, dùng sai số tuyệt đối
            X_new_norm = np.abs(X_new)
            X_new_norm = np.where(X_new_norm < 1e-10, 1.0, X_new_norm)  # Thay thế giá trị nhỏ bằng 1
            error = abs_error / X_new_norm
        
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
            method_name = "Newton" if method == "newton" else "Newton Modified" if method == "modified" else "Fixed-Point"
            print(f"\nConverged using {method_name} method with {error_type} error!")
            return X_new
        
        X = X_new
    
    print("-" * total_width)
    method_name = "Newton" if method == "newton" else "Newton Modified" if method == "modified" else "Fixed-Point"
    print(f"\nMax iterations reached with {method_name} method and {error_type} error.")
    return X

if __name__ == "__main__":
    """Hàm chính để nhập dữ liệu từ người dùng và chạy phương pháp."""
    n = int(input("Nhập số lượng biến: "))
    vars_list = symbols(f'x1:{n+1}')

    F_exprs = []
    print("Nhập các phương trình F(X) = 0 (theo dạng x1, x2, ..., xn):")
    for i in range(n):
        F_exprs.append(eval(input(f"Phương trình {i+1}: "), {**{str(v): v for v in vars_list}, **globals()}))

    X0 = list(map(float, input("Nhập giá trị ban đầu (cách nhau bởi dấu cách): ").split()))

    choice = input("Nhập sai số cho phép (mặc định 1e-6) hoặc số lần lặp tối đa: ")
    if choice.isdigit():
        max_iter = int(choice)
        tol = 1e-6
    else:
        max_iter = 50
        tol = float(choice) if choice else 1e-6

    print("Chọn chế độ tính toán (0-5):")
    print("0: Newton với sai số tuyệt đối")
    print("1: Newton với sai số tương đối")
    print("2: Newton Modified với sai số tuyệt đối")
    print("3: Newton Modified với sai số tương đối")
    print("4: Lặp đơn với sai số tuyệt đối")
    print("5: Lặp đơn với sai số tương đối")
    mode = int(input("Nhập chế độ: "))

    G_exprs = None
    if mode in [4, 5]:  # Fixed-Point modes
        print("Nhập các phương trình X = G(X) cho phương pháp lặp đơn (hoặc nhấn Enter để tự động biến đổi):")
        G_exprs = []
        for i in range(n):
            g_input = input(f"G{i+1}(x1, x2, ..., xn) = ").strip()
            if g_input:
                G_exprs.append(eval(g_input, {**{str(v): v for v in vars_list}, **globals()}))
            else:
                G_exprs = None
                break

    solution = newton_solver(F_exprs, vars_list, X0, mode, tol, max_iter, G_exprs)
    print("\nNghiệm cuối cùng:", solution)

# ------------- Cú pháp SymPy và miền xác định ----------------------------------------------
# Hàm đa thức: 
# - Ví dụ: x

# input ví dụ
# Enter the number of variables: 3
# Enter the equations (in terms of x1, x2, ..., xn):
# Equation 1: 3*x1 - cos(x2*x3) - 0.5
# Equation 2: 4*x1**2 - 625*x2**2 + 2*x2 - 1
# Equation 3: exp(-x1*x2) + 20*x3 + (10*3.1415926536)/3 - 1
# Enter initial values (space-separated): 0 0 0
# Enter tolerance (default 1e-6) or max iterations: 6
# Choose method ('newton' or 'modified'): newton