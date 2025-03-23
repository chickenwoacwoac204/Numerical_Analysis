# Giải hệ phương trình phi tuyến bằng phương pháp Newton Modified (Jacobian cố định).
#    Args:
#        F_expressions: Danh sách các biểu thức phi tuyến (sympy).
#        variables: Danh sách biến tượng trưng (x1, x2, ...).
#        X0: Vectơ giá trị khởi tạo.
#        tol: Sai số cho phép.
#        max_iter: Số lần lặp tối đa.
#    Returns:
#        Nghiệm cuối cùng hoặc thông báo không hội tụ.

# LƯU Ý: CHỌN MÔI TRƯỜNG PYTHON ĐÚNG ĐỂ DÙNG ĐƯỢC THƯ VIỆN SYMPY: CTRL + SHIFT + P => SELECT INTERPRETER => PYTHON 3.12.9 ('BASE') LƯU Ở MINICONDA

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
from sympy import cos, sin, exp, asin, acos, atan  # Thêm asin, acos, atan để hỗ trợ lượng giác ngược

def modified_simple_iteration(F_expressions, variables, X0, tol=1e-6, max_iter=50):
    """
    Giải hệ phương trình phi tuyến bằng phương pháp Newton Modified (Jacobian cố định).
    Args:
        F_expressions: Danh sách các biểu thức phi tuyến (sympy).
        variables: Danh sách biến tượng trưng (x1, x2, ...).
        X0: Vectơ giá trị khởi tạo.
        tol: Sai số cho phép.
        max_iter: Số lần lặp tối đa.
    Returns:
        Nghiệm cuối cùng hoặc thông báo không hội tụ.
    """
    n = len(variables)
    F = Matrix(F_expressions)  # Chuyển danh sách biểu thức thành ma trận sympy
    J = F.jacobian(variables)  # Tính ma trận Jacobian (đạo hàm riêng bậc một)
    
    # Chuyển đổi biểu thức sympy thành hàm numpy để tính toán số
    F_func = lambdify(variables, F, 'numpy')
    J_func = lambdify(variables, J, 'numpy')
    
    # Tính nghịch đảo Jacobian tại X0 và giữ cố định trong toàn bộ quá trình
    J_inv = np.linalg.inv(J_func(*X0))
    
    X = np.array(X0, dtype=float)  # Chuyển X0 thành mảng numpy kiểu float
    print("Iteration |", "X values", "| Error")
    print("-"*50)
    
    for k in range(max_iter):
        # Kiểm tra miền cho lượng giác ngược (asin, acos yêu cầu [-1, 1])
        if any('asin' in str(expr) or 'acos' in str(expr) for expr in F_expressions):
            if any(abs(x) > 1 for x in X):
                raise ValueError("Variable out of domain [-1, 1] for arcsin or arccos")
        
        # Kiểm tra miền cho số mũ nhỏ hơn 1 (yêu cầu x >= 0)
        if any('**' in str(expr) for expr in F_expressions):  # Kiểm tra có lũy thừa không
            for expr in F_expressions:
                if '**' in str(expr):
                    exponent = str(expr).split('**')[1].split()[0]  # Lấy số mũ
                    try:
                        if float(exponent) < 1 and any(x < 0 for x in X):
                            raise ValueError("Variable became negative with fractional exponent < 1")
                    except ValueError:  # Nếu exponent không phải số đơn giản
                        pass  # Bỏ qua nếu không phân tích được

        # Tính giá trị hàm F tại X hiện tại
        F_val = np.array(F_func(*X), dtype=float).flatten()
        
        # Công thức lặp Newton Modified: X_new = X - J_inv * F(X)
        X_new = X - J_inv @ F_val  
        error = np.abs(X_new - X)  # Tính sai số giữa X mới và cũ
        
        # In thông tin mỗi bước lặp
        print(f"{k+1:^9} | {X_new} | {error.max()}")
        
        # Kiểm tra điều kiện hội tụ: sai số nhỏ hơn tol
        if np.all(error < tol):
            print("\nConverged!")
            return X_new
        
        X = X_new  # Cập nhật X cho bước lặp tiếp theo
    
    print("\nMax iterations reached.")  # Thông báo nếu không hội tụ
    return X  # Trả về giá trị cuối cùng

if __name__ == "__main__":
    # Nhập số biến n từ người dùng
    n = int(input("Enter the number of variables: "))
    vars_list = symbols(f'x1:{n+1}')  # Tạo danh sách biến x1, x2, ..., xn (sympy symbols)

    # Nhập hệ phương trình phi tuyến
    F_exprs = []
    print("Enter the equations (in terms of x1, x2, ..., xn):")
    for i in range(n):
        # Dùng eval để chuyển chuỗi nhập thành biểu thức sympy
        # Gộp globals() với các biến tượng trưng để hỗ trợ sin, cos, exp, asin, acos, atan
        F_exprs.append(eval(input(f"Equation {i+1}: "), {**{str(v): v for v in vars_list}, **globals()}))

    # Nhập giá trị khởi tạo X0
    X0 = list(map(float, input("Enter initial values (space-separated): ").split()))

    # Nhập sai số hoặc số lần lặp tối đa
    choice = input("Enter tolerance (default 1e-6) or max iterations: ")
    if choice.isdigit():
        max_iter = int(choice)  # Nếu nhập số nguyên, coi đó là max_iter
        tol = 1e-6
    else:
        max_iter = 50  # Mặc định max_iter nếu không nhập số nguyên
        tol = float(choice) if choice else 1e-6  # Nếu nhập số thực, dùng làm tol

    # Gọi hàm giải hệ phương trình
    solution = modified_simple_iteration(F_exprs, vars_list, X0, tol, max_iter)
    print("\nFinal Solution:", solution)
