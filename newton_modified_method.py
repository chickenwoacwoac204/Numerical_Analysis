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
        ValueError: Nếu biến vượt miền xác định hoặc phương pháp không hợp lệ.
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

import numpy as np  # Thư viện hỗ trợ tính toán số học và ma trận
from sympy import symbols, Matrix, lambdify  # Thư viện SymPy cho biểu thức tượng trưng và Jacobian
from sympy import cos, sin, exp, asin, acos, atan  # Các hàm SymPy: lượng giác, mũ, lượng giác ngược

def newton_solver(F_expressions, variables, X0, method="newton", tol=1e-6, max_iter=50):
    n = len(variables)  # Số biến trong hệ (kích thước của hệ phương trình)
    F = Matrix(F_expressions)  # Chuyển danh sách biểu thức thành ma trận SymPy (vector F(x))
    J = F.jacobian(variables)  # Tính ma trận Jacobian (ma trận đạo hàm riêng bậc một của F)
    
    # Chuyển đổi biểu thức SymPy thành hàm numpy để tính toán số
    F_func = lambdify(variables, F, 'numpy')  # Hàm F(x) dạng numpy
    J_func = lambdify(variables, J, 'numpy')  # Hàm Jacobian J(x) dạng numpy
    
    X = np.array(X0, dtype=float)  # Chuyển X0 thành mảng numpy kiểu float để tính toán
    
    # Nếu dùng Newton Modified, tính nghịch đảo Jacobian tại X0 một lần và giữ cố định
    if method == "modified":
        J_inv_fixed = np.linalg.inv(J_func(*X0))  # Nghịch đảo Jacobian tại X0
    
    # Định dạng tiêu đề bảng
    headers = ["Iteration"] + [f"x{i+1}" for i in range(n)] + ["Max Error"]  # Tiêu đề cột
    col_width_iter = 10  # Độ rộng cột Iteration
    col_width_val = 15   # Độ rộng cột x_i (đủ cho 10 chữ số thập phân)
    col_width_error = 15 # Độ rộng cột Max Error (đủ cho dạng khoa học .6e)
    total_width = col_width_iter + n * col_width_val + col_width_error + (n + 2)  # Tổng chiều rộng bảng
    print("-" * total_width)  # In đường kẻ ngang đầu bảng
    print(f"|{headers[0]:^{col_width_iter}}|", end="")  # In tiêu đề Iteration, căn giữa
    for i in range(1, n+1):
        print(f"{headers[i]:^{col_width_val}}|", end="")  # In tiêu đề x_i, căn giữa
    print(f"{headers[n+1]:^{col_width_error}}|")  # In tiêu đề Max Error, căn giữa
    print("-" * total_width)  # In đường kẻ ngang sau tiêu đề
    
    # Vòng lặp chính của phương pháp Newton
    for k in range(max_iter):
        # Kiểm tra miền xác định cho hàm lượng giác ngược (asin, acos yêu cầu x trong [-1, 1])
        if any('asin' in str(expr) or 'acos' in str(expr) for expr in F_expressions):
            if any(abs(x) > 1 for x in X):
                raise ValueError("Variable out of domain [-1, 1] for arcsin or arccos")
        
        # Kiểm tra miền xác định cho số mũ nhỏ hơn 1 (yêu cầu x >= 0 để tránh nan trong số thực)
        if any('**' in str(expr) for expr in F_expressions):
            for expr in F_expressions:
                if '**' in str(expr):
                    exponent = str(expr).split('**')[1].split()[0]  # Lấy số mũ từ biểu thức
                    try:
                        if float(exponent) < 1 and any(x < 0 for x in X):
                            raise ValueError("Variable became negative with fractional exponent < 1")
                    except ValueError:  # Nếu không phân tích được số mũ (ví dụ biểu thức phức tạp)
                        pass  # Bỏ qua kiểm tra này
        
        # Tính giá trị hàm F tại X hiện tại
        F_val = np.array(F_func(*X), dtype=float).flatten()  # Giá trị F(X) dạng mảng 1D
        
        # Chọn phương pháp dựa trên tham số method
        if method == "newton":
            J_current = J_func(*X)  # Tính Jacobian tại X hiện tại
            J_inv = np.linalg.inv(J_current)  # Tính nghịch đảo Jacobian tại mỗi bước
        elif method == "modified":
            J_inv = J_inv_fixed  # Dùng Jacobian cố định từ X0
        else:
            raise ValueError("Method must be 'newton' or 'modified'")  # Lỗi nếu method không hợp lệ
        
        # Công thức lặp Newton: X_new = X - J^{-1} * F(X)
        X_new = X - J_inv @ F_val  # Cập nhật X mới (@ là phép nhân ma trận)
        error = np.abs(X_new - X)  # Tính sai số tuyệt đối giữa X mới và X cũ
        
        # In dòng kết quả
        print(f"|{k+1:^{col_width_iter}}|", end="")  # In số lần lặp, căn giữa
        for x in X_new:
            print(f"{x:^{col_width_val}.10f}|", end="")  # In nghiệm x_i với 10 chữ số thập phân
        print(f"{error.max():^{col_width_error}.6e}|")  # In Max Error dạng khoa học .6e
        
        # Kiểm tra điều kiện hội tụ: tất cả sai số nhỏ hơn tol
        if np.all(error < tol):
            print("-" * total_width)  # In đường kẻ ngang cuối bảng
            print(f"\nConverged using {method} method!")  # Thông báo hội tụ
            return X_new  # Trả về nghiệm cuối cùng
        
        X = X_new  # Cập nhật X cho bước lặp tiếp theo
    
    # Nếu vượt quá max_iter
    print("-" * total_width)  # In đường kẻ ngang cuối bảng
    print(f"\nMax iterations reached with {method} method.")  # Thông báo không hội tụ
    return X  # Trả về giá trị cuối cùng

if __name__ == "__main__":
    # Nhập số biến từ người dùng
    n = int(input("Enter the number of variables: "))  # Số lượng biến (kích thước hệ)
    vars_list = symbols(f'x1:{n+1}')  # Tạo danh sách biến tượng trưng x1, x2, ..., xn

    # Nhập hệ phương trình phi tuyến
    F_exprs = []  # Danh sách lưu các biểu thức phương trình
    print("Enter the equations (in terms of x1, x2, ..., xn):")
    for i in range(n):
        # Dùng eval để chuyển chuỗi nhập thành biểu thức SymPy
        # Kết hợp globals() với biến tượng trưng để hỗ trợ các hàm như sin, cos, exp
        F_exprs.append(eval(input(f"Equation {i+1}: "), {**{str(v): v for v in vars_list}, **globals()}))

    # Nhập giá trị khởi tạo X0
    X0 = list(map(float, input("Enter initial values (space-separated): ").split()))  # Chuyển chuỗi thành list float

    # Nhập sai số hoặc số lần lặp tối đa
    choice = input("Enter tolerance (default 1e-6) or max iterations: ")
    if choice.isdigit():
        max_iter = int(choice)  # Nếu nhập số nguyên, coi đó là max_iter
        tol = 1e-6  # Sai số mặc định
    else:
        max_iter = 50  # Số lần lặp tối đa mặc định
        tol = float(choice) if choice else 1e-6  # Nếu nhập số thực, dùng làm tol; nếu rỗng, dùng 1e-6

    # Chọn phương pháp
    method = input("Choose method ('newton' or 'modified'): ").strip().lower()  # Nhập phương pháp, chuẩn hóa chữ thường
    if method not in ["newton", "modified"]:
        print("Invalid method! Defaulting to 'newton'.")  # Thông báo nếu nhập sai
        method = "newton"  # Mặc định là Newton cổ điển

    # Gọi hàm giải hệ phương trình
    solution = newton_solver(F_exprs, vars_list, X0, method, tol, max_iter)
    print("\nFinal Solution:", solution)  # In nghiệm cuối cùng


#------------------ Input ví dụ --------------------    
# Enter the number of variables: 2
# Enter the equations (in terms of x1, x2, ...):
# Equation 1: sin(x1) + x2 - 1
# Equation 2: x1 - x2**2
# Enter initial values (space-separated): 1 1
# Enter tolerance (default 1e-6) or max iterations: 
# Choose method ('newton' or 'modified'): newton