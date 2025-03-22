# phương pháp tiếp tuyến (newton) với các mode đánh giá sai số
# LƯU Ý: CHỌN MÔI TRƯỜNG PYTHON ĐÚNG ĐỂ DÙNG ĐƯỢC THƯ VIỆN SYMPY: CTRL + SHIFT + P => SELECT INTERPRETER => PYTHON 3.12.9 ('BASE') LƯU Ở MINICONDA

# (sai số tuyệt đối) (sai số tương đối = sai số tuyệt đối / x)
# m1 là trị tuyệt đối của min đạo hàm cấp 1 (min_derivative)
# M2 là trị tuyệt đối của max đạo hàm cấp 2 (max_derivative)

# Mode 0: Sai số tuyệt đối theo công thức mục tiêu: delta = |f(x_n)| / m
# Mode 1: Sai số tuyệt đối theo bình phương 2 xấp xỉ liên tiếp: delta = M / (2m) * |x_n - x_n-1|^2
# Mode 2: Sai số tương đối theo công thức mục tiêu: delta = |f(x_n)| / (m * |x_n|)
# Mode 3: Sai số tương đối theo bình phương 2 xấp xỉ liên tiếp: delta = M / (2m) * |x_n - x_n-1|^2 / |x_n|

# ------------- cách chọn hàm f(x) -----------------------------------------
# tính căn bậc m của n:  chọn f(x) = x**m - n            [do x^m = n]  
# tính logarit của m cơ số n:  chọn f(x) = m**x - n      [do log_m(n) = ln(n) / ln(m)]
# tính số e:  chọn f(x) = log(x) - 1                     [do  ln(e) - 1 = 0]               [hàm ln(x) đạo hàm = 1/x]
# tính số pi:  chọn f(x) = tan(x/4) - 1                  [do tan(x/4) = 1]                 [hàm tan(x) đạo hàm = 1/(cos(x))^2]

# ------------- cú pháp sympy ----------------------------------------------
# hàm đa thức: x**5 - 0.2*x + 15.0         [x**5 = x^5]
# hàm lượng giác:
# hàm sin: sin(x)
# hàm cos: cos(x)
# hàm tan: tan(x)                         [tan(x) không xác định ở bội số lẻ của π/2]
# hàm cot: cot(x) hoặc 1/tan(x)           [cot(x) không xác định khi tan(x)=0, tức là x=0,π,2π,...]
# hàm căn bậc 2: sqrt(x)
# hàm căn bậc n: x**(1/n)
# hàm mũ và logarit: 
# hàm e mũ: exp(x)
# hàm log với cơ số khác số e: log(x, base)             [base là cơ số]
# hàm ln: 
#def log(x):
#    if x <= 0:
#        raise ValueError("log(x) chỉ xác định với x > 0.")
#    return log(x)
# ------------------------------------------------------------------------------------------------------------

import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def define_functions(f_expr):
    x = sp.Symbol('x')
    f_sym = sp.sympify(f_expr)
    # Tạo hàm f với kiểm tra x > 0 để tránh lỗi log
    def safe_f(x_val):
        if x_val <= 0:
            raise ValueError("Giá trị x phải lớn hơn 0 để tính log(x)")
        return sp.lambdify(x, f_sym, 'numpy')(x_val)
    
    f = safe_f
    df_sym = sp.diff(f_sym, x)
    df = sp.lambdify(x, df_sym, 'numpy')
    ddf_sym = sp.diff(df_sym, x)
    ddf = sp.lambdify(x, ddf_sym, 'numpy')
    return f_sym, f, df, ddf

def min_max_derivative(f_sym, a, b):
    x = sp.Symbol('x')
    df_sym = sp.diff(f_sym, x)
    ddf_sym = sp.diff(df_sym, x)
    df = sp.lambdify(x, df_sym, 'numpy')
    ddf = sp.lambdify(x, ddf_sym, 'numpy')
    df_a, df_b = abs(df(a)), abs(df(b))
    ddf_a, ddf_b = abs(ddf(a)), abs(ddf(b))
    min_derivative = min(df_a, df_b)
    max_derivative = max(ddf_a, ddf_b)
    
    print(f"m1 = {min_derivative}, M2 = {max_derivative}")
    return min_derivative, max_derivative

def newton_method(f, df, ddf, a, b, error=1e-6, max_iterations=100, mode=0, extra_iteration=True):
    """
    Phương pháp Newton-Raphson với các mode đánh giá sai số:
    - Mode 0: Sai số tuyệt đối theo công thức mục tiêu: delta = |f(x_n)| / m
    - Mode 1: Sai số tuyệt đối theo 2 xấp xỉ liên tiếp: delta = M / (2m) * |x_n - x_n-1|^2
    - Mode 2: Sai số tương đối theo công thức mục tiêu: delta = |f(x_n)| / (m * |x_n|)
    - Mode 3: Sai số tương đối theo 2 xấp xỉ liên tiếp: delta = M / (2m) * |x_n - x_n-1|^2 / |x_n|
    """
    if mode not in [0, 1, 2, 3]:
        raise ValueError("Chế độ đánh giá sai số không hợp lệ. Chọn mode từ 0-3.")
    
    x_n = a if f(a) * df(a) > 0 else b
    min_derivative, max_derivative = min_max_derivative(f_sym, a, b)
    
    # Thông báo công thức và loại sai số
    delta_formulas = {
        0: "|f(x_n)| / m",
        1: "M / (2m) * |x_n - x_n-1|^2",
        2: "|f(x_n)| / (m * |x_n|)",
        3: "M / (2m) * |x_n - x_n-1|^2 / |x_n|"
    }
    error_types = {0: "tuyệt đối", 1: "tuyệt đối", 2: "tương đối", 3: "tương đối"}
    print(f"Đang sử dụng công thức sai số: {delta_formulas[mode]}")
    print(f"Loại sai số: {error_types[mode]}")
    
    # Tiêu đề bảng
    if mode in [0, 2]:
        col_name = "|f(x_n)|"
        print(f"{'n':<5} | {'x_n':<15} | {col_name:<15} | {delta_formulas[mode]:<15}")
    else:
        col_name = "|x_n - x_n-1|"
        print(f"{'n':<5} | {'x_n':<15} | {col_name:<15} | {delta_formulas[mode]:<15}")
    print("-" * 60)
    
    count = 0
    prev_x = None
    solution_found = False
    solution_iteration = -1
    solution_value = None
    
    while count < max_iterations:
        if abs(df(x_n)) < 1e-12:
            raise ValueError("Đạo hàm gần 0, phương pháp không hội tụ.")
        
        x_next = x_n - f(x_n) / df(x_n)
        
        # Tính sai số và giá trị cột thứ 3
        if mode in [0, 2]:
            delta = abs(f(x_n)) / (min_derivative * (abs(x_n) if mode == 2 else 1))
            col_value = abs(f(x_n))
        else:
            if prev_x is None:
                col_value = "N/A"
                delta = float('inf')  # Sai số không xác định ở lần đầu
            else:
                diff = abs(x_n - prev_x)
                col_value = diff
                delta = (max_derivative / (2 * min_derivative)) * diff ** 2 / (abs(x_n) if mode == 3 else 1)
        
        # In dữ liệu
        if prev_x is None and mode in [1, 3]:
            print(f"{count:<5} | {x_n:<15.10f} | {col_value:<15} | {'N/A':<25}")
        else:
            print(f"{count:<5} | {x_n:<15.10f} | {col_value:<15.6e} | {delta:<25.6e}")
        
        if delta < error and not solution_found:
            solution_found = True
            solution_iteration = count
            solution_value = x_next
            print(f"-> Đã tìm thấy nghiệm tại vòng lặp {count} với sai số {delta:.6e} < {error:.6e}")
            if not extra_iteration:
                return solution_value
        
        if solution_found and count > solution_iteration:
            print(f"-> Đây là vòng lặp thứ k+1 = {count} sau khi đã tìm thấy nghiệm tại vòng lặp thứ k = {solution_iteration}")
            return solution_value
        
        prev_x = x_n
        x_n = x_next
        count += 1
    
    if solution_found:
        return solution_value
    
    raise ValueError("Phương pháp không hội tụ sau số lần lặp tối đa.")

if __name__ == "__main__":
    f_expr = "x**3 - log(x) - 10"
    f_sym, f, df, ddf = define_functions(f_expr)
    
    try:
        result = newton_method(f, df, ddf, a=1e-5, b=1e-4, error=1e-8, mode=2, extra_iteration=True)
        print(f"Nghiệm gần đúng: {result:.10f}")
        
        """
        Phương pháp Newton-Raphson với các mode đánh giá sai số:
        - Mode 0: Sai số tuyệt đối theo công thức mục tiêu: delta = |f(x_n)| / m
        - Mode 1: Sai số tuyệt đối theo bình phương 2 xấp xỉ liên tiếp: delta = M / (2m) * |x_n - x_n-1|^2
        - Mode 2: Sai số tương đối theo công thức mục tiêu: delta = |f(x_n)| / (m * |x_n|)
        - Mode 3: Sai số tương đối theo bình phương 2 xấp xỉ liên tiếp: delta = M / (2m) * |x_n - x_n-1|^2 / |x_n|
        """ 
        
        # Xử lý lỗi khi gọi f(result)
        try:
            f_result = f(result)
            print(f"Giá trị hàm tại nghiệm: f({result:.10f}) = {f_result:.10e}")
        except (ValueError, RuntimeWarning) as e:
            print(f"Không thể tính giá trị hàm tại nghiệm do lỗi: {e}") 
          
        # Vẽ đồ thị
        x_values = np.linspace(-1, 3, 1000)  # Chỉ vẽ trong khoảng [2, 3]
        y_values = []
        for x in x_values:
            try:
                y_values.append(f(x))
            except ValueError:
                y_values.append(np.nan)  # Bỏ qua các giá trị không hợp lệ
        
        plt.plot(x_values, y_values, label=f"f(x) = {f_expr}", color="blue")
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.axvline(result, color='red', linestyle="--", label=f"Nghiệm x ≈ {result:.6f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Đồ thị hàm số f(x)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
        
    except ValueError as e:
        print(f"Lỗi: {e}")