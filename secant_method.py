# phương pháp dây cung với các mode đánh giá sai số
# LƯU Ý: CHỌN MÔI TRƯỜNG PYTHON ĐÚNG ĐỂ DÙNG ĐƯỢC THƯ VIỆN SYMPY: CTRL + SHIFT + P => SELECT INTERPRETER => PYTHON 3.12.9 ('BASE') LƯU Ở MINICONDA

# (sai số tuyệt đối) (sai số tương đối = sai số tuyệt đối / x)
# công thức sai số mục tiêu: delta = abs(f(x_next)) / min_derivative
# công thức sai số theo 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)

# mode 0: sai số tuyệt đối theo công thức sai số mục tiêu: delta = abs(f(x_next)) / min_derivative
# mode 1: sai số tuyệt đối theo công thức 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
# mode 2: sai số tương đối theo công thức sai số mục tiêu: delta = abs(f(x_next)) / (min_derivative * abs(x_next))
# mode 3: sai số tương đối theo công thức 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n) / abs(x_next)

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
import sympy as sp      # thư viện tính đạo hàm chính xác

# Định nghĩa hàm f(x), df(x), ddf(x) với đạo hàm chính xác bằng SymPy
def define_functions(f_expr):
    x = sp.Symbol('x')
    # Chuyển chuỗi f_expr thành biểu thức SymPy
    f_sym = sp.sympify(f_expr)
    
    # Tạo hàm Python từ biểu thức SymPy để tính giá trị số
    f = sp.lambdify(x, f_sym, 'numpy')
    
    # Tính đạo hàm cấp 1 chính xác bằng SymPy
    df_sym = sp.diff(f_sym, x)
    df = sp.lambdify(x, df_sym, 'numpy')
    
    # Tính đạo hàm cấp 2 chính xác bằng SymPy
    ddf_sym = sp.diff(df_sym, x)
    ddf = sp.lambdify(x, ddf_sym, 'numpy')
    
    return f_sym, f, df, ddf  # Trả về biểu thức SymPy và các hàm Python

# Xác định dấu của một số (âm return -1; dương return 1; 0 return 0)
def sign(x):
    return np.sign(x)

# Hàm tìm min và max của đạo hàm cấp 1 trên đoạn [a,b]
def min_max_derivative(f_sym, a, b):
    x = sp.Symbol('x')
    df_sym = sp.diff(f_sym, x)  # Tính đạo hàm cấp 1 chính xác bằng SymPy
    df = sp.lambdify(x, df_sym, 'numpy')
    
    # Tính giá trị đạo hàm tại a và b
    df_a = abs(df(a))
    df_b = abs(df(b))
    
    # Min và max của đạo hàm trên đoạn [a, b]
    min_derivative = min(df_a, df_b)
    max_derivative = max(df_a, df_b)
    
    print(f"Min derivative: {min_derivative}")
    print(f"Max derivative: {max_derivative}")
    return min_derivative, max_derivative

# Phương pháp dây cung
def secant_method(f, df, ddf, a, b, error=1e-6, max_iterations=100, mode=1, extra_iteration=True):
    if mode not in [0, 1, 2, 3]:
        raise ValueError("Chế độ đánh giá sai số không hợp lệ. Vui lòng chọn mode từ 0-3.")

    sfa = sign(f(a))
    sfb = sign(f(b))
    if sfa * sfb >= 0:
        raise ValueError("Không thể dùng phương pháp dây cung do f(a)*f(b) >= 0.")
    if a >= b:
        raise ValueError("Hãy nhập a < b.")
    if error <= 0:
        raise ValueError("Sai số phải là một giá trị dương.")

    # Kiểm tra tính ổn định dấu của đạo hàm
    sfx = sign(df(a))
    sfxx = sign(ddf(a))
    for x in np.linspace(a, b, 1001):
        if sign(df(x)) != sfx or sign(ddf(x)) != sfxx:
            raise ValueError("Phương pháp dây cung không thực hiện được do dấu của đạo hàm không ổn định.")

    # Chọn mốc d và xấp xỉ đầu
    if sfa * sign(ddf(a)) > 0:
        d = a
        x_n = b
    else:
        d = b
        x_n = a

    # Tính m và M
    min_derivative, max_derivative = min_max_derivative(f_sym, a, b)

    mode_names = [
        "Sai số tuyệt đối theo công thức sai số mục tiêu",
        "Sai số tuyệt đối theo công thức 2 xấp xỉ liên tiếp",
        "Sai số tương đối theo công thức sai số mục tiêu",
        "Sai số tương đối theo công thức 2 xấp xỉ liên tiếp"
    ]
    print(f"Chế độ đánh giá sai số: {mode} - {mode_names[mode]}")

    count = 0
    solution_found = False
    solution_iteration = -1
    solution_value = None

    while count < max_iterations:
        if abs(f(x_n) - f(d)) < 1e-12:
            raise ValueError("Phép chia cho số rất nhỏ hoặc 0 xảy ra.")

        x_next = x_n - f(x_n) * (x_n - d) / (f(x_n) - f(d))

        if mode in [2, 3] and abs(x_next) < 1e-12:
            raise ValueError("Không thể tính sai số tương đối do xấp xỉ quá gần 0.")

        if mode == 0:
            delta = abs(f(x_next)) / min_derivative
        elif mode == 1:
            delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
        elif mode == 2:
            delta = abs(f(x_next)) / (min_derivative * abs(x_next))
        else:
            delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n) / abs(x_next)
        
#______________________________________________________format cách hiển thị kết quả_____________________________________________________

        # Định nghĩa tên cột sai số theo mode
        delta_names = {
            0: "|f(x_n+1)|/m",
            1: "(M-m)/m * |x_n+1 - x_n|",
            2: "|f(x_n+1)|/(m*|x_n+1|)",
            3: "(M-m)/m * |x_n+1 - x_n|/|x_n+1|"
        }

        # In tiêu đề bảng (chỉ in ở lần lặp đầu tiên)
        if count == 0:
            if mode in [0, 2]:
                print(f"{'n':<6} | {'x_n+1':<15} | {'|f(x_n+1)|':<15} | {delta_names[mode]:<20}")
                print("-" * 60)
            else:  # mode 1, 3
                print(f"{'n':<6} | {'x_n+1':<15} | {'|x_n+1 - x_n|':<15} | {delta_names[mode]:<20}")
                print("-" * 60)

        # Tính giá trị cột thay đổi
        if mode in [0, 2]:
            col_value = abs(f(x_next))
            col_format = f"{col_value:<15.6e}"
        else:  # mode 1, 3
            col_value = abs(x_next - x_n)
            col_format = f"{col_value:<15.6e}"

        # In dòng dữ liệu
        print(f"{count:<6} | {x_next:<15.6f} | {col_format} | {delta:<20.6e}")
        
#______________________________________________________format cách hiển thị kết quả_____________________________________________________

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

        x_n = x_next
        count += 1

    if solution_found:
        return solution_value

    raise ValueError("Phương pháp không hội tụ sau số lần lặp tối đa.")

# hàm main bọc trong try-except: Nếu đầu vào không hợp lệ, in thông báo lỗi thay vì dừng chương trình đột ngột
if __name__ == "__main__":
    # Nhập hàm f với cú pháp của sympy
    f_expr = "x**5 - 7"  
    global f_sym  # Khai báo f_sym là biến toàn cục để sử dụng trong secant_method
    f_sym, f, df, ddf = define_functions(f_expr)
#MODE:
# 0: sai số tuyệt đối theo công thức sai số mục tiêu: delta = abs(f(x_next)) / min_derivative
# 1: sai số tuyệt đối theo công thức 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
# 2: sai số tương đối theo công thức sai số mục tiêu: delta = abs(f(x_next)) / (min_derivative * abs(x_next))
# 3: sai số tương đối theo công thức 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n) / abs(x_next)

    try:
        result = secant_method(f, df, ddf, a = 1, b = 2, error = 5e-8, mode = 2, extra_iteration = True)
        print(f"Nghiệm gần đúng: {result:.10f}")
        print(f"Giá trị hàm tại nghiệm: f({result:.10f}) = {f(result):.10e}")

        # Vẽ đồ thị
        import matplotlib.pyplot as plt
        x_values = np.linspace(-4, 3, 1000)
        y_values = [f(x) for x in x_values]

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
        
    # bắt lỗi: hàm nhận giá trị đầu vào không hợp lệ nhưng đúng về kiểu dữ liệu
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")