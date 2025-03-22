# phương pháp lặp đơn với các mode đánh giá sai số
# LƯU Ý: CHỌN MÔI TRƯỜNG PYTHON ĐÚNG ĐỂ DÙNG ĐƯỢC THƯ VIỆN SYMPY: CTRL + SHIFT + P => SELECT INTERPRETER => PYTHON 3.12.9 ('BASE') LƯU Ở MINICONDA

# (sai số tuyệt đối) (sai số tương đối = sai số tuyệt đối / x)
# công thức tiên nghiệm: delta = [q^n / (1 - q)] * |x_1 - x_0|
# công thức hậu nghiệm: delta = [q / (1 - q)] * |x_n - x_n-1|

# mode 0: tuyệt đối, tiên nghiệm: (q^n * |x_n - x_n-1|) / (1 - q)
# mode 1: tuyệt đối, hậu nghiệm: (q * |x_n - x_n-1|) / (1 - q)
# mode 2: tương đối, tiên nghiệm: [(q^n * |x_n - x_n-1|) / (1 - q)] / |x_n|
# mode 3: tương đối, hậu nghiệm: [(q * |x_n - x_n-1|) / (1 - q)] / |x_n|

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
# hàm tan: tan(x)                                        [tan(x) không xác định ở bội số lẻ của π/2]
# hàm cot: cot(x) hoặc 1/tan(x)                          [cot(x) không xác định khi tan(x)=0, tức là x=0,π,2π,...]
# hàm lượng giác ngược:
# hàm arcsin: asin(x) hoặc arcsin(x)                     [x ∈ [-1, 1], kết quả ∈ [-π/2, π/2]]
# hàm arccos: acos(x) hoặc arccos(x)                     [x ∈ [-1, 1], kết quả ∈ [0, π]]
# hàm arctan: atan(x) hoặc arctan(x)                     [kết quả ∈ [-π/2, π/2]]
# hàm arccot: acot(x) hoặc arccot(x)                     [kết quả ∈ [0, π]]
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

def define_functions(expr):
    x = sp.Symbol('x')
    sym = sp.sympify(expr)
    def safe_func(x_val):
        if x_val <= 0 and 'log' in expr:
            raise ValueError("Giá trị x phải lớn hơn 0 để tính log(x)")
        return sp.lambdify(x, sym, 'numpy')(x_val)
    
    return sym, safe_func

def fixed_point_iteration(f_sym, f, phi_expr, a, b, error=1e-8, q=None, max_iterations=1000, mode=0, extra_iteration=True):
    phi_symbol = "\u03A6"
    
    x = sp.Symbol('x')
    phi_sym = sp.sympify(phi_expr)
    
    # Kiểm tra miền xác định của các hàm đặc biệt
    # 1. Hàm logarit (log_e(x) hoặc log_c(x))
    has_log = phi_sym.has(sp.log)
    if has_log:
        log_args = [arg for arg in phi_sym.atoms(sp.log)]
        for log in log_args:
            arg = log.args[0]  # Đối số của log
            arg_values = [float(arg.subs(x, x_val).evalf()) for x_val in np.linspace(a, b, 100)]
            min_arg = min(arg_values)
            if min_arg <= 0:
                raise ValueError(f"Đối số của log ({arg}) phải lớn hơn 0 trên [{a}, {b}], nhưng có giá trị nhỏ nhất là {min_arg}!")

    # 2. Hàm arcsin và arccos
    has_arcsin = phi_sym.has(sp.asin)
    has_arccos = phi_sym.has(sp.acos)
    if has_arcsin or has_arccos:
        arcsin_args = [arg for arg in phi_sym.atoms(sp.asin)]
        arccos_args = [arg for arg in phi_sym.atoms(sp.acos)]
        for arg in arcsin_args + arccos_args:
            arg_values = [float(arg.subs(x, x_val).evalf()) for x_val in np.linspace(a, b, 100)]
            min_arg = min(arg_values)
            max_arg = max(arg_values)
            if min_arg < -1.0001 or max_arg > 1.0001:
                raise ValueError(f"Đối số của arcsin/arccos ({arg}) phải nằm trong [-1, 1] trên [{a}, {b}], nhưng có giá trị trong [{min_arg}, {max_arg}]!")

    # 3. Hàm tan và cot
    has_tan = phi_sym.has(sp.tan)
    has_cot = phi_sym.has(sp.cot)
    if has_tan or has_cot:
        tan_args = [arg for arg in phi_sym.atoms(sp.tan)]
        cot_args = [arg for arg in phi_sym.atoms(sp.cot)]
        for arg in tan_args + cot_args:
            arg_values = [float(arg.subs(x, x_val).evalf()) for x_val in np.linspace(a, b, 100)]
            for val in arg_values:
                # Kiểm tra xem val có gần điểm không xác định của tan/cot hay không
                mod_val = (val + sp.pi/2) % sp.pi - sp.pi/2
                if abs(mod_val) < 1e-5:  # Gần điểm không xác định
                    raise ValueError(f"Đối số của tan/cot ({arg}) không được gần các điểm không xác định (pi/2 + k*pi) trên [{a}, {b}]!")

    # 4. Hàm mũ không nguyên (g(x)^h(x))
    use_manual_exp = False
    base_func = None
    exponent = None
    if phi_sym.func == sp.Pow:
        base_func = phi_sym.args[0]  # g(x)
        exponent = phi_sym.args[1]   # h(x)
        if isinstance(exponent, sp.Rational) and exponent.q != 1:
            base_values = [float(base_func.subs(x, x_val).evalf()) for x_val in np.linspace(a, b, 100)]
            min_base = min(base_values)
            if min_base < 0:
                use_manual_exp = True
        elif not isinstance(exponent, (int, sp.Integer)):
            exp_values = [float(exponent.subs(x, x_val).evalf()) for x_val in np.linspace(a, b, 100)]
            for exp_val in exp_values:
                if abs(exp_val - round(exp_val)) > 1e-5:  # Không nguyên
                    base_values = [float(base_func.subs(x, x_val).evalf()) for x_val in np.linspace(a, b, 100)]
                    min_base = min(base_values)
                    if min_base < 0:
                        use_manual_exp = True
                        break

    # Tạo phi(x) dựa trên kết quả kiểm tra
    if use_manual_exp and base_func is not None:
        print("Cơ số của lũy thừa có thể âm và số mũ không nguyên, sử dụng hàm phi thủ công.")
        def phi(x_val):
            base = float(base_func.subs(x, x_val).evalf())
            exp_val = float(exponent.subs(x, x_val).evalf()) if not isinstance(exponent, (sp.Rational, int, sp.Integer)) else float(exponent)
            if base >= 0:
                return base**exp_val
            else:
                return -abs(base)**exp_val
    else:
        print("Cơ số của lũy thừa không âm hoặc không có lũy thừa, sử dụng sympy với lambdify.")
        phi_lambdified = sp.lambdify(x, phi_sym, modules=['numpy', 'sympy'])
        def phi(x_val):
            result = phi_lambdified(x_val)
            if isinstance(result, complex):
                if abs(result.imag) < 1e-10:  # Phần ảo rất nhỏ, bỏ qua
                    return float(result.real)
                else:
                    raise ValueError(f"Giá trị của phi(x) tại x = {x_val} không phải là số thực: {result}")
            if isinstance(result, (sp.Expr, sp.Number)):
                result = float(result.evalf())
            if not isinstance(result, (float, int)):
                raise ValueError(f"Giá trị của phi(x) tại x = {x_val} không thể chuyển thành số thực: {result}")
            return result
    
    # Bước 0: Kiểm tra điều kiện hội tụ
    # 1. Kiểm tra a < b
    if a >= b:
        raise ValueError("Điều kiện hội tụ không thỏa mãn: a phải nhỏ hơn b!")
    
    # 2. Kiểm tra 0 < q < 1 (do người dùng nhập)
    if q is None:
        raise ValueError("Bạn phải nhập giá trị q (hệ số co)!")
    if not (0 < q < 1):
        raise ValueError("Hệ số co q phải nằm trong khoảng (0, 1)!")
    
    # 3. Kiểm tra error > 0
    if error <= 0:
        raise ValueError("Điều kiện hội tụ không thỏa mãn: Sai số error phải lớn hơn 0!")
    
    # 4. Kiểm tra phi(x) ∈ [a, b]
    x_values = np.linspace(a, b, 100)
    phi_values = [phi(x_val) for x_val in x_values]
    min_phi, max_phi = min(phi_values), max(phi_values)
    print(f"Giá trị của {phi_symbol}(x) trên [{a}, {b}]: [{min_phi}, {max_phi}]")
    if min_phi < a or max_phi > b:
        raise ValueError(f"Điều kiện hội tụ không thỏa mãn: {phi_symbol}(x) không ánh xạ [{a}, {b}] vào chính nó!")
    
    # 5. Kiểm tra mode hợp lệ
    if mode not in [0, 1, 2, 3]:
        raise ValueError("Mode không hợp lệ! Chọn mode từ 0 đến 3.")
    
    # In mode đang thực hiện
    delta_formulas = {
        0: "tuyệt đối, tiên nghiệm: (q^{count} * |x_1 - x_0|) / (1 - q)",
        1: "tuyệt đối, hậu nghiệm: (q * |x_n - x_n-1|) / (1 - q)",
        2: "tương đối, tiên nghiệm: [(q^{count} * |x_1 - x_0|) / (1 - q)] / |x_n|",
        3: "tương đối, hậu nghiệm: [(q * |x_n - x_n-1|) / (1 - q)] / |x_n|"
    }
    print(f"Mode đang thực hiện: {delta_formulas[mode]}")
    
    # Bước 1: Khởi tạo
    count = 1
    x_n_minus_1 = (a + b) / 2
    print(f"Giá trị ban đầu: x_0 = (a + b)/2 = {x_n_minus_1}")
    
    # Tính x_1 = phi(x_0) lần đầu tiên
    x_n = phi(x_n_minus_1)
    
    # Lưu x_0 và x_1 cho mode tiên nghiệm
    x_0 = x_n_minus_1
    x_1 = x_n
    initial_diff = abs(x_1 - x_0)  # |x_1 - x_0| cho tiên nghiệm
    
    print(f"{'n':<5} | {'x_n':<25} | Sai số δ ({'tuyệt đối' if mode in [0, 1] else 'tương đối'})")
    print("-" * 60)
    
    solution_found = False
    solution_iteration = -1
    solution_value = None
    
    while count < max_iterations:
        # Bước 2: Tính sai số δ
        if mode == 0:
            # Tuyệt đối, tiên nghiệm: (q^count * |x_1 - x_0|) / (1 - q)
            delta = (q**count * initial_diff) / (1 - q)
        elif mode == 1:
            # Tuyệt đối, hậu nghiệm: (q * |x_n - x_n-1|) / (1 - q)
            delta = (q * abs(x_n - x_n_minus_1)) / (1 - q)
        elif mode == 2:
            # Tương đối, tiên nghiệm: [(q^count * |x_1 - x_0|) / (1 - q)] / |x_n|
            absolute_error = (q**count * initial_diff) / (1 - q)
            delta = absolute_error / abs(x_n) if x_n != 0 else float('inf')
        else:
            # Tương đối, hậu nghiệm: [(q * |x_n - x_n-1|) / (1 - q)] / |x_n|
            absolute_error = (q * abs(x_n - x_n-1)) / (1 - q)
            delta = absolute_error / abs(x_n) if x_n != 0 else float('inf')
        
        # In kết quả
        print(f"{count:<5} | {x_n:<25.10f} | {delta:.6e}")
        
        # Bước 3: Kiểm tra δ = 0
        if delta == 0:
            print(f"-> Sai số δ = 0 tại vòng lặp {count}, x_n = {x_n:.10f} là nghiệm đúng.")
            # Lặp thêm một lần nữa
            count += 1
            x_n_minus_1 = x_n
            x_n = phi(x_n_minus_1)
            print(f"{count:<5} | {x_n:<25.10f} | (Lặp thêm lần cuối)")
            return x_n
        
        # Bước 4: Kiểm tra δ < error
        if delta < error and not solution_found:
            solution_found = True
            solution_iteration = count
            solution_value = x_n
            print(f"-> Đã tìm thấy nghiệm tại vòng lặp {count} với sai số δ = {delta:.6e} < {error:.6e}")
            if not extra_iteration:
                # Lặp thêm một lần nữa trước khi trả về
                count += 1
                x_n_minus_1 = x_n
                x_n = phi(x_n_minus_1)
                print(f"{count:<5} | {x_n:<25.10f} | (Lặp thêm lần cuối)")
                return x_n
        
        # Bước 5: Kiểm tra vòng lặp bổ sung
        if solution_found and count == solution_iteration + 1:
            print(f"-> Đây là vòng lặp thứ k+1 = {count} sau khi đã tìm thấy nghiệm tại vòng lặp thứ k = {solution_iteration}")
            return x_n
        
        # Bước 6: Cập nhật x_n-1 và x_n cho vòng lặp tiếp theo
        count += 1
        x_n_minus_1 = x_n
        x_n = phi(x_n_minus_1)
    
    if solution_found:
        return solution_value
    
    raise ValueError("Phương pháp không hội tụ sau số lần lặp tối đa.")


#--------------------------------------------------------- NHẬP f, phi_sym và a, b, error, q ----------------------------------------------------------
if __name__ == "__main__":

    f_expr = "x**5 -17*x +2"
    f_sym, f = define_functions(f_expr)
    phi_expr = "(x**5 + 2)/17"
    
    try:
        result = fixed_point_iteration(f_sym, f, phi_expr, a=0, b=1, error=1e-10, q=5/17, mode=0, max_iterations=1000, extra_iteration=True)
        print(f"Nghiệm gần đúng: {result:.10f}")
        
        try:
            f_result = f(result)
            print(f"Giá trị hàm tại nghiệm: f({result:.10f}) = {f_result:.10e}")
        except (ValueError, RuntimeWarning) as e:
            print(f"Không thể tính giá trị hàm tại nghiệm do lỗi: {e}")
        
        x_values = np.linspace(-5, 5, 1000)
        y_values = []
        for x in x_values:
            try:
                y_values.append(f(x))
            except ValueError:
                y_values.append(np.nan)
        
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