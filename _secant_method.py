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
# tính số e:  chọn f(x) = math.log(x) - 1                [do  ln(e) - 1 = 0]               [hàm ln(x) đạo hàm = 1/x]
# tính số pi:  chọn f(x) = math.tan(x/4) - 1             [do tan(x/4) = 1]                 [hàm tan(x) đạo hàm = 1/(cos(x))^2]

# ------------- cú pháp python ----------------------------------------------
# hàm đa thức: x**5 - 0.2*x + 15.0         [x**5 = x^5]
# hàm lượng giác:
# hàm sin: math.sin(x)
# hàm cos: math.cos(x)
# hàm tan: math.tan(x)                     [tan(x) không xác định ở bội số lẻ của π/2]
# hàm cot (không có sẵn, phải định nghĩa qua hàm tan): 1/math.tan(x)            [cot(x) không xác định khi tan(x)=0, tức là x=0,π,2π,...]
# hàm căn bậc 2: math.sqrt(x)
# hàm căn bậc n: x**(1/n)
# hàm mũ và logarit: 
# hàm e mũ: math.exp(x)
# hàm log với cơ số khác số e: math.log(x, base)             [base là cơ số]
# hàm ln: 
#def log(x):
#    if x <= 0:
#        raise ValueError("log(x) chỉ xác định với x > 0.")
#    return math.log(x)
# ------------------------------------------------------------------------------------------------------------
import math
import sympy as sp      # thư viện dùng để tính đạo hàm và giải phương trình

# tính đạo hàm 1 và 2 lần của f rồi chuyển f, f', f'' thành dạng python có thể tính toán
def define_functions(expr):                  # expr: biểu thức toán học của f(x) được nhập dưới dạng sympy
    x = sp.symbols('x')                      # khai báo biến x
    f = sp.lambdify(x, expr, 'math')         # sp.lambdify(x, expr, 'math') chuyển biểu thức expr thành dạng có thể tính toán
    df_expr = sp.diff(expr, x)               # sp.diff(expr, x) để tính đạo hàm cấp 1
    ddf_expr = sp.diff(expr, x, 2)           # sp.diff(expr, x, 2) để tính đạo hàm cấp 2
    df = sp.lambdify(x, df_expr, 'math')     # df là biểu thức của hàm f'(x)
    ddf = sp.lambdify(x, ddf_expr, 'math')   # ddf là biểu thức của hàm f''(x)
    return f, df, ddf, df_expr
# f (biểu thức hàm f để tính f(a), f(b), f(d), f(x_n))
# df và ddf (biểu thức các hàm f' và f'' để kiểm tra điều kiện ban đầu của phương pháp và để tính df(a), ddf(a))
# df_expr (biểu thức đạo hàm cấp 1 của f, dùng để tính min_derivative và max_derivative)

# xác định dấu của một số, nếu là số âm thì trả về -1, dương thì 1, (= 0) thì return cũng là 0
def sign(x):
    return (x > 0) - (x < 0)

# hàm tìm min và max của f'(x) trên đoạn [a,b] bằng cách tìm điểm tới hạn
def min_max_derivative(a, b, df_expr):
    x = sp.symbols('x')
    critical_points = sp.solve(sp.diff(df_expr, x), x)     # giải phương trình f''(x)=0 để tìm các điểm tới hạn (sp.solve(sp.diff(df_expr, x), x))
    critical_points = [p.evalf() for p in critical_points if p.is_real and a <= p <= b]        # lọc các điểm tới hạn nằm trong đoạn [a,b]
    values = [abs(df_expr.subs(x, p)) for p in critical_points] + [abs(df_expr.subs(x, a)), abs(df_expr.subs(x, b))]     # tính giá trị đạo hàm tại các điểm tới hạn và tại biên a, b
    min_derivative = min(values)                   # chọn giá trị nhỏ nhất (min_derivative = m)
    max_derivative = max(values)                   # chọn giá trị lớn nhất (max_derivative = M)
    return min_derivative, max_derivative

# mặc định error = 1e-6 (dùng để tránh lỗi missing item, không cần sửa ở đây)
# max_iterations = 100 (đây là số vòng lặp tối đa, nếu đặt = 0 thì sẽ lặp cho đến khi đạt error nhỏ hơn error truyền vào)
# mode từ 0-3 để chọn công thức sai số
# extra_iteration = True để hiển thị thêm vòng lặp k+1 sau khi đã tìm thấy nghiệm
def secant_method(f, df, ddf, df_expr, a, b, error=1e-6, max_iterations=100, mode=1, extra_iteration=True):
    # kiểm tra chế độ đánh giá sai số
    if mode not in [0, 1, 2, 3]:
        raise ValueError("Chế độ đánh giá sai số không hợp lệ. Vui lòng chọn mode từ 0-3.")
    
    # kiểm tra điều kiện đầu vào
    sfa = sign(f(a))            # dấu của f(a)
    sfb = sign(f(b))            # dấu của f(b)
    if sfa * sfb >= 0:
        raise ValueError("Không thể dùng phương pháp dây cung do f(a)*f(b) lớn hơn hoặc bằng 0.")
    if a >= b:
        raise ValueError("Hãy nhập a (giá trị nhỏ hơn) trước, rồi nhập b (giá trị lớn hơn) sau.")
    if error <= 0:
        raise ValueError("Vui lòng nhập lại sai số, sai số phải là một giá trị dương.")
    
    # kiểm tra tính ổn định dấu của đạo hàm
    sfx = sign(df(a))           # dấu của f'(a)
    sfxx = sign(ddf(a))         # dấu của f''(a)
    # nếu f'(x) hoặc f''(x) thay đổi dấu trên đoạn [a,b] => phương pháp dây cung không hội tụ
    for x in [a + (b-a)*i/1000 for i in range(1001)]:
        if sign(df(x)) != sfx or sign(ddf(x)) != sfxx:
            raise ValueError("Phương pháp dây cung không thực hiện được do dấu của đạo hàm không ổn định")

    # chọn mốc d và xấp xỉ đầu
    if sfa * sfxx > 0:
        d = a
        x_n = b
    else:
        d = b
        x_n = a
    
    # tính m và M qua hàm tìm min max
    min_derivative, max_derivative = min_max_derivative(a, b, df_expr)
    
    # hiển thị thông tin về chế độ đánh giá sai số
    mode_names = [
        "Sai số tuyệt đối theo công thức sai số mục tiêu",
        "Sai số tuyệt đối theo công thức 2 xấp xỉ liên tiếp",
        "Sai số tương đối theo công thức sai số mục tiêu",
        "Sai số tương đối theo công thức 2 xấp xỉ liên tiếp"
    ]
    print(f"Chế độ đánh giá sai số: {mode} - {mode_names[mode]}")
    
    count = 0       # đếm số vòng lặp
    solution_found = False      # biến đánh dấu đã tìm thấy nghiệm
    solution_iteration = -1     # biến lưu lần lặp tìm thấy nghiệm
    solution_value = None       # biến lưu giá trị nghiệm
    
    while count < max_iterations:
        # bắt lỗi để tránh lỗi chia cho số rất nhỏ
        if abs(f(x_n) - f(d)) < 1e-12:
            raise ValueError("Phép chia cho số rất nhỏ hoặc 0 xảy ra")
        
        # công thức lặp của phương pháp dây cung để tính nghiệm mới x_n+1 = x_next
        x_next = x_n - f(x_n) * (x_n - d) / (f(x_n) - f(d))
        
        # tránh chia cho 0 khi tính sai số tương đối
        if mode in [2, 3] and abs(x_next) < 1e-12:
            raise ValueError("Không thể tính sai số tương đối do xấp xỉ quá gần 0")
        
        # tính delta theo chế độ được chọn
        if mode == 0:
            # Sai số tuyệt đối theo công thức sai số mục tiêu
            delta = abs(f(x_next)) / min_derivative
        elif mode == 1:
            # Sai số tuyệt đối theo công thức 2 xấp xỉ liên tiếp
            delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
        elif mode == 2:
            # Sai số tương đối theo công thức sai số mục tiêu
            delta = abs(f(x_next)) / (min_derivative * abs(x_next))
        else:
            # Sai số tương đối theo công thức 2 xấp xỉ liên tiếp
            delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n) / abs(x_next)

        # in output sau mỗi bước lặp
        print(f"Iteration {count}: d = {d:.10f}, x_0 = {x_n:.10f}, x_n = {x_next:.10f}, delta = {delta:.6e}")
        
        # kiểm tra điều kiện dừng: nếu sai số nhỏ hơn error => đánh dấu đã tìm thấy nghiệm
        if delta < error and not solution_found:
            solution_found = True
            solution_iteration = count
            solution_value = x_next
            print(f"-> Đã tìm thấy nghiệm tại vòng lặp {count} với sai số {delta:.6e} < {error:.6e}")
            
            # Nếu không cần hiển thị vòng lặp thêm, trả về kết quả ngay
            if not extra_iteration:
                return solution_value
        
        # Nếu đã tìm thấy nghiệm và đã thực hiện thêm một vòng lặp, trả về kết quả
        if solution_found and count > solution_iteration:
            print(f"-> Đây là vòng lặp thứ k+1 = {count} sau khi đã tìm thấy nghiệm tại vòng lặp thứ k = {solution_iteration}")
            return solution_value
        
        # cập nhật giá trị và tiếp tục vòng lặp
        x_n = x_next
        count += 1
    
    # Nếu đã tìm thấy nghiệm nhưng chưa thực hiện thêm vòng lặp (do đã đạt max_iterations)
    if solution_found:
        return solution_value
        
    raise ValueError("Phương pháp không hội tụ sau số lần lặp tối đa.")

# hàm main bọc trong try-except: Nếu đầu vào không hợp lệ, in thông báo lỗi thay vì dừng chương trình đột ngột
if __name__ == "__main__":
    expr = sp.sympify('x**5 - 7')        # nhập hàm f(x)
    f, df, ddf, df_expr = define_functions(expr) 
    try:
        result = secant_method(f, df, ddf, df_expr, a = 1.0, b = 2.0, error = 1e-7, mode = 0, extra_iteration = 1)  # extra_iteration để thêm 1 vòng lặp sau khi tìm được kết quả
        print(f"Nghiệm gần đúng: {result:.10f}")
        print(f"Giá trị hàm tại nghiệm: f({result:.10f}) = {f(result):.10e}")
    # bắt lỗi: hàm nhận giá trị đầu vào không hợp lệ nhưng đúng về kiểu dữ liệu
    except ValueError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        
# mode 0: sai số tuyệt đối theo công thức sai số mục tiêu: delta = abs(f(x_next)) / min_derivative
# mode 1: sai số tuyệt đối theo công thức 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
# mode 2: sai số tương đối theo công thức sai số mục tiêu: delta = abs(f(x_next)) / (min_derivative * abs(x_next))
# mode 3: sai số tương đối theo công thức 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n) / abs(x_next)  
      
import numpy as np
import matplotlib.pyplot as plt

# Tạo mảng giá trị x từ a đến b với 1000 điểm
x_values = np.linspace(-2, 3, 1000)
y_values = [f(x) for x in x_values]

# Vẽ đồ thị
#plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label="f(x) = x^5 - 7", color="blue")  # Đường cong f(x)
plt.axhline(0, color='black', linewidth=0.5)  # Trục Ox
plt.axvline(0, color='black', linewidth=0.5)  # Trục Oy
plt.axvline(result, color='red', linestyle="--", label=f"Nghiệm x ≈ {result:.6f}")  # Đánh dấu nghiệm

# Thiết lập nhãn và tiêu đề
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Đồ thị hàm số f(x)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)  # Lưới

# Hiển thị đồ thị
plt.show()







# # CODE CŨ
# # import math
# # import sympy as sp      # thư viện dùng để tính đạo hàm và giải phương trình

# # # tính đạo hàm 1 và 2 lần của f rồi chuyển f, f', f'' thành dạng python có thể tính toán
# # def define_functions(expr):                  # expr: biểu thức toán học của f(x) được nhập dưới dạng sympy
# #     x = sp.symbols('x')                      # khai báo biến x
# #     f = sp.lambdify(x, expr, 'math')         # sp.lambdify(x, expr, 'math') chuyển biểu thức expr thành dạng có thể tính toán
# #     df_expr = sp.diff(expr, x)               # sp.diff(expr, x) để tính đạo hàm cấp 1
# #     ddf_expr = sp.diff(expr, x, 2)           # sp.diff(expr, x, 2) để tính đạo hàm cấp 2
# #     df = sp.lambdify(x, df_expr, 'math')     # df là biểu thức của hàm f'(x)
# #     ddf = sp.lambdify(x, ddf_expr, 'math')   # ddf là biểu thức của hàm f''(x)
# #     return f, df, ddf, df_expr
# # # f (biểu thức hàm f để tính f(a), f(b), f(d), f(x_n))
# # # df và ddf (biểu thức các hàm f' và f'' để kiểm tra điều kiện ban đầu của phương pháp và để tính df(a), ddf(a))
# # # df_expr (biểu thức đạo hàm cấp 1 của f, dùng để tính min_derivative và max_derivative)

# # # xác định dấu của một số, nếu là số âm thì trả về -1, dương thì 1, (= 0) thì return cũng là 0
# # def sign(x):
# #     return (x > 0) - (x < 0)

# # # hàm tìm min và max của f'(x) trên đoạn [a,b] bằng cách tìm điểm tới hạn
# # def min_max_derivative(a, b, df_expr):
# #     x = sp.symbols('x')
# #     critical_points = sp.solve(sp.diff(df_expr, x), x)     # giải phương trình f''(x)=0 để tìm các điểm tới hạn (sp.solve(sp.diff(df_expr, x), x))
# #     critical_points = [p.evalf() for p in critical_points if p.is_real and a <= p <= b]        # lọc các điểm tới hạn nằm trong đoạn [a,b]
# #     values = [abs(df_expr.subs(x, p)) for p in critical_points] + [abs(df_expr.subs(x, a)), abs(df_expr.subs(x, b))]     # tính giá trị đạo hàm tại các điểm tới hạn và tại biên a, b
# #     min_derivative = min(values)                   # chọn giá trị nhỏ nhất (min_derivative = m)
# #     max_derivative = max(values)                   # chọn giá trị lớn nhất (max_derivative = M)
# #     return min_derivative, max_derivative

# # # mặc định error = 1e-6 (dùng để tránh lỗi missing item, không cần sửa ở đây)
# # # max_iterations = 100 (đây là số vòng lặp tối đa, nếu đặt = 0 thì sẽ lặp cho đến khi đạt error nhỏ hơn error truyền vào)
# # def secant_method(f, df, ddf, df_expr, a, b, error=1e-6, max_iterations=100):
# #     # kiểm tra điều kiện đầu vào
# #     sfa = sign(f(a))            # dấu của f(a)
# #     sfb = sign(f(b))            # dấu của f(b)
# #     if sfa * sfb >= 0:
# #         raise ValueError("Không thể dùng phương pháp dây cung do f(a)*f(b) lớn hơn hoặc bằng 0.")
# #     if a >= b:
# #         raise ValueError("Hãy nhập a (giá trị nhỏ hơn) trước, rồi nhập b (giá trị lớn hơn) sau.")
# #     if error <= 0:
# #         raise ValueError("Vui lòng nhập lại sai số, sai số phải là một giá trị dương.")
    
# #     # kiểm tra tính ổn định dấu của đạo hàm
# #     sfx = sign(df(a))           # dấu của f'(a)
# #     sfxx = sign(ddf(a))         # dấu của f''(a)
# #     # nếu f'(x) hoặc f''(x) thay đổi dấu trên đoạn [a,b] => phương pháp dây cung không hội tụ
# #     for x in [a + (b-a)*i/1000 for i in range(1001)]:
# #         if sign(df(x)) != sfx or sign(ddf(x)) != sfxx:
# #             raise ValueError("Phương pháp dây cung không thực hiện được do dấu của đạo hàm không ổn định")

# #     # chọn mốc d và xấp xỉ đầu
# #     if sfa * sfxx > 0:
# #         d = a
# #         x_n = b
# #     else:
# #         d = b
# #         x_n = a
    
# #     # tính m và M qua hàm tìm min max
# #     min_derivative, max_derivative = min_max_derivative(a, b, df_expr)
    
# #     count = 0       # đếm số vòng lặp
# #     while count < max_iterations:
# #         # bắt lỗi để tránh lỗi chia cho số rất nhỏ
# #         if abs(f(x_n) - f(d)) < 1e-12:
# #             raise ValueError("Phép chia cho số rất nhỏ hoặc 0 xảy ra")
        
# #         # công thức lặp của phương pháp dây cung để tính nghiệm mới x_n+1 = x_next
# #         x_next = x_n - f(x_n) * (x_n - d) / (f(x_n) - f(d))
        
# # # ----- chọn công thức sai số ----------------------------------------------------------------------------------------------------------        
# #         delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)       # công thức sai số theo 2 xấp xỉ liên tiếp
# # #        delta = abs(f(x_next)) / min_derivative                                              # công thức sai số mục tiêu
# # # --------------------------------------------------------------------------------------------------------------------------------------    

# #         # in output sau mỗi bước lặp; hiển thị tới 10 chữ số thập phân sau dấu phẩy (.10f), delta 6 chữ số thập phân sau dấu phẩy kèm thứ nguyên (.6e)
# #         print(f"Iteration {count}: d = {d:.10f}, x_0 = {x_n:.10f}, x_n = {x_next:.10f}, delta = {delta:.6e}")
        
# #         # kiểm tra điều kiện dừng: nếu sai số nhỏ hơn error => trả về nghiệm
# #         if delta < error:
# #             return x_next
        
# #         # cập nhật giá trị và tiếp tục vòng lặp
# #         x_n = x_next
# #         count += 1
    
# #     raise ValueError("Phương pháp không hội tụ sau số lần lặp tối đa.")      # nếu quá số lần lặp max thì báo lỗi

# # # ----- cần chỉnh sửa tại đây --------------------------------------------------------------------------------
# # # hàm main bọc trong try-except: Nếu đầu vào không hợp lệ, in thông báo lỗi thay vì dừng chương trình đột ngột
# # if __name__ == "__main__":
# #     expr = sp.sympify('x**5 - 0.2*x + 15.0')        # nhập hàm f(x)
# #     f, df, ddf, df_expr = define_functions(expr)
    
# #     try:
# #         result = secant_method(f, df, ddf, df_expr, -2, -1, error=1e-7) # nhập khoảng cách ly nghiệm lần lượt là a và b, sai số [1e-7 = 1*10^(-7)]
# #         print(f"Nghiệm gần đúng: {result:.10f}")
# #     # bắt lỗi: hàm nhận giá trị đầu vào không hợp lệ nhưng đúng về kiểu dữ liệu
# #     except ValueError as e:
# #         print(f"Lỗi: {e}")