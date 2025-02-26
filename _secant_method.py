# phương pháp dây cung (sai số tuyệt đối) (sai số tương đối = sai số tuyệt đối / x)
# công thức sai số mục tiêu: delta = abs(f(x_next)) / min_derivative
# công thức sai số theo 2 xấp xỉ liên tiếp: delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
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

# tìm min và max của f'(x) trên đoạn [a,b]
def min_max_derivative(a, b, df_expr):
    x = sp.symbols('x')
    critical_points = sp.solve(sp.diff(df_expr, x), x)     # giải phương trình f''(x)=0 để tìm các điểm tới hạn (sp.solve(sp.diff(df_expr, x), x))
    critical_points = [p.evalf() for p in critical_points if p.is_real and a <= p <= b]        # lọc các điểm tới hạn nằm trong đoạn [a,b]
    values = [abs(df_expr.subs(x, p)) for p in critical_points] + [abs(df_expr.subs(x, a)), abs(df_expr.subs(x, b))]     # tính giá trị đạo hàm tại các điểm tới hạn và tại biên a, b
    min_derivative = min(values)                   # chọn giá trị nhỏ nhất (min_derivative = m)
    max_derivative = max(values)                   # chọn giá trị lớn nhất (max_derivative = M)
    return min_derivative, max_derivative

def secant_method(f, df, ddf, df_expr, a, b, error=1e-6, max_iterations=100):
    sfa, sfb = sign(f(a)), sign(f(b))
    if sfa * sfb >= 0:
        raise ValueError("(a, b) không phải khoảng cách ly nghiệm")
    
    sfx, sfxx = sign(df(a)), sign(ddf(a))
    for x in [a + (b-a)*i/1000 for i in range(1001)]:
        if sign(df(x)) != sfx or sign(ddf(x)) != sfxx:
            raise ValueError("Phương pháp dây cung không thực hiện được do dấu của đạo hàm không ổn định")
    
    if sfa * sfxx > 0:
        d, x_n = a, b
    else:
        d, x_n = b, a
    
    min_derivative, max_derivative = min_max_derivative(a, b, df_expr)
    
    count = 0
    while count < max_iterations:
        if abs(f(x_n) - f(d)) < 1e-12:
            raise ValueError("Phép chia cho số rất nhỏ hoặc 0 xảy ra")
        
        x_next = x_n - f(x_n) * (x_n - d) / (f(x_n) - f(d))
        
# ----- chọn công thức sai số ----------------------------------------------------------------------------------------------------------        
#        delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)       # công thức sai số theo 2 xấp xỉ liên tiếp
        delta = abs(f(x_next)) / min_derivative                                              # công thức sai số mục tiêu
# --------------------------------------------------------------------------------------------------------------------------------------    
        
        print(f"Iteration {count}: d = {d:.10f}, x_0 = {x_n:.10f}, x_n = {x_next:.10f}, delta = {delta:.6e}")
        
        if delta < error:
            return x_next
        
        x_n = x_next
        count += 1
    
    raise ValueError("Phương pháp không hội tụ sau số lần lặp tối đa.")

if __name__ == "__main__":
    expr = sp.sympify('x**5 - 0.2*x + 15.0')  # Người dùng có thể thay đổi biểu thức hàm f(x)
    f, df, ddf, df_expr = define_functions(expr)
    
    try:
        root = secant_method(f, df, ddf, df_expr, -2, -1, error=1e-7)
        print(f"Nghiệm gần đúng: {root:.10f}")
    except ValueError as e:
        print(f"Lỗi: {e}")
