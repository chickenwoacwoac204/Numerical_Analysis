# phương pháp dây cung
import math
import sympy as sp

def define_functions(expr):
    x = sp.symbols('x')
    f = sp.lambdify(x, expr, 'math')
    df_expr = sp.diff(expr, x)
    ddf_expr = sp.diff(expr, x, 2)
    df = sp.lambdify(x, df_expr, 'math')
    ddf = sp.lambdify(x, ddf_expr, 'math')
    return f, df, ddf, df_expr

def sign(x):
    return (x > 0) - (x < 0)

def min_max_derivative(a, b, df_expr):
    x = sp.symbols('x')
    critical_points = sp.solve(sp.diff(df_expr, x), x)
    critical_points = [p.evalf() for p in critical_points if p.is_real and a <= p <= b]
    values = [abs(df_expr.subs(x, p)) for p in critical_points] + [abs(df_expr.subs(x, a)), abs(df_expr.subs(x, b))]
    min_derivative = min(values)
    max_derivative = max(values)
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
        delta = (max_derivative - min_derivative) / min_derivative * abs(x_next - x_n)
        
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
