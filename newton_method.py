import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def define_functions(f_expr):
    x = sp.Symbol('x')
    f_sym = sp.sympify(f_expr)
    f = sp.lambdify(x, f_sym, 'numpy')
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
    return min_derivative, max_derivative

def newton_method(f, df, ddf, a, b, error=1e-6, max_iterations=100, mode=1, extra_iteration=True):
    if mode not in [0, 1, 2, 3]:
        raise ValueError("Chế độ đánh giá sai số không hợp lệ.")
    
    x_n = a if f(a) * df(a) > 0 else b
    min_derivative, max_derivative = min_max_derivative(f_sym, a, b)
    
    print(f"{'n':<5}{'x_n':<20}{'|f(x_n)|/|x_n|' if mode in [2] else '|x_n - x_n-1|/|x_n|' if mode in [3] else '|f(x_n)|' if mode in [0] else '|x_n - x_n-1|'}")
    count = 0
    prev_x = None
    solution_found = False
    solution_iteration = -1
    solution_value = None
    
    while count < max_iterations:
        if abs(df(x_n)) < 1e-12:
            raise ValueError("Đạo hàm gần 0, phương pháp không hội tụ.")
        
        x_next = x_n - f(x_n) / df(x_n)
        
        if mode in [0, 2]:
            delta = abs(f(x_next)) / (min_derivative * (abs(x_next) if mode == 2 else 1))
        else:
            if prev_x is None:
                prev_x = x_n  # Bỏ qua lần đầu
                x_n = x_next
                count += 1
                continue
            delta = (max_derivative / (2 * min_derivative)) * (abs(x_next - prev_x) ** 2) / (abs(x_next) if mode == 3 else 1)
        
        print(f"{count:<5}{x_n:<20.10f}{(abs(f(x_n)) / abs(x_n)) if mode in [2] else (abs(x_n - prev_x) / abs(x_n)) if mode in [3] else abs(f(x_n)) if mode in [0] else abs(x_n - prev_x):.6e}")
        
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
    f_expr = "x**5 - 7"
    f_sym, f, df, ddf = define_functions(f_expr)
    
    try:
        result = newton_method(f, df, ddf, a=1, b=2, error=5e-8, mode=1, extra_iteration=True)
        print(f"Nghiệm gần đúng: {result:.10f}")
        print(f"Giá trị hàm tại nghiệm: f({result:.10f}) = {f(result):.10e}")
        
        # Vẽ đồ thị
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
        
    except ValueError as e:
        print(f"Lỗi: {e}")
