import numpy as np

# Hàm tính giá trị đa thức tại x
def evaluate_polynomial(coeffs, x):
    result = 0
    for i, coef in enumerate(coeffs):
        result += coef * (x ** (len(coeffs) - 1 - i))
    return result

# Hàm tính đạo hàm của đa thức
def derivative(coeffs):
    n = len(coeffs) - 1
    if n == 0:
        return [0]
    deriv_coeffs = []
    for i in range(n):
        deriv_coeffs.append(coeffs[i] * (n - i))
    return deriv_coeffs

# Hàm tính giới hạn nghiệm (Cauchy Bound)
def cauchy_bound(coeffs):
    n = len(coeffs) - 1
    a_n = coeffs[0]
    max_ratio = max(abs(coef / a_n) for coef in coeffs[1:])
    R = 1 + max_ratio
    return R

# Hàm xác định dấu của một số (-1 nếu âm, 1 nếu dương, 0 nếu bằng 0)
def sign(x):
    return np.sign(x)

# Hàm tìm nghiệm bằng phương pháp chia đôi
def bisection_method(f, a, b, error=1e-6, max_iter=1000, mode=0):
    mode_map = {
        0: ("a_priori", "relative"),
        1: ("a_priori", "absolute"),
        2: ("a_posteriori", "relative"),
        3: ("a_posteriori", "absolute")
    }
    
    if mode not in mode_map:
        raise ValueError("mode phải là 0, 1, 2 hoặc 3.")
    
    method, error_type = mode_map[mode]
    
    if f(a) * f(b) >= 0:
        return None, []
    if a >= b:
        raise ValueError("Hãy nhập a < b.")
    if error <= 0:
        raise ValueError("Sai số phải là số dương.")
    
    count = 0
    x_next = (a + b) * 0.5
    prev_x = float('inf')
    delta = float('inf')
    sfm = sign(f(x_next))
    sfa = sign(f(a))
    
    intermediate_data = []
    solution_found = False
    solution_x = None
    
    while count < max_iter:
        if solution_found:
            break
        
        if sfm == 0:
            solution_found = True
            solution_x = x_next
            delta = 0
            intermediate_data.append([count + 1, a, b, "+" if (sfm * sfa) > 0 else "-", x_next, delta])
            break
        
        if method == "a_priori":
            if error_type == "absolute":
                delta = abs(b - a)
            else:
                delta = abs(b - a) / abs(x_next) if x_next != 0 else float('inf')
        else:
            if error_type == "absolute":
                delta = abs(x_next - prev_x)
            else:
                delta = abs(x_next - prev_x) / abs(x_next) if x_next != 0 else abs(x_next - prev_x)
        
        sign_product = "+" if (sfm * sfa) > 0 else "-"
        intermediate_data.append([count + 1, a, b, sign_product, x_next, delta])
        
        if delta < error:
            solution_found = True
            solution_x = x_next
            break
        
        prev_x = x_next
        
        if sfm != sfa:
            b = x_next
        else:
            a = x_next
            sfa = sign(f(a))
        
        x_next = (a + b) * 0.5
        sfm = sign(f(x_next))
        count += 1
    
    return solution_x, intermediate_data

# Mode 0 và 1: Tìm khoảng phân ly nghiệm
def find_isolated_intervals(coeffs, epsilon, max_iter, mode="absolute"):
    R = cauchy_bound(coeffs)
    intervals = []
    intermediate_data = []
    step = 0.01
    x = -R
    points = []
    while x <= R:
        points.append(x)
        x += step
    
    for i in range(len(points) - 1):
        x1, x2 = points[i], points[i + 1]
        p1 = evaluate_polynomial(coeffs, x1)
        p2 = evaluate_polynomial(coeffs, x2)
        
        if p1 * p2 < 0:
            interval_data = []
            iteration = 0
            while iteration < max_iter:
                length = x2 - x1
                mid = (x1 + x2) / 2
                p_mid = evaluate_polynomial(coeffs, mid)
                interval_data.append([iteration + 1, x1, x2, mid, p1, p_mid, p2, length])
                
                if mode == "absolute":
                    if length < epsilon:
                        intervals.append((x1, x2))
                        intermediate_data.append(interval_data)
                        break
                else:
                    if length / abs(mid) < epsilon and mid != 0:
                        intervals.append((x1, x2))
                        intermediate_data.append(interval_data)
                        break
                
                if p1 * p_mid < 0:
                    x2 = mid
                    p2 = p_mid
                else:
                    x1 = mid
                    p1 = p_mid
                iteration += 1
    
    return intervals, intermediate_data

# Mode 2 và 3: Tìm nghiệm thực của đa thức
def find_real_roots(coeffs, epsilon, max_iter, mode="absolute"):
    intervals, _ = find_isolated_intervals(coeffs, epsilon, max_iter, mode)
    roots = []
    all_intermediate_data = []
    
    bisection_mode = 3 if mode == "absolute" else 2  # 3: Hậu nghiệm + Sai số tuyệt đối, 2: Hậu nghiệm + Sai số tương đối
    
    for a, b in intervals:
        f = lambda x: evaluate_polynomial(coeffs, x)
        root, intermediate_data = bisection_method(f, a, b, epsilon, max_iter, bisection_mode)
        if root is not None:
            roots.append(root)
            all_intermediate_data.append((root, intermediate_data))
    
    return roots, all_intermediate_data

# Mode 4, 5, 6, 7: Tìm max, min của f(x) hoặc |f(x)| với sai số
def find_extrema(coeffs, a, b, epsilon, max_iter, mode="absolute", use_absolute=False):
    # Tính đạo hàm
    deriv_coeffs = derivative(coeffs)
    
    # Tìm các điểm cực trị (nghiệm của f'(x) = 0) trong khoảng [a, b]
    intervals, intermediate_data_intervals = find_isolated_intervals(deriv_coeffs, epsilon, max_iter, mode)
    critical_points = []
    all_intermediate_data = []
    
    bisection_mode = 3 if mode == "absolute" else 2  # 3: Hậu nghiệm + Sai số tuyệt đối, 2: Hậu nghiệm + Sai số tương đối
    
    for x1, x2 in intervals:
        if x1 < a or x2 > b:  # Chỉ xét các khoảng nằm trong [a, b]
            continue
        f_deriv = lambda x: evaluate_polynomial(deriv_coeffs, x)
        point, intermediate_data = bisection_method(f_deriv, x1, x2, epsilon, max_iter, bisection_mode)
        if point is not None:
            critical_points.append(point)
            all_intermediate_data.append((point, intermediate_data))
    
    # Tìm các nghiệm của f(x) = 0 trong khoảng [a, b]
    roots, roots_intermediate_data = find_real_roots(coeffs, epsilon, max_iter, mode)
    roots = [root for root in roots if a <= root <= b]  # Chỉ lấy các nghiệm trong [a, b]
    
    # Thêm các điểm biên và các nghiệm
    points = [a, b] + critical_points + roots
    points = sorted(list(set(points)))  # Loại bỏ trùng lặp và sắp xếp
    
    # Tính giá trị f(x) hoặc |f(x)| tại các điểm
    intermediate_values = []
    for x in points:
        f_x = evaluate_polynomial(coeffs, x)
        value = abs(f_x) if use_absolute else f_x
        intermediate_values.append([x, f_x, value])
    
    # Tìm max và min
    values = [abs(evaluate_polynomial(coeffs, x)) if use_absolute else evaluate_polynomial(coeffs, x) for x in points]
    max_val = max(values)
    min_val = min(values)
    
    return min_val, max_val, intermediate_values, intermediate_data_intervals, all_intermediate_data, roots, roots_intermediate_data

# Hàm in bảng với khoảng trống cố định
def print_table(headers, data, col_width=15):
    header_line = ""
    for header in headers:
        header_line += f"{header:<{col_width}}"
    print(header_line)
    print("-" * (col_width * len(headers)))
    
    for row in data:
        row_line = ""
        for item in row:
            if isinstance(item, (int, float)):
                row_line += f"{item:<{col_width}.6f}"
            else:
                row_line += f"{item:<{col_width}}"
        print(row_line)

# Hàm in đa thức
def print_polynomial(coeffs):
    terms = []
    degree = len(coeffs) - 1
    for i, coef in enumerate(coeffs):
        power = degree - i
        if coef != 0:
            if power == 0:
                terms.append(f"{coef:+.2f}")
            elif power == 1:
                terms.append(f"{coef:+.2f}x")
            else:
                terms.append(f"{coef:+.2f}x^{power}")
    polynomial = " ".join(terms).lstrip("+")
    return polynomial

# Hàm chính để chạy chương trình
def main():
    print("=== Chương trình giải các bài toán về phương trình đa thức ===")
    
    # Nhập đa thức
    degree = int(input("Nhập bậc của đa thức: "))
    coeffs = []
    print("Nhập các hệ số của đa thức (từ bậc cao nhất đến bậc 0):")
    for i in range(degree, -1, -1):
        coef = float(input(f"Hệ số của x^{i}: "))
        coeffs.append(coef)
    
    # Nhập khoảng [a, b]
    a = float(input("Nhập giới hạn dưới của khoảng [a, b]: "))
    b = float(input("Nhập giới hạn trên của khoảng [a, b]: "))
    
    # Chọn mode
    print("\nChọn mode (0-7):")
    print("0. Tìm khoảng phân ly nghiệm + Sai số tuyệt đối")
    print("1. Tìm khoảng phân ly nghiệm + Sai số tương đối")
    print("2. Tìm nghiệm thực (Hậu nghiệm, Sai số tương đối)")
    print("3. Tìm nghiệm thực (Hậu nghiệm, Sai số tuyệt đối)")
    print("4. Tìm max, min của f(x) + Sai số tương đối")
    print("5. Tìm max, min của f(x) + Sai số tuyệt đối")
    print("6. Tìm max, min của |f(x)| + Sai số tương đối")
    print("7. Tìm max, min của |f(x)| + Sai số tuyệt đối")
    mode_choice = int(input("Nhập mode (0-7): "))
    
    # Chọn sai số hoặc số lần lặp tối đa
    use_epsilon = input("Bạn muốn nhập sai số (e) hay số lần lặp tối đa (i)? (e/i): ").lower()
    if use_epsilon == 'e':
        epsilon = float(input("Nhập sai số epsilon: "))
        max_iter = 1000
    else:
        max_iter = int(input("Nhập số lần lặp tối đa: "))
        epsilon = (b - a) / (2 ** max_iter)
    
    # In đầu vào
    print("\n=== Đầu vào ===")
    print(f"Đa thức: f(x) = {print_polynomial(coeffs)}")
    print(f"Khoảng [a, b]: [{a}, {b}]")
    print(f"Sai số epsilon: {epsilon}")
    print(f"Số lần lặp tối đa: {max_iter}")
    
    # Xác định mode
    mode = "absolute" if mode_choice in [0, 3, 5, 7] else "relative"
    use_absolute = mode_choice in [6, 7]
    
    # Xử lý từng bài toán
    if mode_choice in [0, 1]:  # Mode 0 và 1: Tìm khoảng phân ly nghiệm
        print("\n=== Phương pháp sử dụng ===")
        print(f"Tìm khoảng phân ly nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'}")
        
        intervals, intermediate_data = find_isolated_intervals(coeffs, epsilon, max_iter, mode)
        
        print("\nBảng giá trị trung gian:")
        headers = ["Lần lặp", "x1", "x2", "Mid", "p(x1)", "p(Mid)", "p(x2)", "Độ dài"]
        col_width = 15
        header_line = ""
        for header in headers:
            header_line += f"{header:<{col_width}}"
        print(header_line)
        print("-" * (col_width * len(headers)))
        
        for i, interval_data in enumerate(intermediate_data):
            print(f"\nKhoảng cách ly {i+1}:")
            for row in interval_data:
                row_line = ""
                for item in row:
                    if isinstance(item, (int, float)):
                        row_line += f"{item:<{col_width}.6f}"
                    else:
                        row_line += f"{item:<{col_width}}"
                print(row_line)
            if i < len(intermediate_data) - 1:
                print("-" * (col_width * len(headers)))
        
        print("\n=== Đầu ra ===")
        result_data = []
        for i, interval in enumerate(intervals):
            result_data.append([i+1, f"[{interval[0]:.6f}, {interval[1]:.6f}]"])
        headers = ["Khoảng", "Khoảng cách ly"]
        print_table(headers, result_data)
    
    elif mode_choice in [2, 3]:  # Mode 2 và 3: Tìm nghiệm thực
        print("\n=== Phương pháp sử dụng ===")
        print(f"Phương pháp chia đôi - Hậu nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'}")
        
        roots, all_intermediate_data = find_real_roots(coeffs, epsilon, max_iter, mode)
        
        print("\nBảng giá trị trung gian:")
        headers = ["Lần lặp", "a", "b", "f(a)*f(Mid)", "Mid", "Sai số"]
        col_width = 15
        header_line = ""
        for header in headers:
            header_line += f"{header:<{col_width}}"
        print(header_line)
        print("-" * (col_width * len(headers)))
        
        for i, (root, intermediate_data) in enumerate(all_intermediate_data):
            print(f"\nNghiệm {i+1}: {root:.6f}")
            for row in intermediate_data:
                row_line = ""
                for item in row:
                    if isinstance(item, (int, float)):
                        row_line += f"{item:<{col_width}.6f}"
                    else:
                        row_line += f"{item:<{col_width}}"
                print(row_line)
            if i < len(all_intermediate_data) - 1:
                print("-" * (col_width * len(headers)))
        
        print("\n=== Đầu ra ===")
        result_data = []
        for i, root in enumerate(roots):
            result_data.append([i+1, root])
        headers = ["Nghiệm", "Giá trị"]
        print_table(headers, result_data)
    
    elif mode_choice in [4, 5, 6, 7]:  # Mode 4, 5, 6, 7: Tìm max, min với sai số
        print("\n=== Phương pháp sử dụng ===")
        print(f"Phương pháp chia đôi - Hậu nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'} để tìm cực trị")
        
        min_val, max_val, intermediate_values, intermediate_data_intervals, all_intermediate_data, roots, roots_intermediate_data = find_extrema(coeffs, a, b, epsilon, max_iter, mode, use_absolute)
        
        # In bảng giá trị trung gian: Tìm khoảng phân ly nghiệm của f'(x)
        print("\nBảng giá trị trung gian (Tìm khoảng phân ly nghiệm của f'(x)):")
        headers = ["Lần lặp", "x1", "x2", "Mid", "p(x1)", "p(Mid)", "p(x2)", "Độ dài"]
        col_width = 15
        header_line = ""
        for header in headers:
            header_line += f"{header:<{col_width}}"
        print(header_line)
        print("-" * (col_width * len(headers)))
        
        for i, interval_data in enumerate(intermediate_data_intervals):
            print(f"\nKhoảng cách ly {i+1}:")
            for row in interval_data:
                row_line = ""
                for item in row:
                    if isinstance(item, (int, float)):
                        row_line += f"{item:<{col_width}.6f}"
                    else:
                        row_line += f"{item:<{col_width}}"
                print(row_line)
            if i < len(intermediate_data_intervals) - 1:
                print("-" * (col_width * len(headers)))
        
        # In bảng giá trị trung gian: Tìm điểm cực trị
        print("\nBảng giá trị trung gian (Tìm điểm cực trị):")
        headers = ["Lần lặp", "a", "b", "f(a)*f(Mid)", "Mid", "Sai số"]
        col_width = 15
        header_line = ""
        for header in headers:
            header_line += f"{header:<{col_width}}"
        print(header_line)
        print("-" * (col_width * len(headers)))
        
        for i, (point, intermediate_data) in enumerate(all_intermediate_data):
            print(f"\nĐiểm cực trị {i+1}: {point:.6f}")
            for row in intermediate_data:
                row_line = ""
                for item in row:
                    if isinstance(item, (int, float)):
                        row_line += f"{item:<{col_width}.6f}"
                    else:
                        row_line += f"{item:<{col_width}}"
                print(row_line)
            if i < len(all_intermediate_data) - 1:
                print("-" * (col_width * len(headers)))
        
        # In bảng giá trị trung gian: Tìm nghiệm của f(x) = 0 (nếu có)
        if use_absolute:  # Chỉ in nếu đang xét |f(x)|
            print("\nBảng giá trị trung gian (Tìm nghiệm của f(x) = 0):")
            headers = ["Lần lặp", "a", "b", "f(a)*f(Mid)", "Mid", "Sai số"]
            col_width = 15
            header_line = ""
            for header in headers:
                header_line += f"{header:<{col_width}}"
            print(header_line)
            print("-" * (col_width * len(headers)))
            
            for i, (root, intermediate_data) in enumerate(roots_intermediate_data):
                print(f"\nNghiệm {i+1}: {root:.6f}")
                for row in intermediate_data:
                    row_line = ""
                    for item in row:
                        if isinstance(item, (int, float)):
                            row_line += f"{item:<{col_width}.6f}"
                        else:
                            row_line += f"{item:<{col_width}}"
                    print(row_line)
                if i < len(roots_intermediate_data) - 1:
                    print("-" * (col_width * len(headers)))
        
        # In bảng giá trị tại các điểm
        print("\nBảng giá trị tại các điểm:")
        headers = ["x", "f(x)", "|f(x)|" if use_absolute else "f(x)"]
        print_table(headers, intermediate_values)
        
        print("\n=== Đầu ra ===")
        result_data = [
            ["Giá trị nhỏ nhất", min_val],
            ["Giá trị lớn nhất", max_val]
        ]
        headers = ["Hàm", f"{'|f(x)|' if use_absolute else 'f(x)'}"]
        print_table(headers, result_data)

if __name__ == "__main__":
    main()