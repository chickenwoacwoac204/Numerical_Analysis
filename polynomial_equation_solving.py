import numpy as np
import matplotlib.pyplot as plt

class PolynomialSolver:
    """
    Lớp giải phương trình đa thức với nhiều phương pháp khác nhau
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs
        
    def evaluate_polynomial(self, x):
        """Tính giá trị đa thức tại x"""
        result = 0
        for i, coef in enumerate(self.coeffs):
            result += coef * (x ** (len(self.coeffs) - 1 - i))
        return result

    def derivative(self):
        """Tính đạo hàm của đa thức"""
        n = len(self.coeffs) - 1
        if n == 0:
            return [0]
        deriv_coeffs = []
        for i in range(n):
            deriv_coeffs.append(self.coeffs[i] * (n - i))
        return deriv_coeffs

    def cauchy_bound(self):
        """Tính giới hạn nghiệm (Cauchy Bound)"""
        n = len(self.coeffs) - 1
        a_n = self.coeffs[0]
        max_ratio = max(abs(coef / a_n) for coef in self.coeffs[1:])
        R = 1 + max_ratio
        return R

    @staticmethod
    def sign(x):
        """Xác định dấu của một số"""
        return np.sign(x)

    def bisection_method(self, a, b, error=1e-6, max_iter=1000, mode=0):
        """
        Tìm nghiệm bằng phương pháp chia đôi
        
        Args:
            a, b: Khoảng phân ly nghiệm
            error: Sai số cho phép
            max_iter: Số lần lặp tối đa
            mode: Chế độ tính sai số (0-3)
            
        Returns:
            solution_x: Nghiệm tìm được
            intermediate_data: Dữ liệu trung gian trong quá trình tính
        """
        mode_map = {
            0: ("a_priori", "relative"),
            1: ("a_priori", "absolute"), 
            2: ("a_posteriori", "relative"),
            3: ("a_posteriori", "absolute")
        }
        
        if mode not in mode_map:
            raise ValueError("mode phải là 0, 1, 2 hoặc 3.")
        
        method, error_type = mode_map[mode]
        
        f = lambda x: self.evaluate_polynomial(x)
        
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
        sfm = self.sign(f(x_next))
        sfa = self.sign(f(a))
        
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
                sfa = self.sign(f(a))
            
            x_next = (a + b) * 0.5
            sfm = self.sign(f(x_next))
            count += 1
        
        return solution_x, intermediate_data

    def find_isolated_intervals(self, epsilon, max_iter, mode="absolute", initial_step=None):
        """
        Tìm khoảng phân ly nghiệm
        
        Args:
            epsilon: Sai số cho phép
            max_iter: Số lần lặp tối đa
            mode: Loại sai số ("absolute" hoặc "relative")
            initial_step: Độ dài đoạn chia ban đầu (mặc định là epsilon)
            
        Returns:
            intervals: Danh sách các khoảng phân ly nghiệm
            intermediate_data: Dữ liệu trung gian trong quá trình tính
        """
        R = self.cauchy_bound()
        intervals = []
        intermediate_data = []
        
        # Sử dụng epsilon làm độ dài đoạn chia nếu initial_step không được chỉ định
        step = initial_step if initial_step is not None else epsilon
        
        # Đảm bảo step không quá lớn
        if step > R/5:
            step = R/5
        
        x = -R
        points = []
        while x <= R:
            points.append(x)
            x += step
        
        # Thêm điểm cuối nếu cần
        if points[-1] < R:
            points.append(R)
        
        for i in range(len(points) - 1):
            x1, x2 = points[i], points[i + 1]
            p1 = self.evaluate_polynomial(x1)
            p2 = self.evaluate_polynomial(x2)
            
            if p1 * p2 < 0:
                interval_data = []
                iteration = 0
                while iteration < max_iter:
                    length = x2 - x1
                    mid = (x1 + x2) / 2
                    p_mid = self.evaluate_polynomial(mid)
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

    def find_real_roots(self, epsilon_isolation, epsilon_root, max_iter, mode="absolute", initial_step=None):
        """
        Tìm nghiệm của đa thức với hai sai số khác nhau.
        
        Args:
            epsilon_isolation (float): Sai số cho việc tìm khoảng phân ly (Mode 0-1)
            epsilon_root (float): Sai số cho việc tìm nghiệm chính xác (Mode 2-3)
            max_iter (int): Số lần lặp tối đa
            mode (str): "absolute" hoặc "relative" - chế độ tính sai số
            initial_step (float, optional): Độ dài đoạn chia ban đầu cho việc tìm khoảng phân ly
            
        Returns:
            tuple: (danh sách nghiệm, dữ liệu trung gian tìm nghiệm, dữ liệu trung gian tìm khoảng phân ly)
        """
        # Tìm khoảng phân ly với epsilon_isolation
        intervals, isolation_data = self.find_isolated_intervals(
            epsilon_isolation, 
            max_iter, 
            mode,
            initial_step=initial_step
        )
        roots = []
        all_intermediate_data = []
        failed_intervals = []
        
        bisection_mode = 3 if mode == "absolute" else 2
        
        # Tìm nghiệm chính xác với epsilon_root
        for i, (a, b) in enumerate(intervals):
            root, intermediate_data = self.bisection_method(a, b, epsilon_root, max_iter, bisection_mode)
            if root is not None:
                roots.append(root)
                all_intermediate_data.append((root, intermediate_data))
            else:
                failed_intervals.append((i+1, a, b))
        
        # In thông báo về các khoảng không tìm được nghiệm
        if failed_intervals:
            print("\nCảnh báo: Không tìm được nghiệm trong các khoảng phân ly sau:")
            for idx, a, b in failed_intervals:
                print(f"Khoảng {idx}: [{a}, {b}]")
        
        return roots, all_intermediate_data, isolation_data

    def find_extrema(self, a, b, epsilon, max_iter, mode="absolute", use_absolute=False):
        """
        Tìm cực trị của hàm
        
        Args:
            a, b: Khoảng tìm cực trị
            epsilon: Sai số cho phép
            max_iter: Số lần lặp tối đa
            mode: Loại sai số ("absolute" hoặc "relative")
            use_absolute: Có tìm cực trị của |f(x)| không
            
        Returns:
            min_val, max_val: Giá trị cực tiểu và cực đại
            intermediate_values: Giá trị tại các điểm đặc biệt
            intermediate_data_intervals: Dữ liệu trung gian khi tìm khoảng phân ly
            all_intermediate_data: Dữ liệu trung gian khi tìm điểm cực trị
            roots: Các nghiệm của f(x) = 0
            roots_intermediate_data: Dữ liệu trung gian khi tìm nghiệm
        """
        deriv_solver = PolynomialSolver(self.derivative())
        
        intervals, intermediate_data_intervals = deriv_solver.find_isolated_intervals(epsilon, max_iter, mode)
        critical_points = []
        all_intermediate_data = []
        
        bisection_mode = 3 if mode == "absolute" else 2
        
        for x1, x2 in intervals:
            if x1 < a or x2 > b:
                continue
            point, intermediate_data = deriv_solver.bisection_method(x1, x2, epsilon, max_iter, bisection_mode)
            if point is not None:
                critical_points.append(point)
                all_intermediate_data.append((point, intermediate_data))
        
        roots, roots_intermediate_data = self.find_real_roots(epsilon, max_iter, mode)
        roots = [root for root in roots if a <= root <= b]
        
        points = [a, b] + critical_points + roots
        points = sorted(list(set(points)))
        
        intermediate_values = []
        for x in points:
            f_x = self.evaluate_polynomial(x)
            value = abs(f_x) if use_absolute else f_x
            intermediate_values.append([x, f_x, value])
        
        values = [abs(self.evaluate_polynomial(x)) if use_absolute else self.evaluate_polynomial(x) for x in points]
        max_val = max(values)
        min_val = min(values)
        
        return min_val, max_val, intermediate_values, intermediate_data_intervals, all_intermediate_data, roots, roots_intermediate_data

    def plot_results(self, a, b, mode_choice, roots=None, critical_points=None, min_val=None, max_val=None):
        """
        Vẽ đồ thị kết quả
        
        Args:
            a, b: Khoảng vẽ đồ thị
            mode_choice: Chế độ tính toán (0-7)
            roots: Danh sách các nghiệm (nếu có)
            critical_points: Danh sách các điểm cực trị (nếu có)
            min_val, max_val: Giá trị cực tiểu và cực đại (nếu có)
        """
        # Tạo dữ liệu cho đồ thị
        margin = (b - a) * 0.1
        x = np.linspace(a - margin, b + margin, 1000)
        y = np.array([self.evaluate_polynomial(xi) for xi in x])
        
        # Vẽ đồ thị hàm số
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-', label='f(x)')
        
        # Vẽ trục tọa độ
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Vẽ khoảng [a, b]
        plt.axvline(x=a, color='g', linestyle='--', alpha=0.5, label='x = a')
        plt.axvline(x=b, color='g', linestyle='--', alpha=0.5, label='x = b')
        
        # Vẽ các điểm đặc biệt
        if mode_choice in [0, 1]:  # Khoảng phân ly nghiệm
            if roots:
                for root in roots:
                    plt.plot([root], [0], 'ro', label='Nghiệm')
        elif mode_choice in [2, 3]:  # Nghiệm thực
            if roots:
                for root in roots:
                    plt.plot([root], [0], 'ro', label='Nghiệm')
        elif mode_choice in [4, 5, 6, 7]:  # Cực trị
            if critical_points:
                for point in critical_points:
                    y_val = self.evaluate_polynomial(point)
                    plt.plot([point], [y_val], 'go', label='Điểm cực trị')
            if min_val is not None and max_val is not None:
                plt.axhline(y=min_val, color='r', linestyle='--', alpha=0.5, label='Min')
                plt.axhline(y=max_val, color='r', linestyle='--', alpha=0.5, label='Max')
        
        # Cấu hình đồ thị
        plt.grid(True, alpha=0.3)
        plt.title('Đồ thị hàm số f(x)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        
        # Xử lý chồng lặp nhãn trong legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.show()

class PolynomialIO:
    """
    Lớp xử lý input/output cho bài toán đa thức
    """
    @staticmethod
    def print_table(headers, data):
        """
        In bảng dữ liệu với định dạng số theo yêu cầu
        - Cột lần lặp: số nguyên
        - Các cột giữa: số thực với 10 chữ số thập phân
        - Cột cuối: định dạng .6e
        """
        if not data:
            return
            
        # Xác định độ rộng cột
        col_widths = []
        for i, header in enumerate(headers):
            # Tìm độ rộng tối đa cho mỗi cột
            width = len(header)
            for row in data:
                cell_width = len(str(row[i]))
                if isinstance(row[i], (int, float)):
                    if i == 0:  # Cột lần lặp
                        cell_width = len(str(int(row[i])))
                    elif i == len(row) - 1:  # Cột cuối
                        cell_width = 13  # .6e format
                    else:  # Các cột giữa
                        cell_width = 17  # .10f format
                width = max(width, cell_width)
            col_widths.append(width)
        
        # In tiêu đề
        header_line = "|"
        for width, header in zip(col_widths, headers):
            header_line += f" {header:^{width}} |"
        print(header_line)
        print("-" * len(header_line))
        
        # In dữ liệu
        for row in data:
            line = "|"
            for i, (width, value) in enumerate(zip(col_widths, row)):
                if isinstance(value, (int, float)):
                    if i == 0:  # Cột lần lặp
                        line += f" {int(value):>{width}} |"
                    elif i == len(row) - 1:  # Cột cuối
                        line += f" {value:>{width}.6e} |"
                    else:  # Các cột giữa
                        line += f" {value:>{width}.10f} |"
                else:  # Giá trị chuỗi
                    line += f" {str(value):>{width}} |"
            print(line)

    @staticmethod
    def print_polynomial(coeffs):
        """In đa thức dưới dạng chuỗi"""
        terms = []
        n = len(coeffs) - 1
        for i, coef in enumerate(coeffs):
            if coef == 0:
                continue
            power = n - i
            if power == 0:
                term = f"{coef:+g}"
            elif power == 1:
                term = f"{coef:+g}x"
            else:
                term = f"{coef:+g}x^{power}"
            terms.append(term)
        if not terms:
            return "0"
        result = "".join(terms)
        return result[1:] if result[0] == '+' else result

def main():
    """Hàm chính để chạy chương trình"""
    try:
        print("=== Chương trình giải các bài toán về phương trình đa thức ===")
        
        # Nhập đa thức
        while True:
            try:
                degree = int(input("Nhập bậc của đa thức: "))
                if degree < 0:
                    print("Bậc của đa thức phải là số nguyên không âm.")
                    continue
                break
            except ValueError:
                print("Vui lòng nhập một số nguyên.")
        
        coeffs = []
        print("Nhập các hệ số của đa thức (từ bậc cao nhất đến bậc 0):")
        for i in range(degree, -1, -1):
            while True:
                try:
                    coef = float(input(f"Hệ số của x^{i}: "))
                    if i == degree and coef == 0:
                        print("Hệ số của bậc cao nhất không được bằng 0.")
                        continue
                    break
                except ValueError:
                    print("Vui lòng nhập một số thực.")
            coeffs.append(coef)
        
        # Nhập khoảng [a, b]
        while True:
            try:
                a = float(input("Nhập giới hạn dưới của khoảng [a, b]: "))
                b = float(input("Nhập giới hạn trên của khoảng [a, b]: "))
                if a >= b:
                    print("Giới hạn dưới phải nhỏ hơn giới hạn trên.")
                    continue
                break
            except ValueError:
                print("Vui lòng nhập các số thực.")
        
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
        
        while True:
            try:
                mode_choice = int(input("Nhập mode (0-7): "))
                if mode_choice not in range(8):
                    print("Mode phải từ 0 đến 7.")
                    continue
                break
            except ValueError:
                print("Vui lòng nhập một số nguyên từ 0 đến 7.")
        
        if mode_choice in [2, 3]:
            print("\nNhập sai số cho việc tìm khoảng phân ly (epsilon_0):")
            epsilon_isolation = float(input())
            while epsilon_isolation <= 0:
                print("Sai số phải dương. Vui lòng nhập lại:")
                epsilon_isolation = float(input())
                
            print("\nNhập sai số cho việc tìm nghiệm chính xác (epsilon_1):")
            epsilon_root = float(input())
            while epsilon_root <= 0:
                print("Sai số phải dương. Vui lòng nhập lại:")
                epsilon_root = float(input())
                
            print("\nBạn có muốn tùy chỉnh độ dài đoạn chia ban đầu không? (y/n):")
            customize_step = input().lower() == 'y'
            initial_step = None
            if customize_step:
                print("Nhập độ dài đoạn chia ban đầu (để trống để sử dụng epsilon_0):")
                step_input = input()
                if step_input:
                    initial_step = float(step_input)
                    while initial_step <= 0:
                        print("Độ dài đoạn chia phải dương. Vui lòng nhập lại:")
                        initial_step = float(input())
        else:
            print("\nNhập sai số epsilon:")
            epsilon = float(input())
            while epsilon <= 0:
                print("Sai số phải dương. Vui lòng nhập lại:")
                epsilon = float(input())
            
            print("\nBạn có muốn tùy chỉnh độ dài đoạn chia ban đầu không? (y/n):")
            customize_step = input().lower() == 'y'
            initial_step = None
            if customize_step:
                print("Nhập độ dài đoạn chia ban đầu (để trống để sử dụng epsilon):")
                step_input = input()
                if step_input:
                    initial_step = float(step_input)
                    while initial_step <= 0:
                        print("Độ dài đoạn chia phải dương. Vui lòng nhập lại:")
                        initial_step = float(input())
                
        print("\nNhập số lần lặp tối đa:")
        max_iter = int(input())
        while max_iter <= 0:
            print("Số lần lặp phải dương. Vui lòng nhập lại:")
            max_iter = int(input())
        
        # In đầu vào
        print("\n=== Đầu vào ===")
        print(f"Đa thức: f(x) = {PolynomialIO.print_polynomial(coeffs)}")
        print(f"Khoảng [a, b]: [{a}, {b}]")
        if mode_choice in [2, 3]:
            print(f"Sai số tìm khoảng phân ly (epsilon_0): {epsilon_isolation}")
            print(f"Sai số tìm nghiệm chính xác (epsilon_1): {epsilon_root}")
        else:
            print(f"Sai số epsilon: {epsilon}")
        print(f"Số lần lặp tối đa: {max_iter}")
        
        # Xác định mode
        mode = "absolute" if mode_choice in [0, 3, 5, 7] else "relative"
        use_absolute = mode_choice in [6, 7]
        
        try:
            # Xử lý từng bài toán
            if mode_choice in [0, 1]:  # Mode 0 và 1: Tìm khoảng phân ly nghiệm
                print("\n=== Phương pháp sử dụng ===")
                print(f"Tìm khoảng phân ly nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'}")
                
                intervals, intermediate_data = PolynomialSolver(coeffs).find_isolated_intervals(
                    epsilon, 
                    max_iter, 
                    mode,
                    initial_step=initial_step
                )
                
                if not intervals:
                    print("\nKhông tìm thấy khoảng phân ly nghiệm nào trong khoảng đã cho.")
                else:
                    print("\nBảng giá trị trung gian:")
                    headers = ["Lần lặp", "x1", "x2", "Mid", "p(x1)", "p(Mid)", "p(x2)", "Độ dài"]
                    PolynomialIO.print_table(headers, intermediate_data[0])
                    
                    print("\n=== Đầu ra ===")
                    result_data = []
                    for i, interval in enumerate(intervals):
                        result_data.append([i+1, f"[{interval[0]:.6f}, {interval[1]:.6f}]"])
                    headers = ["Khoảng", "Khoảng cách ly"]
                    PolynomialIO.print_table(headers, result_data)
                    
                    if intervals:
                        # Vẽ đồ thị
                        roots = [(interval[0] + interval[1]) / 2 for interval in intervals]
                        PolynomialSolver(coeffs).plot_results(a, b, mode_choice, roots=roots)
            
            elif mode_choice in [2, 3]:  # Mode 2 và 3: Tìm nghiệm thực
                print("\n=== Phương pháp sử dụng ===")
                print(f"Phương pháp chia đôi - Hậu nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'}")
                
                roots, all_intermediate_data, isolation_data = PolynomialSolver(coeffs).find_real_roots(
                    epsilon_isolation,
                    epsilon_root,
                    max_iter,
                    mode,
                    initial_step=initial_step
                )
                
                # Kết luận về khoảng phân ly
                print("\n=== Kết luận về khoảng phân ly nghiệm ===")
                if not isolation_data:
                    print("Không tìm thấy khoảng phân ly nghiệm nào trong khoảng đã cho.")
                else:
                    print(f"Tìm thấy {len(isolation_data)} khoảng phân ly nghiệm:")
                    print("\nBảng giá trị trung gian cho việc tìm khoảng phân ly:")
                    headers = ["Lần lặp", "x1", "x2", "Mid", "p(x1)", "p(Mid)", "p(x2)", "Độ dài"]
                    for i, interval_data in enumerate(isolation_data):
                        print(f"\nKhoảng phân ly {i+1}:")
                        PolynomialIO.print_table(headers, interval_data)
                        # Lấy khoảng cuối cùng từ dữ liệu
                        last_data = interval_data[-1]
                        x1, x2 = last_data[1], last_data[2]
                        print(f"Kết luận: Tồn tại duy nhất một nghiệm trong khoảng [{x1:.6f}, {x2:.6f}]")
                
                print("\n=== Kết quả tìm nghiệm chính xác ===")
                if not roots:
                    print("Không tìm được nghiệm thực nào trong các khoảng phân ly.")
                else:
                    print("\nBảng giá trị trung gian cho việc tìm nghiệm chính xác:")
                    headers = ["Lần lặp", "a", "b", "f(a)*f(Mid)", "Mid", "Sai số"]
                    for i, (root, intermediate_data) in enumerate(all_intermediate_data):
                        print(f"\nNghiệm {i+1}: {root:.6f}")
                        PolynomialIO.print_table(headers, intermediate_data)
                    
                    print("\n=== Đầu ra ===")
                    result_data = []
                    for i, root in enumerate(roots):
                        result_data.append([i+1, root])
                    headers = ["Nghiệm", "Giá trị"]
                    PolynomialIO.print_table(headers, result_data)
                    
                    if roots:
                        # Vẽ đồ thị
                        PolynomialSolver(coeffs).plot_results(a, b, mode_choice, roots=roots)
            
            elif mode_choice in [4, 5, 6, 7]:  # Mode 4, 5, 6, 7: Tìm max, min với sai số
                print("\n=== Phương pháp sử dụng ===")
                print(f"Phương pháp chia đôi - Hậu nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'} để tìm cực trị")
                
                min_val, max_val, intermediate_values, intermediate_data_intervals, all_intermediate_data, roots, roots_intermediate_data = PolynomialSolver(coeffs).find_extrema(a, b, epsilon, max_iter, mode, use_absolute)
                
                if not intermediate_data_intervals and not all_intermediate_data:
                    print("\nKhông tìm thấy điểm cực trị nào trong khoảng đã cho.")
                else:
                    if intermediate_data_intervals:
                        print("\nBảng giá trị trung gian (Tìm khoảng phân ly nghiệm của f'(x)):")
                        headers = ["Lần lặp", "x1", "x2", "Mid", "p(x1)", "p(Mid)", "p(x2)", "Độ dài"]
                        for i, interval_data in enumerate(intermediate_data_intervals):
                            print(f"\nKhoảng cách ly {i+1}:")
                            PolynomialIO.print_table(headers, interval_data)
                    
                    if all_intermediate_data:
                        print("\nBảng giá trị trung gian (Tìm điểm cực trị):")
                        headers = ["Lần lặp", "a", "b", "f(a)*f(Mid)", "Mid", "Sai số"]
                        for i, (point, intermediate_data) in enumerate(all_intermediate_data):
                            print(f"\nĐiểm cực trị {i+1}: {point:.6f}")
                            PolynomialIO.print_table(headers, intermediate_data)
                    
                    if use_absolute and roots:
                        print("\nBảng giá trị trung gian (Tìm nghiệm của f(x) = 0):")
                        headers = ["Lần lặp", "a", "b", "f(a)*f(Mid)", "Mid", "Sai số"]
                        for i, (root, intermediate_data) in enumerate(roots_intermediate_data):
                            print(f"\nNghiệm {i+1}: {root:.6f}")
                            PolynomialIO.print_table(headers, intermediate_data)
                    
                    print("\nBảng giá trị tại các điểm:")
                    headers = ["x", "f(x)", "|f(x)|" if use_absolute else "f(x)"]
                    PolynomialIO.print_table(headers, intermediate_values)
                    
                    print("\n=== Đầu ra ===")
                    result_data = [
                        ["Giá trị nhỏ nhất", min_val],
                        ["Giá trị lớn nhất", max_val]
                    ]
                    headers = ["Hàm", f"{'|f(x)|' if use_absolute else 'f(x)'}"]
                    PolynomialIO.print_table(headers, result_data)
                    
                    if intermediate_data_intervals or all_intermediate_data:
                        # Vẽ đồ thị
                        critical_points = [point for point, _ in all_intermediate_data]
                        PolynomialSolver(coeffs).plot_results(a, b, mode_choice, critical_points=critical_points, min_val=min_val, max_val=max_val)
        
        except ValueError as e:
            print(f"\nLỗi trong quá trình tính toán: {e}")
        except Exception as e:
            print(f"\nLỗi không xác định: {e}")
    
    except KeyboardInterrupt:
        print("\nChương trình đã bị dừng bởi người dùng.")
    except Exception as e:
        print(f"\nLỗi không xác định: {e}")

if __name__ == "__main__":
    main()