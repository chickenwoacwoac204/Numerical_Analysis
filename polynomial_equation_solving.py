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

    def find_isolated_intervals(self, epsilon, max_iter, mode="absolute", initial_step=None, a=None, b=None):
        """
        Tìm khoảng phân ly nghiệm với kiểm tra điểm đặc biệt
        """
        if a is None or b is None:
            R = self.cauchy_bound()
            left, right = -R, R
        else:
            left, right = a, b
        
        intervals = []
        intermediate_data = []
        
        # Tạo lưới điểm mịn hơn
        step = initial_step if initial_step is not None else epsilon
        min_step = min((right - left)/200, epsilon/10)  # Tăng độ mịn của lưới
        step = min(step, min_step)
        
        # Tạo lưới điểm cơ bản
        points = []
        x = left
        while x <= right:
            points.append(x)
            x += step
        
        # Thêm điểm cuối nếu cần
        if points[-1] < right:
            points.append(right)
        
        # Thêm các điểm đặc biệt
        special_points = set([left, right])
        
        # Tính các hệ số đạo hàm
        deriv1_coeffs = self.derivative()
        if len(deriv1_coeffs) > 1:  # Nếu đạo hàm không phải hằng số
            deriv2_coeffs = PolynomialSolver(deriv1_coeffs).derivative()
            if len(deriv2_coeffs) > 1:  # Nếu đạo hàm bậc 2 không phải hằng số
                # Thêm các điểm chia đều trong khoảng để kiểm tra dấu
                num_check_points = 50  # Số điểm kiểm tra
                check_points = np.linspace(left, right, num_check_points)
                
                # Kiểm tra đổi dấu của đạo hàm bậc 2
                for i in range(len(check_points) - 1):
                    x1, x2 = check_points[i], check_points[i + 1]
                    d2_x1 = PolynomialSolver(deriv2_coeffs).evaluate_polynomial(x1)
                    d2_x2 = PolynomialSolver(deriv2_coeffs).evaluate_polynomial(x2)
                    if d2_x1 * d2_x2 <= 0:  # Có điểm uốn
                        mid = (x1 + x2) / 2
                        special_points.add(mid)
        
        # Thêm các điểm đặc biệt vào lưới điểm
        points.extend(special_points)
        points = sorted(list(set(points)))  # Loại bỏ trùng lặp và sắp xếp
        
        # Tìm khoảng phân ly
        for i in range(len(points) - 1):
            x1, x2 = points[i], points[i + 1]
            p1 = self.evaluate_polynomial(x1)
            p2 = self.evaluate_polynomial(x2)
            
            if p1 * p2 < 0:  # Dấu hiệu có nghiệm
                interval_data = []
                iteration = 0
                current_x1, current_x2 = x1, x2
                current_p1, current_p2 = p1, p2
                
                while iteration < max_iter:
                    length = current_x2 - current_x1
                    mid = (current_x1 + current_x2) / 2
                    p_mid = self.evaluate_polynomial(mid)
                    interval_data.append([iteration + 1, current_x1, current_x2, mid, 
                                       current_p1, p_mid, current_p2, length])
                    
                    # Kiểm tra điều kiện dừng
                    if mode == "absolute":
                        if length < epsilon:
                            intervals.append((current_x1, current_x2))
                            intermediate_data.append(interval_data)
                            break
                        else:  # relative
                            if length / abs(mid) < epsilon and mid != 0:
                                intervals.append((current_x1, current_x2))
                                intermediate_data.append(interval_data)
                                break
                    
                    # Thu hẹp khoảng
                    if current_p1 * p_mid < 0:
                        current_x2 = mid
                        current_p2 = p_mid
                    else:
                        current_x1 = mid
                        current_p1 = p_mid
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

    def find_extrema(self, a, b, epsilon_isolation, epsilon_root, max_iter, mode="absolute", use_absolute=False):
        """
        Tìm cực trị của hàm số trong khoảng [a, b] bằng cách tìm nghiệm của đạo hàm
        
        Args:
            a, b: Khoảng tìm cực trị
            epsilon_isolation: Sai số cho việc tìm khoảng phân ly
            epsilon_root: Sai số cho việc tìm nghiệm chính xác
            max_iter: Số lần lặp tối đa
            mode: "absolute" hoặc "relative" - chế độ tính sai số
            use_absolute: True nếu tìm cực trị của |f(x)|, False nếu tìm của f(x)
        """
        # Kiểm tra trường hợp đặc biệt: đa thức bậc 0 hoặc 1
        if len(self.coeffs) <= 2:
            f_a = self.evaluate_polynomial(a)
            f_b = self.evaluate_polynomial(b)
            if use_absolute:
                f_a, f_b = abs(f_a), abs(f_b)
            return min(f_a, f_b), max(f_a, f_b), [[a, f_a, abs(f_a) if use_absolute else f_a], 
                                                 [b, f_b, abs(f_b) if use_absolute else f_b]], [], [], [], []

        # Tính đạo hàm
        deriv_coeffs = self.derivative()
        deriv_solver = PolynomialSolver(deriv_coeffs)
        
        # Tìm nghiệm của đạo hàm (điểm cực trị) bằng mode 2-3
        critical_points, critical_data, isolation_data = deriv_solver.find_real_roots(
            epsilon_isolation,
            epsilon_root,
            max_iter,
            mode,
            initial_step=min((b-a)/10, epsilon_isolation)
        )
        
        # Thêm điểm đầu và cuối khoảng
        points = [a] + critical_points + [b]
        points = sorted(list(set(points)))  # Loại bỏ điểm trùng lặp
        
        # Tính giá trị tại các điểm
        values = []
        for x in points:
            f_x = self.evaluate_polynomial(x)
            value = abs(f_x) if use_absolute else f_x
            values.append([x, f_x, value])
        
        if not values:
            return None, None, [], [], [], [], []
        
        # Tìm max, min
        min_val = min(v[2] for v in values)
        max_val = max(v[2] for v in values)
        
        return min_val, max_val, values, isolation_data, critical_data, [], []

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
        
        # Vẽ |f(x)| cho mode 6-7
        if mode_choice in [6, 7]:
            y_abs = np.abs(y)
            plt.plot(x, y_abs, 'r-', label='|f(x)|')
        
        # Vẽ trục tọa độ
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Vẽ khoảng [a, b]
        plt.axvline(x=a, color='g', linestyle='--', alpha=0.5, label='x = a')
        plt.axvline(x=b, color='g', linestyle='--', alpha=0.5, label='x = b')
        
        # Vẽ các điểm đặc biệt
        if mode_choice in [0, 1, 2, 3]:  # Nghiệm
            if roots:
                for root in roots:
                    plt.plot([root], [0], 'ro', label='Nghiệm')
        elif mode_choice in [4, 5]:  # Cực trị của f(x)
            if critical_points:
                for point in critical_points:
                    y_val = self.evaluate_polynomial(point)
                    plt.plot([point], [y_val], 'go', label='Điểm cực trị')
            if min_val is not None and max_val is not None:
                plt.axhline(y=min_val, color='r', linestyle='--', alpha=0.5, label='Min')
                plt.axhline(y=max_val, color='r', linestyle='--', alpha=0.5, label='Max')
        else:  # Mode 6-7: Cực trị của |f(x)|
            if critical_points:
                for point in critical_points:
                    y_val = self.evaluate_polynomial(point)
                    plt.plot([point], [y_val], 'go', label='f(x)')
                    plt.plot([point], [abs(y_val)], 'ro', label='|f(x)|')
            if min_val is not None and max_val is not None:
                plt.axhline(y=min_val, color='r', linestyle='--', alpha=0.5, label='Min |f(x)|')
                plt.axhline(y=max_val, color='r', linestyle='--', alpha=0.5, label='Max |f(x)|')
        
        # Cấu hình đồ thị
        plt.grid(True, alpha=0.3)
        plt.title('Đồ thị hàm số' + (' và trị tuyệt đối' if mode_choice in [6, 7] else ''))
        plt.xlabel('x')
        plt.ylabel('y')
        
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

def print_root_finding_results(roots, all_intermediate_data, isolation_data, epsilon_isolation, epsilon_root, mode, function_name="f(x)"):
    """
    In kết quả trung gian của quá trình tìm nghiệm
    Args:
        roots: danh sách nghiệm
        all_intermediate_data: dữ liệu trung gian tìm nghiệm
        isolation_data: dữ liệu trung gian tìm khoảng phân ly
        epsilon_isolation: sai số khoảng phân ly
        epsilon_root: sai số nghiệm
        mode: chế độ tính sai số
        function_name: tên hàm số (f(x) hoặc f'(x))
    """
    # In kết quả tìm khoảng phân ly
    print(f"\n=== Kết luận về khoảng phân ly nghiệm của {function_name} ===")
    if not isolation_data:
        print(f"Không tìm thấy khoảng phân ly nghiệm nào của {function_name} trong khoảng đã cho.")
    else:
        print(f"Tìm thấy {len(isolation_data)} khoảng phân ly nghiệm:")
        headers = ["Lần lặp", "x1", "x2", "Mid", f"{function_name}(x1)", 
                  f"{function_name}(Mid)", f"{function_name}(x2)", "Độ dài"]
        for i, interval_data in enumerate(isolation_data):
            print(f"\nKhoảng phân ly {i+1}:")
            PolynomialIO.print_table(headers, interval_data)
            last_data = interval_data[-1]
            x1, x2 = last_data[1], last_data[2]
            length = last_data[-1]
            print(f"Kết luận: Tồn tại duy nhất một nghiệm của {function_name} trong [{x1:.6f}, {x2:.6f}]")
            print(f"Sai số tìm khoảng phân ly: {length:.6e} < {epsilon_isolation:.6e}")

    # In kết quả tìm nghiệm chính xác
    print(f"\n=== Kết quả tìm nghiệm chính xác của {function_name} ===")
    if not roots:
        print(f"Không tìm được nghiệm thực nào của {function_name} trong các khoảng phân ly.")
    else:
        headers = ["Lần lặp", "a", "b", f"{function_name}(a)*{function_name}(Mid)", "Mid", "Sai số"]
        for i, (root, intermediate_data) in enumerate(all_intermediate_data):
            print(f"\nNghiệm {i+1}: {root:.6f}")
            PolynomialIO.print_table(headers, intermediate_data)
            last_data = intermediate_data[-1]
            error = last_data[-1]
            print(f"Sai số {'tương đối' if mode == 'relative' else 'tuyệt đối'}: {error:.6e} < {epsilon_root:.6e}")

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
        
        if mode_choice in [2, 3, 4, 5]:  # Mode 2-3-4-5
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

        elif mode_choice in [6, 7]:  # Mode 6, 7: Tìm max, min của |f(x)|
            print("\nNhập sai số cho việc tìm khoảng phân ly của f'(x) (epsilon_0):")
            epsilon_isolation = float(input())
            while epsilon_isolation <= 0:
                print("Sai số phải dương. Vui lòng nhập lại:")
                epsilon_isolation = float(input())
                
            print("\nNhập sai số cho việc tìm nghiệm của f'(x) (epsilon_1):")
            epsilon_root = float(input())
            while epsilon_root <= 0:
                print("Sai số phải dương. Vui lòng nhập lại:")
                epsilon_root = float(input())
                
            print("\nNhập sai số cho việc tìm nghiệm của f(x) (epsilon_2):")
            epsilon_zero = float(input())
            while epsilon_zero <= 0:
                print("Sai số phải dương. Vui lòng nhập lại:")
                epsilon_zero = float(input())

        print("\nNhập số lần lặp tối đa:")
        max_iter = int(input())
        while max_iter <= 0:
            print("Số lần lặp phải dương. Vui lòng nhập lại:")
            max_iter = int(input())
    
        # In đầu vào
        print("\n=== Đầu vào ===")
        print(f"Đa thức: f(x) = {PolynomialIO.print_polynomial(coeffs)}")
        print(f"Khoảng [a, b]: [{a}, {b}]")
        if mode_choice in [2, 3, 4, 5]:  # Mode 2-3-4-5
            print(f"Sai số tìm khoảng phân ly (epsilon_0): {epsilon_isolation}")
            print(f"Sai số tìm nghiệm chính xác (epsilon_1): {epsilon_root}")
        elif mode_choice in [6, 7]:  # Mode 6-7
            print(f"Sai số tìm khoảng phân ly của f'(x) (epsilon_0): {epsilon_isolation}")
            print(f"Sai số tìm nghiệm của f'(x) (epsilon_1): {epsilon_root}")
            print(f"Sai số tìm nghiệm của f(x) (epsilon_2): {epsilon_zero}")
        print(f"Số lần lặp tối đa: {max_iter}")
    
        # Xác định mode
        mode = "absolute" if mode_choice in [0, 3, 5, 7] else "relative"
        use_absolute = mode_choice in [6, 7]
        
        try:
            # Xử lý từng bài toán
            if mode_choice in [0, 1]:  # Mode 0 và 1: Tìm khoảng phân ly nghiệm
                print("\n=== Phương pháp sử dụng ===")
                print(f"Tìm khoảng phân ly nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'}")
                
                intervals, isolation_data = PolynomialSolver(coeffs).find_isolated_intervals(
                    epsilon_isolation, 
                    max_iter, 
                    mode,
                    initial_step=epsilon_isolation
                )
                
                # Kết luận về khoảng phân ly
                print_root_finding_results(intervals, isolation_data, isolation_data,
                                         epsilon_isolation, epsilon_isolation, mode, "f(x)")
    
            elif mode_choice in [2, 3]:  # Mode 2 và 3: Tìm nghiệm thực
                print("\n=== Phương pháp sử dụng ===")
                print(f"Phương pháp chia đôi - Hậu nghiệm - Sai số {'tuyệt đối' if mode == 'absolute' else 'tương đối'}")
                
                roots, all_intermediate_data, isolation_data = PolynomialSolver(coeffs).find_real_roots(
                    epsilon_isolation,
                    epsilon_root,
                    max_iter,
                    mode,
                    initial_step=epsilon_isolation
                )
                
                # Kết luận về khoảng phân ly
                print_root_finding_results(roots, all_intermediate_data, isolation_data,
                                         epsilon_isolation, epsilon_root, mode, "f(x)")
            
            elif mode_choice in [4, 5]:  # Mode 4, 5: Tìm max, min của f(x)
                print("\n=== Phương pháp sử dụng ===")
                print(f"Tìm nghiệm của đạo hàm f'(x) = 0 bằng phương pháp chia đôi")
                print(f"Sai số {'tương đối' if mode == 'relative' else 'tuyệt đối'}")
                
                # Tạo solver cho đa thức gốc
                solver = PolynomialSolver(coeffs)
                # Tạo solver cho đạo hàm
                deriv_solver = PolynomialSolver(solver.derivative())
                
                # Tìm cực trị của f(x)
                critical_points, critical_data, critical_isolation_data = deriv_solver.find_real_roots(
                    epsilon_isolation,
                    epsilon_root,
                    max_iter,
                    mode,
                    initial_step=min((b-a)/10, epsilon_isolation)
                )
                
                # In kết quả trung gian của việc tìm nghiệm f'(x)
                print_root_finding_results(critical_points, critical_data, critical_isolation_data,
                                         epsilon_isolation, epsilon_root, mode, "f'(x)")
                
                # Tính giá trị tại các điểm cực trị và điểm đầu mút
                points = [a] + (critical_points if critical_points else []) + [b]
                values = []
                for x in points:
                    f_x = solver.evaluate_polynomial(x)
                    values.append([x, f_x, f_x])  # Giá trị thứ 3 để tương thích với mode 6-7
                
                # Tìm max, min
                if not values:
                    print("\nKhông thể tìm được cực trị trong khoảng đã cho.")
                    return
                
                min_val = min(v[1] for v in values)
                max_val = max(v[1] for v in values)
                
                # In kết quả chi tiết
                print("\nChi tiết các điểm cực trị:")
                for x, f_val, _ in values:
                    if f_val == max_val:
                        point_type = "Cực đại"
                    elif f_val == min_val:
                        point_type = "Cực tiểu"
                    else:
                        continue
                    print(f"\nĐiểm {point_type}: x = {x:.6f}")
                    print(f"Giá trị f(x) = {f_val:.6f}")
                
                print(f"\nGiá trị tại điểm đầu mút:")
                print(f"f({a}) = {solver.evaluate_polynomial(a):.6f}")
                print(f"f({b}) = {solver.evaluate_polynomial(b):.6f}")
                
                print("\n=== Đầu ra ===")
                result_data = [
                    ["Giá trị nhỏ nhất của f(x)", min_val],
                    ["Giá trị lớn nhất của f(x)", max_val]
                ]
                headers = ["f(x)", "Giá trị"]
                PolynomialIO.print_table(headers, result_data)
                
                # Vẽ đồ thị
                PolynomialSolver(coeffs).plot_results(a, b, mode_choice, critical_points=critical_points, min_val=min_val, max_val=max_val)
            
            elif mode_choice in [6, 7]:  # Mode 6, 7: Tìm max, min của |f(x)|
                print("\n=== Phương pháp sử dụng ===")
                print(f"1. Tìm nghiệm của đạo hàm f'(x) = 0 để xác định cực trị của f(x)")
                print(f"2. Tìm nghiệm của f(x) = 0 để xác định cực trị của |f(x)|")
                print(f"Sai số {'tương đối' if mode == 'relative' else 'tuyệt đối'}")
                
                # Tạo solver cho đa thức gốc
                solver = PolynomialSolver(coeffs)
                # Tạo solver cho đạo hàm
                deriv_solver = PolynomialSolver(solver.derivative())
                
                # Tìm cực trị của f(x)
                critical_points, critical_data, critical_isolation_data = deriv_solver.find_real_roots(
                    epsilon_isolation,
                    epsilon_root,
                    max_iter,
                    mode,
                    initial_step=min((b-a)/10, epsilon_isolation)
                )
                
                # In kết quả trung gian của việc tìm nghiệm f'(x)
                print_root_finding_results(critical_points, critical_data, critical_isolation_data,
                                         epsilon_isolation, epsilon_root, mode, "f'(x)")
                
                # Tìm nghiệm của f(x) = 0
                roots, roots_data, roots_isolation_data = solver.find_real_roots(
                    epsilon_isolation,
                    epsilon_root,
                    max_iter,
                    mode,
                    initial_step=min((b-a)/10, epsilon_isolation)
                )
                
                # In kết quả trung gian của việc tìm nghiệm f(x)
                print_root_finding_results(roots, roots_data, roots_isolation_data,
                                         epsilon_isolation, epsilon_root, mode, "f(x)")
                
                # Tính giá trị tại các điểm đặc biệt
                all_points = [a] + (critical_points if critical_points else []) + (roots if roots else []) + [b]
                all_points = sorted(list(set(all_points)))  # Loại bỏ trùng lặp và sắp xếp
                
                values = []
                for x in all_points:
                    f_x = solver.evaluate_polynomial(x)
                    values.append([x, f_x, abs(f_x)])
                
                if not values:
                    print("\nKhông thể tìm được cực trị trong khoảng đã cho.")
                    return
                
                # Tìm max, min của |f(x)|
                abs_min = min(v[2] for v in values)
                abs_max = max(v[2] for v in values)
                
                # In kết quả chi tiết
                print("\nChi tiết các điểm cực trị của |f(x)|:")
                for x, f_val, abs_val in values:
                    if abs_val == abs_max:
                        point_type = "Cực đại"
                    elif abs_val == abs_min:
                        point_type = "Cực tiểu"
                    else:
                        continue
                    print(f"\nĐiểm {point_type}: x = {x:.6f}")
                    print(f"Giá trị f(x) = {f_val:.6f}")
                    print(f"Giá trị |f(x)| = {abs_val:.6f}")
                
                print(f"\nGiá trị tại điểm đầu mút:")
                print(f"|f({a})| = {abs(solver.evaluate_polynomial(a)):.6f}")
                print(f"|f({b})| = {abs(solver.evaluate_polynomial(b)):.6f}")
                
                print("\n=== Đầu ra ===")
                result_data = [
                    ["Giá trị nhỏ nhất của |f(x)|", abs_min],
                    ["Giá trị lớn nhất của |f(x)|", abs_max]
                ]
                headers = ["|f(x)|", "Giá trị"]
                PolynomialIO.print_table(headers, result_data)
                
                # Vẽ đồ thị
                PolynomialSolver(coeffs).plot_results(a, b, mode_choice, critical_points=all_points, min_val=abs_min, max_val=abs_max)
        
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