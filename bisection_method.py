# phương pháp chia đôi
# nhập hàm f(x) => chọn mode ở line 150 (công thức sai số + loại sai số (tương đối/tuyệt đối))
# công thức hậu nghiệm: điều kiện dừng xác định bởi sai số cho trước (biến error)
# công thức tiên nghiệm: điều kiện dừng xác định bởi số lần lặp (biến iterations)
# NẾU ĐỀ BÀI CHO SỐ LẦN LẶP n THÌ TÍNH ERROR BẰNG CÔNG THỨC (b-a)/2^n 
# NẾU ĐỀ BÀI Y/C 8 CHỮ SỐ SAU DẤU PHẨY thì epsilon = 0,5*10^-8 => ĐIỀN VÀO ERROR Ở LINE 150
# số lần lặp n <= log_2[(b-a)/epsilon]
#import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return np.exp(x) + np.cos(2*x)  

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
# --------------------------------------------------------------------------

# Hàm xác định dấu của một số (-1 nếu âm, 1 nếu dương, 0 nếu bằng 0)
def sign(x):
    return np.sign(x)       # là (x > 0) - (x < 0) nếu sử dụng math

# Hàm tìm nghiệm bằng phương pháp chia đôi
# các giá trị error và mode ở đây được set mặc định, khi gọi hàm sẽ được ghi đè
def bisection_method(f, a, b, error = 1e-6, mode = 0):
    # Ánh xạ giá trị mode với phương thức và loại sai số
    mode_map = {
        0: ("a_priori", "relative"),   # Tiên nghiệm + Sai số tương đối
        1: ("a_priori", "absolute"),   # Tiên nghiệm + Sai số tuyệt đối
        2: ("a_posteriori", "relative"), # Hậu nghiệm + Sai số tương đối
        3: ("a_posteriori", "absolute")  # Hậu nghiệm + Sai số tuyệt đối
    }
    
    # Kiểm tra mode có hợp lệ không
    if mode not in mode_map:
        raise ValueError("mode phải là 0, 1, 2 hoặc 3.")
    
    # Lấy phương thức và loại sai số tương ứng
    method, error_type = mode_map[mode]
    
    # Kiểm tra điều kiện ban đầu
    if f(a) * f(b) >= 0:
        raise ValueError("Không thể dùng phương pháp chia đôi do f(a)*f(b) >= 0.")
    if a >= b:
        raise ValueError("Hãy nhập a < b.")
    if error <= 0:
        raise ValueError("Sai số phải là số dương.")
    
    # Khởi tạo biến
    count = 0  # Đếm số lần lặp
    x_next = (a + b) * 0.5  # Điểm giữa khoảng hiện tại
    prev_x = float('inf')  # Lưu giá trị nghiệm của lần lặp trước (giá trị mặc định ban đầu là lớn)
    delta = float('inf')  # Sai số ban đầu lớn để đảm bảo vòng lặp chạy ít nhất một lần
    sfm = sign(f(x_next))  # Dấu của f(x_next)
    sfa = sign(f(a))  # Dấu của f(a)               
    
    # Biến lưu trữ thông tin nghiệm thỏa mãn đầu tiên
    solution_found = False         # đánh dấu nghiệm đã được tìm thấy
    solution_iter = 0              # lưu lại n tại kết quả
    solution_x = None              # lưu lại x_n+1 tại kết quả
    solution_delta = None          # lưu lại delta tại kết quả
    
    # In thông tin phương pháp đang sử dụng
    print(f"Chia đôi - {'Tiên nghiệm' if method == 'a_priori' else 'Hậu nghiệm'} - {'Sai số tương đối' if error_type == 'relative' else 'Sai số tuyệt đối'}")
    
    # In tiêu đề bảng
    print(f"{'n':<5} {'a_n':<15} {'b_n':<15} {'f(a_n)*f(x_n+1)':<20} {'x_n+1':<15} {'delta rela' if error_type == 'relative' else 'delta abs'}")
    
    # Vòng lặp chính
    while True:
        if solution_found:
            # Hiển thị kết quả khi đạt được độ chính xác yêu cầu
# dòng này hiện đang hiển thị bị lệch sang kết quả khác đằng sau kết quả đúng            
#            print(f"Nghiệm gần đúng tại n = {solution_iter}: x_{solution_iter+1} = {prev_x:.10f} với sai số {'tương đối' if error_type == 'relative' else 'tuyệt đối'} là {solution_delta:.6e} < {error} thỏa mãn")
            break
        
        # Nếu tìm được nghiệm chính xác (f(x_next) = 0)
        if sfm == 0:
            solution_found = True
            solution_iter = count  
            solution_x = x_next
            solution_delta = 0
            
            # Hiển thị khi tìm thấy nghiệm chính xác
            print(f"Tìm thấy nghiệm chính xác tại n = {count}: x_{count+1} = {x_next:.10f} với f(x_{count+1}) = 0")
            break  # Dừng ngay lập tức khi tìm thấy nghiệm chính xác
        
        if delta < error:
            solution_found = True
            solution_iter = count
            solution_x = x_next
            solution_delta = delta
            
        if method == "a_priori":                                  # tiên nghiệm
            if error_type == "absolute":    # sai số tuyệt đối
                delta = abs(b - a)
            else:                           # sai số tương đối
                delta = abs(b - a) / abs(x_next)
                
        elif method == "a_posteriori":                            # hậu nghiệm
            if error_type == "absolute":    # sai số tuyệt đối
                delta = abs(x_next - prev_x)
            else:                           # sai số tương đối
                delta = abs(x_next - prev_x) / abs(x_next) if x_next != 0 else abs(x_next - prev_x)
                
        
        sign_product = "+" if (sfm * sfa) > 0 else "-"     # Hiển thị dấu + hoặc - thay vì giá trị số
        print(f"{count:<5} {a:<15.10f} {b:<15.10f} {sign_product:<20} {x_next:<15.10f} {delta:.6e}")   # in vòng lặp
            
        # Lưu giá trị x_n+1 của lần lặp trước
        prev_x = x_next
        
        # Cập nhật khoảng chứa nghiệm
        if sfm != sfa:
            b = x_next  # Nghiệm nằm trong [a, x_next]
        else:
            a = x_next  # Nghiệm nằm trong [x_next, b]
            sfa = sign(f(a))
            
        # Tính giá trị mới
        x_next = (a + b) * 0.5
        sfm = sign(f(x_next))
        count += 1  # Tăng số lần lặp 
                              
    # Trả về nghiệm tìm được
    return solution_x

# Hàm main để chạy chương trình
if __name__ == "__main__":
    try:
        result = bisection_method(f, a = -3, b = -2, error = 1e-8, mode = 2)     # nhập khoảng cách li a, b, epsilon và chọn mode
    except ValueError as e:
        print(f"Error: {e}")

    # Tạo mảng giá trị x từ -3 đến 3 với 400 điểm
    x_values = np.linspace(-3, 3, 400)
    y_values = f(x_values)  # Tính giá trị của hàm tại các điểm x

    # Vẽ đồ thị
    plt.plot(x_values, y_values, label=r"$f(x) = e^x + cos(2x)$", color="b")
    plt.axhline(0, color='black', linewidth=0.5)  # Trục Ox
    plt.axvline(0, color='black', linewidth=0.5)  # Trục Oy
    plt.grid(True, linestyle="--", alpha=0.6)  # Lưới

    # Hiển thị nhãn trục
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Đồ thị hàm số f(x)")
    plt.legend()

    # Hiển thị đồ thị
    plt.show()


# MODE:
# 0 --- Tiên nghiệm + Sai số tương đối
# 1 --- Tiên nghiệm + Sai số tuyệt đối
# 2 --- Hậu nghiệm + Sai số tương đối
# 3 --- Hậu nghiệm + Sai số tuyệt đối