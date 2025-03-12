# phương pháp chia đôi
# nhập hàm f(x) => chọn mode ở line 160 (công thức sai số + loại sai số (tương đối/tuyệt đối))
# công thức hậu nghiệm: điều kiện dừng xác định bởi sai số cho trước (biến error)
# công thức tiên nghiệm: điều kiện dừng xác định bởi số lần lặp (biến iterations)
# NẾU ĐỀ BÀI CHO SỐ LẦN LẶP THÌ SỬA LINE 88
# NẾU ĐỀ BÀI Y/C 8 CHỮ SỐ SAU DẤU PHẨY thì epsilon = 0,5*10^-8 => ĐIỀN VÀO ERROR Ở LINE 160
# số lần lặp n <= log_2[(b-a)/epsilon]
import math

def f(x):
    return x**5 - 7  

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
    return (x > 0) - (x < 0)

# Hàm tìm nghiệm bằng phương pháp chia đôi
def bisection_method(f, a, b, error=1e-6, mode=0):
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
    prev_x = None  # Lưu giá trị nghiệm của lần lặp trước
    delta = float('inf')  # Sai số ban đầu lớn để đảm bảo vòng lặp chạy ít nhất một lần
    sfm = sign(f(x_next))  # Dấu của f(x_next)
    sfa = sign(f(a))  # Dấu của f(a)
    
    # Biến đánh dấu nghiệm đã được tìm thấy
    solution_found = False
    extra_iterations = 0
    max_extra_iterations = 2  # Số vòng lặp thêm sau khi tìm được nghiệm
    
    # Biến lưu trữ thông tin nghiệm thỏa mãn đầu tiên
    solution_iter = 0
    solution_x = None
    solution_delta = None
    
    # Xác định số lần lặp tối đa hoặc điều kiện dừng
    if method == "a_priori":                                          # Tiên nghiệm: chỉ dừng theo số lần lặp
        max_iterations = math.ceil(math.log2((b - a) / error)) + 1    # Dùng math.ceil để làm tròn số vòng lặp lên và cộng thêm 1 lần lặp sau kết quả 
#        max_iterations = 17                                          # Nếu đề bài yêu cầu 18 lần lặp thì nhập vào 19 và lấy kết quả ở lần 18
    else:
        max_iterations = float('inf')  # Hậu nghiệm: chỉ dừng khi sai số đạt yêu cầu
    
    # In thông tin phương pháp đang sử dụng
    print(f"Chia đôi - {'Tiên nghiệm' if method == 'a_priori' else 'Hậu nghiệm'} - {'Sai số tương đối' if error_type == 'relative' else 'Sai số tuyệt đối'}")
    
    # In tiêu đề bảng kết quả
    print(f"{'n':<5}{'a_n':<15}{'b_n':<15}{'f(a_n)*f(x_n+1)':<20}{'x_n+1':<15}{'delta rela' if error_type == 'relative' else 'delta abs'}")
    
    # Vòng lặp chính
    while ((method == "a_priori" and count < max_iterations) or 
          (method == "a_posteriori" and (not solution_found or extra_iterations < max_extra_iterations))):
        prev_x = x_next  # Lưu giá trị x_n+1 của lần lặp trước
        
        # Hiển thị dấu + hoặc - thay vì giá trị số
        sign_product = "+" if (sfm * sfa) > 0 else "-"
        
        # In kết quả mỗi bước lặp
        print(f"{count:<5}{a:<15.10f}{b:<15.10f}{sign_product:<20}{x_next:<15.10f}{delta:.6e}")
        
        # Nếu tìm được nghiệm chính xác (f(x_next) = 0)
        if sfm == 0:
            if not solution_found:
                solution_found = True
                solution_iter = count + 1    # do tại lần lặp thứ n có được x_n+1
                solution_x = x_next
                solution_delta = 0  # Đây là nghiệm chính xác
            extra_iterations += 1
        
        # Kiểm tra điều kiện hội tụ cho phương pháp hậu nghiệm
        if method == "a_posteriori" and delta < error and not solution_found:
            solution_found = True
            solution_iter = count + 1    # do tại lần lặp thứ n có được x_n+1
            solution_x = x_next
            solution_delta = delta
        
        # Nếu đã tìm thấy nghiệm, đếm số vòng lặp bổ sung
        if solution_found:
            extra_iterations += 1
        
        # Cập nhật khoảng chứa nghiệm
        if sfm != sfa:
            b = x_next  # Nghiệm nằm trong [a, x_next]
        else:
            a = x_next  # Nghiệm nằm trong [x_next, b]
            sfa = sign(f(a))
        
        # Tính giá trị mới
        x_next = (a + b) * 0.5
        sfm = sign(f(x_next))
        
        # Tính sai số
        if prev_x is not None:
            if error_type == "absolute":
                delta = (b - a)
            else:  # relative error
                delta = abs(x_next - prev_x) / abs(x_next) if x_next != 0 else abs(x_next - prev_x)
        
        count += 1  # Tăng số lần lặp
    
    # In kết quả cuối cùng
        print(f"Nghiệm gần đúng tại n = {solution_iter - 1}: x_{solution_iter} = {solution_x:.10f} với sai số {'tương đối' if error_type == 'relative' else 'tuyệt đối'} là {solution_delta:.6e} < {error} thỏa mãn")
    else:
        print(f"Nghiệm gần đúng tại : x_{count} = {x_next:.10f}")
    
    return x_next if not solution_found else solution_x  # Trả về nghiệm gần đúng

# Hàm main để chạy chương trình
if __name__ == "__main__":
    try:
        result = bisection_method(f, 1, 2, 1e-7, mode = 0)     # nhập khoảng cách li a, b, epsilon và chọn mode
    except ValueError as e:
        print(f"Error: {e}")

# MODE:
# 0 --- Tiên nghiệm + Sai số tương đối
# 1 --- Tiên nghiệm + Sai số tuyệt đối
# 2 --- Hậu nghiệm + Sai số tương đối
# 3 --- Hậu nghiệm + Sai số tuyệt đối