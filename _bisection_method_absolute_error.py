# phương pháp chia đôi (sai số tuyệt đối => tiên nghiệm)
# công thức tiên nghiệm: điều kiện dừng xác định bởi số lần lặp (biến iterations) => ĐỀ BÀI CHO SỐ LẦN LẶP
# NẾU ĐỀ BÀI Y/C 8 CHỮ SỐ SAU DẤU PHẨY thì epsilon = 0,5*10^-8
# số lần lặp n <= log_2[(b-a)/epsilon]
import math

# ----- cần chỉnh sửa tại đây ----------------------------------------------
def f(x):
    return math.tan(x/4) -1  # nhập hàm f(x)
# --------------------------------------------------------------------------

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

# xác định dấu của một số, nếu là số âm thì trả về -1, dương thì 1, (= 0) thì return cũng là 0
def sign(x):
    return (x > 0) - (x < 0)

def bisection_method(f, a, b, max_iterations=0):
    # bắt lỗi
    if f(a) * f(b) >= 0:
        raise ValueError("Không thể dùng phương pháp chia đôi do f(a)*f(b) lớn hơn hoặc bằng 0.")
    if a >= b:
        raise ValueError("Hãy nhập a (giá trị nhỏ hơn) trước, rồi nhập b (giá trị lớn hơn) sau.")
    
    # khởi tạo các giá trị
    count = 0                                # đếm số vòng lặp
    x_next = (a + b) * 0.5                   # x_next = x_n+1 là điểm chính giữa a và b
    delta = b - a                            # sai số tuyệt đối
    sfm = sign(f(x_next))                    # dấu của điểm chính giữa
    sfa = sign(f(a))                         # dấu của f(a)
    
    # in vòng lặp đầu tiên (n=0) với các giá trị khởi tạo
    # x_n+1 hiển thị tới 10 chữ số thập phân sau dấu phẩy (.10f), delta 6 chữ số thập phân sau dấu phẩy kèm thứ nguyên (.6e)
    print(f"n = {count}  a_n = {a:.10f}  b_n = {b:.10f}  f(x_n+1)*f(a_n) = {(sfm * sfa):+}  x_n+1 = {x_next:.10f}  delta = {delta:.6e}")
    
    # điều kiện dừng
    # vòng lặp dừng lặp khi đã vượt quá số lần lặp iterations
    while max_iterations == 0 or count < max_iterations:
        # Kiểm tra nghiệm chính xác
        if sfm == 0:          # Nếu f(x_next)=0 thì x_next là nghiệm chính xác => Kết thúc
            return x_next
        # Cập nhật khoảng tìm nghiệm
        if sfm != sfa:        # Nếu f(x_next) và f(a) khác dấu, thì nghiệm nằm trong khoảng [a, x_next] => cập nhật b = x_next
            b = x_next
        else:                 # Nếu f(x_next) và f(a) cùng dấu, nghiệm nằm trong [x_next, b], cập nhật a = x_next
            a = x_next
            sfa = sign(f(a))
        
        # Tính giá trị mới
        x_next = (a + b) * 0.5
        sfm = sign(f(x_next))
        delta = b - a
        count += 1
        
        # in output sau mỗi bước lặp
        print(f"n = {count}  a_n = {a:.10f}  b_n = {b:.10f}  f(x_n+1)*f(a_n) = {(sfm * sfa):+}  x_n+1 = {x_next:.10f}  delta = {delta:.6e}")
    
    # Khi vòng lặp kết thúc, trả về nghiệm gần đúng
    return x_next


# ----- cần chỉnh sửa tại đây --------------------------------------------------------------------------------
# hàm main bọc trong try-except: Nếu đầu vào không hợp lệ, in thông báo lỗi thay vì dừng chương trình đột ngột
if __name__ == "__main__":
    try:
        result = bisection_method(f, 3, 3.3, 15)   # nhập khoảng cách ly nghiệm lần lượt là a, b và số lần lặp tối đa
        print(f"Nghiệm gần đúng: {result:.10f}")
    # bắt lỗi: hàm nhận giá trị đầu vào không hợp lệ nhưng đúng về kiểu dữ liệu
    except ValueError as e:
        print(f"Error: {e}")
# ------------------------------------------------------------------------------------------------------------