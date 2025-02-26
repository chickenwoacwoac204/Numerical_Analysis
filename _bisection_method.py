# phương pháp chia đôi (sai số tuyệt đối) (sai số tương đối = sai số tuyệt đối / x)
# công thức tiên nghiệm: điều kiện dừng xác định bởi số lần lặp (biến iterations)
# công thức hậu nghiệm: điều kiện dừng xác định bởi sai số cho trước (biến error)
import math

# ----- cần chỉnh sửa tại đây ----------------------------------------------
def f(x):
    return x**5 - 0.2*x + 15.0  # nhập hàm f(x)

# ------------- cách chọn hàm f(x) -----------------------------------------
# tính căn bậc m của n:  chọn f(x) = x**m - n            [do x^m = n]  
# tính logarit của m cơ số n:  chọn f(x) = m**x - n      [do log_m(n) = ln(n) / ln(m)]
# tính số e:  chọn f(x) = math.log(x) - 1                [do  ln(e) - 1 = 0]
# tính số pi:  chọn f(x) = math.tan(x/4) - 1             [do tan(x/4) = 1]

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

# Mặc định error = 1e-6 (dùng để tránh lỗi missing item, không cần sửa ở đây);
# iterations = 0 (đây là số vòng lặp tối đa, nếu đặt = 0 thì sẽ lặp cho đến khi đạt error nhỏ hơn error truyền vào)
def bisection_method(f, a, b, error=1e-6, max_iterations=0):
    # bắt lỗi
    if f(a) * f(b) > 0:
        raise ValueError("Không thể dùng phương pháp chia đôi do f(a) f(b) cùng dấu.")
    if a >= b:
        raise ValueError("Hãy nhập a (giá trị nhỏ hơn) trước, rồi nhập b (giá trị lớn hơn) sau.")
    if error <= 0:
        raise ValueError("Vui lòng nhập lại sai số, sai số phải là một giá trị dương.")
    
    # khởi tạo các giá trị
    count = 0                 # đếm số vòng lặp
    middle = (a + b) * 0.5    # điểm chính giữa a và b
    delta = b - a             # độ dài đoạn a b
    sfm = sign(f(middle))     # dấu của điểm chính giữa
    sfa = sign(f(a))          # dấu của f(a)
    
    # in vòng lặp đầu tiên (n=0) với các giá trị khởi tạo
    # x hiển thị tới 10 chữ số thập phân sau dấu phẩy (.10f), delta 6 chữ số thập phân sau dấu phẩy kèm thứ nguyên (.6e)
    print(f"n = {count}  a = {a:.10f}  b = {b:.10f}  f(middle)*f(a) = {sfm * sfa}  x = {middle:.10f}  delta = {delta:.6e}")
    
    # điều kiện dừng
    # vòng lặp dừng lặp khi: Sai số delta < error (đạt độ chính xác yêu cầu). Hoặc đã vượt quá số lần lặp iterations
    while delta >= error and (max_iterations == 0 or count < max_iterations):
        # Kiểm tra nghiệm chính xác
        if sfm == 0:          # Nếu f(middle)=0 thì middle là nghiệm chính xác => Kết thúc
            return middle
        # Cập nhật khoảng tìm nghiệm
        if sfm != sfa:        # Nếu f(middle) và f(a) khác dấu, thì nghiệm nằm trong khoảng [a, middle] => cập nhật b = middle
            b = middle
        else:                 # Nếu f(middle) và f(a) cùng dấu, nghiệm nằm trong [middle, b], cập nhật a = middle
            a = middle
            sfa = sign(f(a))
        
        # Tính giá trị mới
        middle = (a + b) * 0.5
        sfm = sign(f(middle))
        delta = b - a
        count += 1
        
        # in output sau mỗi bước lặp
        print(f"n = {count}  a = {a:.10f}  b = {b:.10f}  f(middle)*f(a) = {sfm * sfa}  x = {middle:.10f}  delta = {delta:.6e}")
    
    # Khi vòng lặp kết thúc, trả về nghiệm gần đúng
    return middle


# ----- cần chỉnh sửa tại đây --------------------------------------------------------------------------------
# hàm main bọc trong try-except: Nếu đầu vào không hợp lệ, in thông báo lỗi thay vì dừng chương trình đột ngột
if __name__ == "__main__":
    try:
        result = bisection_method(f, -2, -1, 1e-7, 30)   # 1e-7 = 1*10^(-7)
        print(f"Ket qua: {result:.10f}")
    # bắt lỗi: hàm nhận giá trị đầu vào không hợp lệ nhưng đúng về kiểu dữ liệu
    except ValueError as e:
        print(f"Error: {e}")
# ------------------------------------------------------------------------------------------------------------