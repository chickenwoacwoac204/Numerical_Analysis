#include <iostream>
#include <cmath> //log(x)
#include <iomanip> //setprecision - (do chinh xac sau dau phay)
using namespace std;

template <typename type> 
type f(type x) {
	return log(x)-1;
}

template <typename type>  
int sign(type x) {
	return (x>type(0))-(x<type(0)); //sign = (x > 0.0) ? 1 : (x < 0.0) ? -1 : 0;
}

template <typename type> 
type bisectionMethod(type (*f)(type x), type a, type b, type error, int iterations = 0) { //iterations = n+1
	int count = 0;
	type middle = (a+b)*0.5;
	type delta = b-a;
	int sfm = sign(f(middle));
	int sfa = sign(f(a));
//in ra cac gia tri ban dau (n=0)
	cout << "n = " << count
		 << "  a = " << setprecision(12) << a 
		 << "  b = " << setprecision(12) << b
		 << "  f(middle)*f(a) = " << sfm*sfa 
		 << "  x = " << setprecision(12) << middle
		 << "  delta = " << setprecision(12) << delta << endl;
		 
	while(delta >= error || count < iterations) {
//Gia su den lan thu n, sai so da thoa man
//Dieu kien "delta >= error"  => n
//Dieu kien "count < iterations" => n+1
		if (sfm==0) {
		return middle;
		}
		if (sfm!=sfa) {
			b=middle;
		}
		else {
			a=middle;
			sfa = sign(f(a));
		}
		middle = (a+b)*0.5;
		sfm = sign(f(middle));
		delta = b-a;
		++count;
		
		cout << "n = " << count
		 << "  a = " << setprecision(12) << a 
		 << "  b = " << setprecision(12) << b
		 << "  f(middle)*f(a) = " << sfm*sfa 
		 << "  x = " << setprecision(12) << middle
		 << "  delta = " << setprecision(12) << delta << endl;
	}
	return 0; 
}

int main() {
	long double e = bisectionMethod(f, 2.0, 3.0, 0.5*(1e-8), 30);
	return 0; 
}