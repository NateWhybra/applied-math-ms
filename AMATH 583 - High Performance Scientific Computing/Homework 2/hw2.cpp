# include <iostream>
# include <vector>
# include <cmath>
# include <iomanip>
using namespace std;

// A container to hold a a float and a double value.
struct FloatDoublePair {
    float f = 0;
    double d = 0;
};

// A container to hold a an int and a long value.
struct IntLongPair {
    int i = 0;
    long l = 0;
};

FloatDoublePair p1() {
    // Variables for the machine precision.
    float eps_f = 1.0f;
    float prev_eps_f = 0.0f;
    double eps_d = 1.0;
    double prev_eps_d = 0.0;
    
    // Find precision for floats.
    while(1.0f + eps_f != 1.0f) {
        prev_eps_f = eps_f;
        eps_f /= 2.0f;
    }
    
    // Find precision for doubles.
    while(1.0 + eps_d != 1.0) {
        prev_eps_d = eps_d;
        eps_d /= 2.0;
    }

    FloatDoublePair output; 
    output.f = prev_eps_f;
    output.d = prev_eps_d;

    // Print out.
    cout << scientific << setprecision(10);
    cout << "Float epsilon:  " << prev_eps_f << endl;
    cout << "Double epsilon: " << prev_eps_d << endl;

    return output;
}

IntLongPair p3() {
    // 
    int i = 200 * 300 * 400 * 500;
    long l = 200L * 300L * 400L * 500L;
    IntLongPair output;
    output.i = i;
    output.l = l;

    // Print out.
    cout << scientific << setprecision(10);
    cout << "int value:  " << to_string(i) << endl;
    cout << "long value: " << to_string(l) << endl;

    return output;
}

int p4() {
    unsigned int counter = 0;
    for(int i = 0; i < 3; ++i) --counter;
    cout << "Counter: " << to_string(counter) << endl;
    return counter;
}

int main() {
    FloatDoublePair output_1 = p1();
    IntLongPair output_3 = p3();
    int output_4 = p4();
    
    return 0;
}
