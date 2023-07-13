#include "cformer.h"

// static tensor& forward(tensor &x)
// {
//     static tensor w(af::constant(2, 1, 3));
//     static tensor b(af::constant(3, 1, 3));

//     return w * x + b;
// }

int main(int argc, char* argv[])
{
    tensor a(af::constant(1, 1, 3));
    tensor b(af::constant(2, 3, 3));
    tensor c(af::constant(3, 1, 3));
    // tensor d(af::constant(4, 1, 3));
    tensor &z = a.matmul(b);

    z += c;
    z.backward();
    af_print(z.data);
    af_print(a.grad);
    af_print(b.grad);
    af_print(c.grad);
    // af_print(d.grad);
    z.print_graph();
    z.destroy_graph();
    return 0;
}
