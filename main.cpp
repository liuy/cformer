#include "cformer.h"

tensor& add(tensor &a, tensor &b)
{
    return a + b;
}

int main(int argc, char* argv[])
{
    tensor a(af::constant(1, 1, 3));
    tensor b(af::constant(2, 1, 3));
    tensor c(af::constant(3, 1, 3));
    // tensor d(af::constant(4, 1, 3));
    tensor z = a + b;

    for (int i = 0; i < 2; i++)
        z = add(a, b);
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
