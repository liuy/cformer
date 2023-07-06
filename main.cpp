#include "cformer.h"

int main(int argc, char* argv[])
{
    tensor a(af::constant(1, 1, 3));
    tensor b(af::constant(2, 1, 3));
    tensor c(af::constant(3, 1, 3));
    tensor d(af::constant(4, 1, 3));
    tensor &y = (a + b) * c / d;
    y.backward();
    af_print(y.data);
    af_print(a.grad);
    af_print(b.grad);
    af_print(c.grad);
    af_print(d.grad);
    y.destroy_graph();
    return 0;
}
