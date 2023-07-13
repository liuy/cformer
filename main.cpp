#include "cformer.h"

static array read_mnist_images(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        panic("Cannot open file: %s", path.c_str());

    int magic_number = bswap_32(read_i32(file));
    int num_images = bswap_32(read_i32(file));
    int num_rows = bswap_32(read_i32(file));
    int num_cols = bswap_32(read_i32(file));

    cf_debug("magic_number: %d, num_images: %d, num_rows: %d, num_cols: %d\n",
        magic_number, num_images, num_rows, num_cols);

    if (magic_number != 2051)
        panic("Invalid MNIST magic number: %d", magic_number);

    std::vector<uint8_t> images(num_images * num_rows * num_cols);
    read_data(file, images.data(), images.size());

    return array(num_cols*num_rows, num_images, images.data()).T() / 255.0f;
}

static array read_mnist_labels(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        panic("Cannot open file: %s", path.c_str());

    int magic_number = bswap_32(read_i32(file));
    int num_labels = bswap_32(read_i32(file));
    if (magic_number != 2049)
        panic("Invalid MNIST magic number: %d", magic_number);

    cf_debug("magic_number: %d, num_labels: %d\n", magic_number, num_labels);
    std::vector<uint8_t> images(num_labels);
    read_data(file, images.data(), images.size());

    return onehot(array(num_labels, images.data()));
}

void read_mnist(const std::string &images, const std::string &labels, tensor &x, tensor &y)
{
    array a = read_mnist_images(images);
    x.assign_data(a);
    array b = read_mnist_labels(labels);
    y.assign_data(b);
}

int main(int argc, char* argv[])
{
    tensor train_x, train_y;
    read_mnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", train_x, train_y);
    printf("%d\n", train_y.data.type());
    af_print(train_y.data.row(0));
    af_print(onehot(array(2, {3,4}), 5));
    return 0;
}
