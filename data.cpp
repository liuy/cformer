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

void mnist_reader(tensor &tr_input, tensor &tr_label, tensor &ts_input, tensor &ts_label)
{
    array a = read_mnist_images("data/mnist/train-images-idx3-ubyte");
    tr_input.assign_data(a);
    a = read_mnist_labels("data/mnist/train-labels-idx1-ubyte");
    tr_label.assign_data(a);
    a = read_mnist_images("data/mnist/t10k-images-idx3-ubyte");
    ts_input.assign_data(a);
    a = read_mnist_labels("data/mnist/t10k-labels-idx1-ubyte");
    ts_label.assign_data(a);
}

