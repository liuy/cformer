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

static tensor& softmax(tensor &x)
{
    return x.exp()/x.exp().bsum(1);
}

static tensor& categorical_cross_entropy(tensor &y_pred, tensor &y_true, bool from_logits=false)
{
    if (from_logits)
        return  (y_true * (y_pred.exp().bsum(1).log() - y_pred)).sum(1);
    return -(y_true*y_pred.log()).sum(1);
}

int main(int argc, char* argv[])
{
    tensor train_x, train_y;
    tensor test_x, test_y;
    read_mnist("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", train_x, train_y);
    read_mnist("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", test_x, test_y);

    seq_net model {
        new linear(28*28, 256, ReLU),
    //    new linear(256, 128, ReLU),
        new linear(256, 10, Softmax),
    };
    // tensor x(train_x.data.rows(0,9));
    // tensor y(train_y.data.rows(0,9));
    model.train(train_x, train_y, 0.005, 100, 10);
    tensor tx(test_x.data.rows(0, 20));
    tensor ty(test_y.data.rows(0, 20));
    tensor &y_pred = model(tx);
    af_print(argmax(y_pred.data).T());
    af_print(argmax(ty.data).T());
    // tensor y_true(array({3,2}, {0.f,1.f,0.f, 0.f,0.f,1.f}).T());
    // tensor y_logs(array({3,2}, {0.05f,0.95f,0.f, 0.1f,0.8f,0.1f}).T());

    // af_print(y_true.data);
    // tensor &t = softmax(y_logs);
    // t.forward();
    // af_print(t.data, 10);
    // //printf("%d\n", y_true.data.type());
    // //af_print(y_logs.log().data);
    // tensor &y = categorical_cross_entropy(y_logs, y_true, true);
    // y.backward();
    // af_print(y.data, 10);

    return 0;
}
