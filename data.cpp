#include "cformer.h"

static array read_mnist_images(const std::string& path, size_t &nrow, size_t &ncol)
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

    nrow = num_rows;
    ncol = num_cols;
    // array is column-majored, so we need to transpose it as layers expect row-majored array
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

// Note array is column-majored, so we need to transpose it before calling this function
void write_mnist_images(const array &a, const std::string& path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
        panic("Cannot open file: %s", path.c_str());
    std::vector<uint8_t> images(a.elements() + 16); // header has 16 bytes
    (a * 255).as(u8).host(images.data() + 16);
    write_data(file, images.data(), images.size());
}

void mnist_reader(struct data &d)
{
    size_t rows, cols;
    array a = read_mnist_images("data/mnist/train-images-idx3-ubyte", d.nrow, d.ncol);
    d.train_x.init(a);
    a = read_mnist_labels("data/mnist/train-labels-idx1-ubyte");
    d.train_y.init(a);
    a = read_mnist_images("data/mnist/t10k-images-idx3-ubyte", rows, cols);
    d.test_x.init(a);
    a = read_mnist_labels("data/mnist/t10k-labels-idx1-ubyte");
    d.test_y.init(a);
}

/**
 * af::rotate() expects images with shape (width, height, batch_size).
 * Tensor is shaped as (batch_size, width*height), so we need to reshape it to
 * (width, height, batch_size). After calling rotate, reshape it back to
 * (batch_size, width*height).
 */
array random_rotate::operator()(const array &x, struct data &d)
{
#define PIf		3.14159265358979323846f
    size_t batch_size = x.dims(0) * ratio;
    array r(batch_size, d.nrow * d.ncol);
    for (size_t i = 0; i < batch_size; i++) {
        array img = af::moddims(x.row(i), d.nrow, d.ncol);
        float angle = random(-degree, degree) * PIf / 180.0f;
        r.row(i) = af::moddims(af::rotate(img, angle), 1, d.nrow * d.ncol);
    }
    //write_mnist_images(r.T(), "mnist_rr_images");
    return r;
}

array elastic_transform::operator()(const array &x, struct data &d)
{
    size_t batch_size = x.dims(0) * ratio;
    array r(batch_size, d.nrow * d.ncol);
    for (size_t i = 0; i < batch_size; i++) {
        array img = af::moddims(x.row(i), d.nrow, d.ncol);

        // Generate random displacement fields
        af::array x_disp = af::randn(d.nrow, d.ncol) * alpha;
        af::array y_disp = af::randn(d.nrow, d.ncol) * alpha;

        // Smooth displacement fields
        af::array kernel = af::gaussianKernel(6, 6, sigma, sigma);
        x_disp = af::convolve(x_disp, kernel);
        y_disp = af::convolve(y_disp, kernel);

        // Generate grid coordinates
        af::array x_coords = af::tile(af::range(d.nrow), 1, d.ncol);
        af::array y_coords = af::tile(af::range(d.ncol), 1, d.nrow).T();

        // Add displacement fields to grid coordinates
        x_coords += x_disp;
        y_coords += y_disp;

        // Interpolate input using grid coordinates
        r.row(i) = af::moddims(af::approx2(img, x_coords, y_coords, AF_INTERP_BILINEAR), 1, d.nrow * d.ncol);
    }
    //write_mnist_images(r.T(), "mnist_et_images");
    return r;
}

void data::load(std::initializer_list<transform *> transforms)
{
    printf("Loading data...");
    af::timer::start();
    data_reader(*this);
    array joint_x = train_x.data;
    array joint_y = train_y.data;
    for (auto tf : transforms) {
        array new_x = (*tf)(train_x.data, *this);
        joint_x = af::join(0, joint_x, new_x);
        joint_y = af::join(0, train_y.data, joint_y);
        delete tf;
    }
    train_x.init(joint_x);
    train_y.init(joint_y);

    printf("%lld training samples, %lld test samples (Used %.1fs)\n",
           train_x.data.dims(0), test_x.data.dims(0), af::timer::stop());
}

void data::init_train_idx(size_t batch_size)
{
    train_idx.resize(DIV_ROUND_UP(train_x.data.dims(0), batch_size));
    std::iota(train_idx.begin(), train_idx.end(), 0);
}

void data::get_mini_batch(tensor &x, tensor &y, size_t idx, size_t batch_size)
{
    size_t start = idx * batch_size;
    size_t end = MIN(start + batch_size, train_x.data.dims(0));
    x.init(train_x.data.rows(start, end - 1));
    y.init(train_y.data.rows(start, end - 1));
}
