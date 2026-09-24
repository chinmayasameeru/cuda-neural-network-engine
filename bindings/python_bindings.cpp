// CUDA Neural Network Engine — Python Bindings v0.2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "model.h"
#include "layers.h"
#include "advanced_layers.h"
#include "schedulers.h"

namespace py = pybind11;
using namespace cnn;

static TensorPtr numpy_to_tensor(py::array_t<float> arr, bool requires_grad = false) {
    auto buf = arr.request();
    std::vector<int> shape;
    for (int i = 0; i < (int)buf.ndim; ++i) shape.push_back((int)buf.shape[i]);

    auto t = std::make_shared<Tensor>(shape, requires_grad);
    auto host = std::vector<float>((float*)buf.ptr, (float*)buf.ptr + t->numel);
    t->from_cpu(host);
    return t;
}

static py::array_t<float> tensor_to_numpy(TensorPtr t) {
    auto host = t->to_cpu();
    std::vector<ssize_t> shape(t->shape.begin(), t->shape.end());
    return py::array_t<float>(shape, host.data());
}

class PySequential {
public:
    PySequential() = default;

    void add_layer(const std::string& type, const std::vector<int>& args) {
        if (type == "Linear") {
            if (args.size() >= 2) model_.add<Linear>(args[0], args[1], args.size() > 2 ? args[2] : 1);
        } else if (type == "Conv2D") {
            if (args.size() >= 3) model_.add<Conv2D>(args[0], args[1], args[2],
                args.size() > 3 ? args[3] : 1, args.size() > 4 ? args[4] : 0,
                args.size() > 5 ? args[5] : 1);
        } else if (type == "BatchNorm2D") {
            if (args.size() >= 1) model_.add<BatchNorm2D>(args[0]);
        } else if (type == "MaxPool2D") {
            if (args.size() >= 1) model_.add<MaxPool2D>(args[0], args.size() > 1 ? args[1] : args[0]);
        } else if (type == "AdaptiveAvgPool2D") {
            if (args.size() >= 2) model_.add<AdaptiveAvgPool2D>(args[0], args[1]);
        } else if (type == "ReLU") {
            model_.add<ReLU>();
        } else if (type == "Sigmoid") {
            model_.add<Sigmoid>();
        } else if (type == "Tanh") {
            model_.add<Tanh>();
        } else if (type == "LeakyReLU") {
            model_.add<LeakyReLU>(args.size() > 0 ? *(float*)&args[0] : 0.01f);
        } else if (type == "GELU") {
            model_.add<GELU>();
        } else if (type == "SiLU") {
            model_.add<SiLU>();
        } else if (type == "Softmax") {
            model_.add<Softmax>();
        } else if (type == "Flatten") {
            model_.add<Flatten>();
        } else if (type == "Dropout") {
            model_.add<Dropout>(args.size() > 0 ? *(float*)&args[0] : 0.5f);
        } else if (type == "ResidualBlock") {
            if (args.size() >= 2) model_.add<ResidualBlock>(args[0], args[1], args.size() > 2 ? args[2] : 1);
        } else if (type == "LSTMCell") {
            if (args.size() >= 2) model_.add<LSTMCell>(args[0], args[1]);
        } else if (type == "GRUCell") {
            if (args.size() >= 2) model_.add<GRUCell>(args[0], args[1]);
        } else if (type == "LayerNorm") {
            if (args.size() >= 1) model_.add<LayerNorm>(args[0]);
        } else if (type == "MultiHeadAttention") {
            if (args.size() >= 2) model_.add<MultiHeadAttention>(args[0], args[1]);
        }
    }

    py::array_t<float> forward(py::array_t<float> input) {
        auto inp = numpy_to_tensor(input, false);
        auto out = model_.forward(inp);
        return tensor_to_numpy(out);
    }

    void zero_grad() { model_.zero_grad(); }
    void train() { model_.train(); }
    void eval() { model_.eval(); }

    std::string summary() const {
        std::string s = "Sequential Model:\n";
        s += "Layers: " + std::to_string(model_.size()) + "\n";
        for (size_t i = 0; i < model_.size(); ++i) {
            s += "  [" + std::to_string(i) + "] " + model_[i]->name() + "\n";
        }
        return s;
    }

private:
    Sequential model_;
};

PYBIND11_MODULE(cnn_engine, m) {
    m.doc() = "CUDA Neural Network Engine v0.2.0";

    py::class_<PySequential>(m, "Sequential")
        .def(py::init<>())
        .def("add", &PySequential::add_layer)
        .def("forward", &PySequential::forward)
        .def("zero_grad", &PySequential::zero_grad)
        .def("train", &PySequential::train)
        .def("eval", &PySequential::eval)
        .def("summary", &PySequential::summary);

    // Basic layers
    py::class_<Linear, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, bool>(), py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true);

    py::class_<Conv2D, std::shared_ptr<Conv2D>>(m, "Conv2D")
        .def(py::init<int, int, int, int, int, bool>(), py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"), py::arg("stride") = 1, py::arg("padding") = 0, py::arg("bias") = true);

    py::class_<BatchNorm2D, std::shared_ptr<BatchNorm2D>>(m, "BatchNorm2D")
        .def(py::init<int, float, float>(), py::arg("num_features"), py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f);

    py::class_<MaxPool2D, std::shared_ptr<MaxPool2D>>(m, "MaxPool2D")
        .def(py::init<int, int>(), py::arg("pool_size"), py::arg("stride") = -1);

    py::class_<AdaptiveAvgPool2D, std::shared_ptr<AdaptiveAvgPool2D>>(m, "AdaptiveAvgPool2D")
        .def(py::init<int, int>(), py::arg("output_h"), py::arg("output_w"));

    py::class_<ReLU, std::shared_ptr<ReLU>>(m, "ReLU").def(py::init<>());
    py::class_<Sigmoid, std::shared_ptr<Sigmoid>>(m, "Sigmoid").def(py::init<>());
    py::class_<Tanh, std::shared_ptr<Tanh>>(m, "Tanh").def(py::init<>());
    py::class_<LeakyReLU, std::shared_ptr<LeakyReLU>>(m, "LeakyReLU").def(py::init<float>(), py::arg("negative_slope") = 0.01f);
    py::class_<GELU, std::shared_ptr<GELU>>(m, "GELU").def(py::init<>());
    py::class_<SiLU, std::shared_ptr<SiLU>>(m, "SiLU").def(py::init<>());
    py::class_<Softmax, std::shared_ptr<Softmax>>(m, "Softmax").def(py::init<>());
    py::class_<Flatten, std::shared_ptr<Flatten>>(m, "Flatten").def(py::init<>());
    py::class_<Dropout, std::shared_ptr<Dropout>>(m, "Dropout").def(py::init<float>(), py::arg("p") = 0.5f);

    // Advanced layers
    py::class_<ResidualBlock, std::shared_ptr<ResidualBlock>>(m, "ResidualBlock")
        .def(py::init<int, int, int>(), py::arg("in_channels"), py::arg("out_channels"), py::arg("stride") = 1);

    py::class_<LSTMCell, std::shared_ptr<LSTMCell>>(m, "LSTMCell")
        .def(py::init<int, int>(), py::arg("input_size"), py::arg("hidden_size"))
        .def("reset_state", &LSTMCell::reset_state);

    py::class_<GRUCell, std::shared_ptr<GRUCell>>(m, "GRUCell")
        .def(py::init<int, int>(), py::arg("input_size"), py::arg("hidden_size"))
        .def("reset_state", &GRUCell::reset_state);

    py::class_<LayerNorm, std::shared_ptr<LayerNorm>>(m, "LayerNorm")
        .def(py::init<int, float>(), py::arg("normalized_shape"), py::arg("eps") = 1e-5f);

    py::class_<MultiHeadAttention, std::shared_ptr<MultiHeadAttention>>(m, "MultiHeadAttention")
        .def(py::init<int, int, float>(), py::arg("embed_dim"), py::arg("num_heads"), py::arg("dropout") = 0.0f);

    // Optimizers
    py::class_<SGD, std::shared_ptr<SGD>>(m, "SGD")
        .def(py::init<float, float, float>(), py::arg("lr") = 0.01f, py::arg("momentum") = 0.0f, py::arg("weight_decay") = 0.0f)
        .def("step", &SGD::step).def("zero_grad", &SGD::zero_grad);

    py::class_<Adam, std::shared_ptr<Adam>>(m, "Adam")
        .def(py::init<float, float, float, float, float>(), py::arg("lr") = 0.001f, py::arg("beta1") = 0.9f, py::arg("beta2") = 0.999f, py::arg("eps") = 1e-8f, py::arg("weight_decay") = 0.0f)
        .def("step", &Adam::step).def("zero_grad", &Adam::zero_grad);

    // Loss functions
    py::class_<CrossEntropyLoss, std::shared_ptr<CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<>()).def("compute", &CrossEntropyLoss::compute);

    py::class_<MSELoss, std::shared_ptr<MSELoss>>(m, "MSELoss")
        .def(py::init<>()).def("compute", &MSELoss::compute);

    // Schedulers
    py::class_<StepLR, std::shared_ptr<StepLR>>(m, "StepLR")
        .def(py::init<float, int, float>(), py::arg("initial_lr"), py::arg("step_size"), py::arg("gamma") = 0.1f);

    py::class_<CosineAnnealingLR, std::shared_ptr<CosineAnnealingLR>>(m, "CosineAnnealingLR")
        .def(py::init<float, int, float>(), py::arg("initial_lr"), py::arg("T_max"), py::arg("eta_min") = 0.0f);

    m.attr("__version__") = "0.2.0";
}
