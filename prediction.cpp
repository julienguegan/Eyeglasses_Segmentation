#include <iostream>
#include <stdio.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// Show OpenCV image
void show_image(cv::Mat& img, std::string title) {
    cv::namedWindow(title, cv::WINDOW_NORMAL); 
    cv::imshow(title, img);
    cv::waitKey(0);
}

// normalize tensor (require for model prediction)
auto normalize(at::Tensor tensor) {
    at::Tensor mean = torch::tensor({0.485, 0.456, 0.406});
    at::Tensor std = torch::tensor({0.229, 0.224, 0.225});
    tensor = (tensor - mean) / std ;
    return tensor;
}

// transpose tensor (require for model prediction)
auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 }) {
    tensor = tensor.permute(dims);
    return tensor;
}

// convert cv::image to torch::tensor
auto ToTensor(cv::Mat img) {
    at::Tensor tensor_image = torch::from_blob(img.data, { img.cols, img.rows, 3 }, at::kByte);
    return tensor_image;
}

// convert Tensor to jit format for model
auto ToInput(at::Tensor tensor_image) {
    return std::vector<torch::jit::IValue>{tensor_image};
}

// convert torch::tensor to cv::image 
auto ToCvImage(at::Tensor tensor, int n_channel=3) {
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    //tensor = tensor.squeeze().detach().mul(255).clamp(0, 255);
    tensor = tensor.to(torch::kU8);
    uchar type_color;
    if (n_channel == 3)
        type_color = CV_8UC3;
    else
        type_color = CV_8UC1;
    try {
        cv::Mat output_mat(cv::Size(width,height), type_color, tensor.data_ptr()); // initialize a Mat object <uchar>
        return output_mat.clone();
    }
    catch (const c10::Error& e) {
        std::cout << "Error OpenCV conversion : """ << e.msg() << "" << std::endl;
        return cv::Mat(height, width, CV_8UC1);
    }
}

int main(int argc, const char* argv[]) {

    // Load the model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("C:\\Users\\gueganj\\source\\repos\\essaiPascal_2\\x64\\Release\\frame_model.ckpt");
        std::cout << "Model loaded !\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return 0;
    }

    // Load image with OpenCV (image size should be multiple of 32 for the model due to architecture)
    cv::Mat image = cv::imread("C:\\Users\\gueganj\\source\\repos\\essaiPascal_2\\x64\\Release\\image.jpg", cv::ImreadModes::IMREAD_UNCHANGED);
    int width     = image.size().width - image.size().width % 32;
    int height    = image.size().height - image.size().height % 32;
    cv::resize(image, image, cv::Size(960,544));

    // Convert the image into tensor and format it for the model
    auto input_tensor = ToTensor(image).toType(c10::kFloat);
    input_tensor      = normalize(input_tensor.div(255));
    input_tensor      = transpose(input_tensor, { (2),(0),(1) }).unsqueeze_(0);
    auto input_to_net = ToInput(input_tensor);
    cv::Mat test2     = ToCvImage(transpose(input_tensor.squeeze(), { (1),(2),(0) }).mul(255).clamp(0, 255));
    show_image(test2, "test image");

    // Run the model
    std::cout << "Prediction ..." << std::endl;
    at::Tensor prediction = model.forward(input_to_net).toTensor();
    std::cout << input_tensor.max() << std::endl;
    
    // Display and Save prediction
    prediction = prediction.squeeze();// .unsqueeze(2).expand({ -1, -1, 3 });
    cv::Mat image_prediction = ToCvImage(prediction.mul(255).clamp(0, 255), 1);
    show_image(image_prediction, "prediction image");
    cv::imwrite("prediction.jpg", image_prediction);
    std::cout << "image saved !" << std::endl;
}