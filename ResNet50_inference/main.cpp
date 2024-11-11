#include <torch/script.h> // One-stop header for TorchScript
#include <torch/torch.h>  // Main libtorch header
#include <iostream>       // For std::cerr and std::cout
#include <memory>         // For std::shared_ptr
#include <vector>         // For std::vector
#include <chrono>         // For timing
#include <fstream>        // For file I/O (if needed)

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: inference <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1], torch::kCPU); // Explicitly load to CPU
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.what() << "\n";
    return -1;
  }
        // Move model to GPU if available
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Using GPU for inference.\n";
    module.to(torch::kCUDA);
  }
  else {
    std::cout << "CUDA not available. Using CPU for inference.\n";
  }

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::cout << "ok\n";
}


////// Approach 1: loading image and preprocess images here

// #include <torch/script.h> // One-stop header for TorchScript
// #include <torch/torch.h>  // Main libtorch header
// #include <iostream>       // For std::cerr and std::cout
// #include <memory>         // For std::shared_ptr
// #include <vector>         // For std::vector
// #include <chrono>         // For timing
// #include <fstream>        // For file I/O
// #include <string>         // For std::string
// #include <sstream>        // For std::stringstream
// #include <iomanip>        // For std::setprecision

// // Include stb_image for image loading
// #define STB_IMAGE_IMPLEMENTATION
// #include "include/stb_image.h"

// // Function to load and preprocess image
// torch::Tensor preprocess_image(const std::string& image_path) {
//     int width, height, channels;
//     unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
//     if (img == nullptr) {
//         std::cerr << "Error loading image: " << image_path << "\n";
//         // Return a tensor filled with zeros if image loading fails
//         return torch::zeros({1, 3, 224, 224});
//     }

//     // Convert to float tensor
//     torch::Tensor tensor_image = torch::from_blob(img, {height, width, 3}, torch::kUInt8);
//     tensor_image = tensor_image.permute({2, 0, 1}); // Convert to CxHxW
//     tensor_image = tensor_image.to(torch::kFloat32) / 255.0;

//     // Normalize using ImageNet means and stds
//     tensor_image[0] = (tensor_image[0] - 0.485) / 0.229;
//     tensor_image[1] = (tensor_image[1] - 0.456) / 0.224;
//     tensor_image[2] = (tensor_image[2] - 0.406) / 0.225;

//     // Resize to 224x224 using bilinear interpolation
//     tensor_image = torch::nn::functional::interpolate(
//         tensor_image.unsqueeze(0),
//         torch::nn::functional::InterpolateFuncOptions()
//             .size(std::vector<int64_t>{224, 224})
//             .mode(torch::kBilinear)
//             .align_corners(false)
//     ).squeeze(0);

//     stbi_image_free(img); // Free the image memory

//     return tensor_image.unsqueeze(0); // Add batch dimension
// }

// int main(int argc, const char* argv[]) {
//     if (argc != 3) { // Expecting two arguments: model path and dataset file
//         std::cerr << "Usage: inference <path-to-exported-script-module> <path-to-dataset.csv>\n";
//         return -1;
//     }

//     std::string model_path = argv[1];
//     std::string dataset_path = argv[2];

//     // Load the TorchScript model
//     torch::jit::script::Module module;
//     try {
//         module = torch::jit::load(model_path, torch::kCPU);
//     }
//     catch (const c10::Error& e) {
//         std::cerr << "Error loading the model: " << e.what() << "\n";
//         return -1;
//     }

//     std::cout << "Model loaded successfully.\n";

//     // Move model to GPU if available
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA is available! Using GPU for inference.\n";
//         module.to(torch::kCUDA);
//     }
//     else {
//         std::cout << "CUDA not available. Using CPU for inference.\n";
//     }

//     // Load the dataset
//     std::ifstream dataset_file(dataset_path);
//     if (!dataset_file.is_open()) {
//         std::cerr << "Unable to open dataset file: " << dataset_path << "\n";
//         return -1;
//     }

//     std::vector<std::pair<std::string, int>> dataset;
//     std::string line;

//     // Read the header line and ignore it
//     if (!std::getline(dataset_file, line)) {
//         std::cerr << "Dataset file is empty.\n";
//         return -1;
//     }

//     while (std::getline(dataset_file, line)) {
//         std::stringstream ss(line);
//         std::string image_path;
//         std::string label_str;

//         if (!std::getline(ss, image_path, ',')) {
//             std::cerr << "Invalid line in dataset file: " << line << "\n";
//             continue;
//         }

//         if (!std::getline(ss, label_str, ',')) {
//             std::cerr << "Invalid line in dataset file: " << line << "\n";
//             continue;
//         }

//         int label = std::stoi(label_str);
//         dataset.emplace_back(image_path, label);
//     }
//     dataset_file.close();

//     if (dataset.empty()) {
//         std::cerr << "Dataset is empty. Please check the dataset file.\n";
//         return -1;
//     }

//     std::cout << "Loaded " << dataset.size() << " samples from the dataset.\n";

//     // Initialize counters
//     int correct = 0;
//     int total = 0;
//     double total_time = 0.0; // in milliseconds

//     // Iterate over the dataset
//     for (const auto& sample : dataset) {
//         std::string image_path = sample.first;
//         int true_label = sample.second;

//         // Preprocess the image
//         torch::Tensor input = preprocess_image(image_path);
//         if (torch::cuda::is_available()) {
//             input = input.to(torch::kCUDA);
//         }

//         // Start timer
//         auto start = std::chrono::high_resolution_clock::now();

//         // Inference
//         std::vector<torch::jit::IValue> inputs;
//         inputs.emplace_back(input);
//         at::Tensor output;
//         try {
//             output = module.forward(inputs).toTensor();
//         }
//         catch (const c10::Error& e) {
//             std::cerr << "Error during model inference: " << e.what() << "\n";
//             continue;
//         }

//         // End timer
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> elapsed = end - start;
//         total_time += elapsed.count();

//         // Get prediction
//         torch::Tensor prediction = output.argmax(1);
//         int predicted_label = prediction.item<int>();

//         // Compare with true label
//         if (predicted_label == true_label) {
//             correct++;
//         }
//         total++;

//         // Optional: Print progress every 100 samples
//         if (total % 100 == 0) {
//             std::cout << "Processed " << total << " samples.\n";
//         }
//     }

//     // Calculate accuracy and average latency
//     double accuracy = static_cast<double>(correct) / total * 100.0;
//     double avg_latency = total_time / total; // milliseconds per inference

//     // Output results
//     std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << ")\n";
//     std::cout << "Average Inference Latency: " << std::fixed << std::setprecision(2) << avg_latency << " ms per iteration\n";

//     // Optional: Save detailed results to a CSV file
//     std::ofstream results_file("inference_results.csv");
//     if (results_file.is_open()) {
//         results_file << "Total Samples,Correct Predictions,Accuracy (%),Average Latency (ms)\n";
//         results_file << total << "," << correct << "," << accuracy << "," << avg_latency << "\n";
//         results_file.close();
//         std::cout << "Results saved to inference_results.csv\n";
//     }
//     else {
//         std::cerr << "Unable to open results file for writing.\n";
//     }

//     return 0;
// }




////// Approach 2: loading already preprocessed and serialized batch data

// #include <torch/script.h>
// #include <torch/torch.h>
// #include <iostream>
// #include <memory>
// #include <vector>
// #include <chrono>
// #include <fstream>
// #include <string>
// #include <sstream>
// #include <iomanip>
// #include <filesystem>

// torch::Tensor load_tensor(const std::string& path) {
//     torch::Tensor tensor;
//     try {
//         torch::load(tensor, path); // Correct usage
//     }
//     catch (const c10::Error& e) {
//         std::cerr << "Error loading tensor from " << path << ": " << e.what() << "\n";
//     }
//     return tensor;
// }

// int main(int argc, const char* argv[]) {
//     if (argc != 3) {
//         std::cerr << "Usage: inference <path-to-exported-script-module> <path-to-serialized-batches-directory>\n";
//         return -1;
//     }

//     std::string model_path = argv[1];
//     std::string data_dir = argv[2];

//     // Load the TorchScript model
//     torch::jit::script::Module module;
//     try {
//         module = torch::jit::load(model_path, torch::kCPU);
//     }
//     catch (const c10::Error& e) {
//         std::cerr << "Error loading the model: " << e.what() << "\n";
//         return -1;
//     }

//     std::cout << "Model loaded successfully.\n";

//     // Move model to GPU if available
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA is available! Using GPU for inference.\n";
//         module.to(torch::kCUDA);
//     }
//     else {
//         std::cout << "CUDA not available. Using CPU for inference.\n";
//     }

//     // Iterate over serialized batch files
//     int correct = 0;
//     int total = 0;
//     double total_time = 0.0; // in milliseconds

//     for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
//         if (entry.is_regular_file() && entry.path().extension() == ".pt") {
//             std::string file_path = entry.path().string();

//             if (file_path.find("data_batch") != std::string::npos) {
//                 // Extract corresponding label batch file
//                 std::string label_file = file_path;
//                 label_file.replace(label_file.find("data_batch"), 10, "label_batch");

//                 // Load batch tensors
//                 torch::Tensor input_batch = load_tensor(file_path);
//                 torch::Tensor label_batch = load_tensor(label_file);

//                 if (!input_batch.defined() || !label_batch.defined()) {
//                     std::cerr << "Failed to load data or label batch for " << file_path << "\n";
//                     continue;
//                 }

//                 if (torch::cuda::is_available()) {
//                     input_batch = input_batch.to(torch::kCUDA);
//                 }

//                 // Start timer
//                 auto start = std::chrono::high_resolution_clock::now();

//                 // Inference
//                 std::vector<torch::jit::IValue> inputs_vec;
//                 inputs_vec.emplace_back(input_batch);
//                 at::Tensor output_batch;
//                 try {
//                     output_batch = module.forward(inputs_vec).toTensor();
//                 }
//                 catch (const c10::Error& e) {
//                     std::cerr << "Error during model inference: " << e.what() << "\n";
//                     continue;
//                 }

//                 // End timer
//                 auto end = std::chrono::high_resolution_clock::now();
//                 std::chrono::duration<double, std::milli> elapsed = end - start;
//                 total_time += elapsed.count();

//                 // Get predictions
//                 torch::Tensor predictions = output_batch.argmax(1);

//                 // Compare with true labels
//                 for (int i = 0; i < predictions.size(0); ++i) {
//                     int predicted_label = predictions[i].item<int>();
//                     int true_label = label_batch[i].item<int>();
//                     if (predicted_label == true_label) {
//                         correct++;
//                     }
//                     total++;
//                 }

//                 // Optional: Print progress
//                 if (total % 1000 == 0) {
//                     std::cout << "Processed " << total << " samples.\n";
//                 }
//             }
//         }
//     }

//     // Calculate accuracy and average latency
//     double accuracy = static_cast<double>(correct) / total * 100.0;
//     double avg_latency = total_time / total; // milliseconds per inference

//     // Output results
//     std::cout << "Accuracy: " << accuracy << "% (" << correct << "/" << total << ")\n";
//     std::cout << "Average Inference Latency: " << std::fixed << std::setprecision(2) << avg_latency << " ms per iteration\n";

//     // Optional: Save detailed results to a CSV file
//     std::ofstream results_file("inference_results.csv");
//     if (results_file.is_open()) {
//         results_file << "Total Samples,Correct Predictions,Accuracy (%),Average Latency (ms)\n";
//         results_file << total << "," << correct << "," << accuracy << "," << avg_latency << "\n";
//         results_file.close();
//         std::cout << "Results saved to inference_results.csv\n";
//     }
//     else {
//         std::cerr << "Unable to open results file for writing.\n";
//     }

//     return 0;
// }