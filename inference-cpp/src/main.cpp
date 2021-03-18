//------------------------------------------------------------------------
// Project     : esc-app
// Filename    : main.cpp
// Created by  : lhannink, 01/2021
// Description : loading a pytorch model and running inference
//------------------------------------------------------------------------

#include <chrono>
#include <iostream>
#include <sstream>

#include <torch/torch.h>
#include <torch/script.h>

#include "vocab-gen.h"

using Clock = std::chrono::high_resolution_clock;

//------------------------------------------------------------------------
void print_usage ()
{
	std::cerr
		<< "Usage:\n"
		<< "\tesc-app <module-path> <inputs-path> [<outputs-path>]\n\n"
		<< "Arguments:\n"
		<< "\t module-path : path to TorchScript module to run\n"
		<< "\t inputs-path : path to TorchScript module containing one or more named buffers\n"
		<< "\t               of identical shape. These buffers will be stacked into a single tensor\n"
		<< "\t               that will be passed to the model's forward() method.\n"
		<< "\toutputs-path : (optional) where to save the model output. E.g., this can be \n"
		<< "\t               loaded again in Python by running\n"
		<< "\t               >>> import torch\n"
		<< "\t               >>> output_module = torch.load('path/to/outputs.pt')\n"
		<< "\t               >>> outputs = next(output_module.parameters())\n\n";
}

//------------------------------------------------------------------------
bool load_module (const std::string& filename, torch::jit::script::Module& module)
{
	std::cout << "Loading TorchScript module from '" << filename << "'...";
	try
	{
		module = torch::jit::load (filename);
		std::cout << "done.\n";
		return true;
	}
	catch (const c10::Error& e)
	{
		std::cerr << "error\n";
		return false;
	}
}

//------------------------------------------------------------------------
std::string get_shape (const at::Tensor& t)
{	std::stringstream shape;
	shape << "[";
	for (int d{ 0 }; d < t.ndimension () - 1; ++d)
		shape << t.size(d) << ", ";
	shape << t.size (t.ndimension () - 1) << "]";
	return shape.str();
}

//------------------------------------------------------------------------
int with_error_msg(const std::string& msg)
{
	std::cerr << msg << "\n";
	return -1;
}

//------------------------------------------------------------------------
int main (int argc, const char* argv[])
{
	std::cout << "==========================================\n";
	std::cout << "===   Environmental Sound Classifier   ===\n";
	std::cout << "==========================================\n\n";
	if (argc < 2)
	{
		print_usage();
		return -1;
	}

	// deserialize the model script module from file
	torch::jit::script::Module model;
	if (!load_module(argv[1], model))
		return with_error_msg("failed");

	if (argc < 3)
		return 0;

	// deserialize the input container script module from file
	torch::jit::script::Module input_container;
	if (!load_module(argv[2], input_container))
		return with_error_msg("loading of module failed");

	// aggregate all named buffers from <input_container> into a single tensor
	std::cout << "\nInspecting '" << argv[2] << "' for named buffers.\n";
	auto n_samples = static_cast<int> (input_container.named_buffers().size ());
	auto inputs = torch::zeros ({n_samples, 3, 128, 157});
	std::vector<std::string> labels;

	int idx = 0;
	for (const auto& named_buffer : input_container.named_buffers())
	{
		labels.push_back (named_buffer.name);

		std::cout << "  - found buffer '" << named_buffer.name << "' with shape " << get_shape(named_buffer.value) << "\n";

		// copy buffer into stacked <inputs> tensor
		inputs[idx++] = named_buffer.value;
	}

	std::cout << "\nRunning inference using stacked input tensor with shape " << get_shape(inputs) << "...";
	const auto start = Clock::now();

	// run inference step by calling forward()
	const auto outputs = model.forward ({inputs}).toTensor ();

	const auto duration = std::chrono::duration<double, std::milli> (Clock::now () - start);
	std::cout << "took " << duration.count () << " ms.\n";

	// the output is a matrix of shape [n_samples, 50]
	// each row contains an unnormalized score for every of our categories
	// to convert this into probabilities (s.t. row.sum() == 1.0), we compute
	// the softmax across the 1st dimension
	auto out_softmax = outputs.softmax(/* dim = */ 1);

	// for each sample (== row), we compute the max value and its index 
	torch::Tensor values;
	torch::Tensor indices;
	std::tie (values, indices) = out_softmax.max (/* dim = */ 1);

	const auto* probs = values.data_ptr<float> ();
	const auto* idcs = indices.data_ptr<int64_t> ();

	// print sample labels vs model predictions
	std::cout << "\nModel Predictions:\n";
	for (int i {0}; i < n_samples; ++i)
	{
		std::cout << "  - '" << labels[i] << "' classified as '" 
		          << vocab[idcs[i]]
		          << "' (p=" << std::setprecision(2) << probs[i] << ")"
		          << "\n";

	}

	// save output tensor for further processing back in python
	if (argc >= 4)
	{
		std::cout << "\nSaving output tensor to '" << argv[3] << "'.";
		torch::save (outputs, argv[3]);
	}

	return 0;
}
